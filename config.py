from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from loguru import logger
import os
import re
import yaml
from pathlib import Path
from logic.ocr_routing import validate_routes
from logic.menu_alias import MenuIndex, build_menu_index, load_alias_knowledge
from logic.ticket_modifiers import ModifierKnowledge, load_modifier_knowledge

# ---------- Data Classes ----------

@dataclass
class MqttCfg:
    """MQTT Broker Configuration"""
    host: str
    port: int
    base_topic: str
    session: str
    # 留 None（config.yaml 不寫這個 key）= 每個 process 自動產生唯一 id
    # （`ai-<主機後4碼>-<pid>`，見 utils/mqtt_bus.py::default_client_id）。
    # 寫死固定值只有在 App 端真的需要認 id 時才做 —— 固定值會讓「重啟時新舊
    # 行程重疊」變成 broker 層的互踢。
    client_id: Optional[str] = None

@dataclass
class ModelCfg:
    """AI Model and Text Configuration"""
    object_det: str
    ocr_det: str
    ocr_cls: str
    ocr_rec: str
    text: str

@dataclass
class CameraCfg:
    """Camera Source Configuration"""
    source: Any
    width: int
    height: int
    device: str  # 'aravis' or 'hik'


@dataclass
class DepthCamCfg:
    """
    Detph Camera Source Configuration

    現場不一定會接深度相機(目前只是先驗證硬體可行),而且驗證完要能關掉,
    不能叫客人去拔插頭 —— 所以開關放在設定檔的 depth_camera.enable。
    舊設定檔沒有這個欄位時維持原本行為(照樣嘗試啟動,沒插相機就自己
    退回 enable=False)。
    """
    cfg: dict

    @property
    def enable(self) -> bool:
        return bool(self.cfg.get("enable", True))


@dataclass
class StreamCfg:
    """Output Stream Configuration"""
    port: int
    stream_size: list

@dataclass
class RuntimeCfg:
    """Runtime configurations including model, camera, and stream"""
    model: ModelCfg
    camera: CameraCfg
    depth_cam: DepthCamCfg
    stream: StreamCfg

@dataclass
class ThresholdsCfg:
    """Thresholds for AI inference and object assignment"""
    ai_conf: float
    ai_iou: float

@dataclass
class PlacementCfg:
    """Geometric validation rules"""
    roi_strink: float
    iou_assign: float
    iou_candidate: float
    drift_iou_thresh: float

@dataclass
class StabilityCfg:
    """Configuration for temporal stability checks"""
    n_settle_frame: int    
    k_container_fail: int
    k_container_new: int
    tray_missing_frame: int
    sticker_missing_frame: int
    ticket_leave_frame: int
    # 靜止判定：相鄰兩幀中心位移超過框短邊的這個比例，就算還在動、穩定幀數歸零。
    # 0 = 關閉（只要偵測對得上就累積，舊行為）。不寫也可以。
    still_shift_ratio: float = 0.0
    # 錯誤品項確認：同一張貼紙連續這麼多次 OCR 都沒比對成功（讀到不在訂單內的品名，
    # 或讀不出菜名）才成立、才通知前端。用來擋畫面模糊/遮擋時一時看錯字造成的 NGtoOK。
    # 0 = 關閉（第一次判錯就成立、讀不出字不通知，舊行為）。不寫也可以。
    wrong_item_confirm: int = 0
    # 強證據一次成立：讀到不在訂單內的品名，且比對分數 >= strong_score、rec 信心 >= strong_conf，
    # 不必等滿 wrong_item_confirm 次。0 = 關閉。
    wrong_item_strong_score: float = 0.0
    wrong_item_strong_conf: float = 0.9

@dataclass
class ClassInfo:
    """Holds the ID and name for a single class from classes.yaml"""
    id: int
    name: str  

@dataclass
class MappingCasesCfg:
    """組合品項拆分規則：當 OCR 辨識到此品項名稱時，展開為子品項清單"""
    name: str
    sub_items: List[str]

@dataclass
class AutoRoiCfg:
    """
    單一 ROI 的自動校正參數 (該 ROI 的 source == "auto" 時才會用到)。

    做法:開機後在主迴圈裡盯 class_name 這個類別的偵測框,同一個物件連續
    confirm_frames 幀都穩定(幀間 IoU >= stable_iou)就定案,取它旋轉框的
    「外接正矩形」再向外擴 padding_ratio,寫進這個 ROI 的 points。

    class_name 刻意做成設定項而不是寫死 ticket_mat:判斷訂單結束用的實體
    標記以後可能會換(換成別的墊子、托盤、或另一個新類別),換的時候只要改
    設定檔,不用改程式。只要是偵測器認得的類別(classes.yaml 裡有的)都能用。

    刻意用正矩形而不是旋轉框:現場慣例是把 ROI 拉得比物件大一點、方向水平,
    留一點操作彈性,而不是貼著物件的斜邊。

    只在啟動時跑一次(定案或 timeout_frames 逾時就停),不會一直改 ROI;
    要重跑就發 MQTT cmd/preset_roi {"action": "auto", "name": "<roi 名稱>"}。
    """
    class_name: str = "ticket_mat"  # 拿哪個偵測類別的框當 ROI 基準
    padding_ratio: float = 0.12     # 以該物件 bbox 寬高為基準向外擴張的比例
    confirm_frames: int = 15        # 連續幾幀穩定才定案
    stable_iou: float = 0.9         # 幀間 IoU 門檻,判斷是不是同一個物件
    timeout_frames: int = 600       # 這麼多幀還沒定案就放棄,沿用設定檔原本的 points (<=0 代表不逾時)
    persist: bool = True            # 定案後是否寫回 config.yaml (讓 App 下次載入就看到現值)


@dataclass
class RoiCfg:
    """
    單一 ROI 區域的定義:四個角點的多邊形(可以是旋轉矩形),
    座標格式跟專案既有的 PolygonXYXY 一致(4 點攤平成 8 個數字),
    這樣可以直接餵給 geometry.py 裡的 iou_poly_poly / is_center_in_polygon 等工具,
    不需要為 ROI 另外寫一套幾何運算。
    """
    name: str
    # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] = 左上/右上/右下/左下,
    # 座標系 = runtime.camera.width/height 的取像解析度 (跟 App 溝通好的約定)
    points: List[List[float]]
    # 顯示顏色,跟 App 約定為 RGB [r,g,b] 0~255 (App 直接唯讀載入這個區塊)。
    # cv2 畫圖要的是 BGR,所以繪圖端請用 to_bgr(),不要直接拿 color 去畫。
    color: Optional[List[int]] = None
    # "manual" = 直接用上面的 points (人工填的,或 App 用 cmd/preset_roi 設定的)
    # "auto"   = 開機時依 auto.class_name 這個類別的偵測框自動算出 points
    # 放在每個 ROI 底下而不是 preset_roi 那一層,是因為每個 ROI 的來源本來就
    # 可以各自不同(ticket_roi 自動、packing_roi 人工),而且 App 只會送
    # name/points/color,它沒送到的 key 留在原地不受影響(見 save_preset_roi)。
    source: str = "manual"
    auto: AutoRoiCfg = field(default_factory=AutoRoiCfg)

    def to_polygon_xyxy(self) -> Tuple[float, ...]:
        """攤平成 (x1,y1,x2,y2,x3,y3,x4,y4) 供 geometry 工具直接使用"""
        flat: List[float] = []
        for pt in self.points:
            flat.extend([float(pt[0]), float(pt[1])])
        return tuple(flat)

    def to_bgr(self, default: Tuple[int, int, int] = (0, 255, 255)) -> Tuple[int, int, int]:
        """把設定檔裡的 RGB 轉成 cv2 要的 BGR;沒設定就用呼叫端給的預設色"""
        if not self.color or len(self.color) != 3:
            return default
        r, g, b = (int(c) for c in self.color)
        return (b, g, r)


@dataclass
class PresetRoiCfg:
    """
    preset_roi 模式專用的 ROI 集合。

    目前只有 ticket_roi 會影響訂單生命週期判斷(訂單是否進入/離開這個
    區域,決定要不要開始處理、要不要視為完成/結束)。其餘 ROI(辨識物件
    區域、餐品打包區域...)先預留欄位,之後要串接對應行為時,直接在
    PresetRoiTracker / TrayStateMachine 裡讀取對應欄位即可,不需要再
    回頭改這裡的結構或 load_config 的解析邏輯。
    """
    ticket_roi: RoiCfg
    object_roi: Optional[RoiCfg] = None    # 預留:辨識物件的偵測/篩選區域
    packing_roi: Optional[RoiCfg] = None   # 預留:餐品打包完成區域

    # 可被 App (cmd/preset_roi) 或自動校正改寫的 ROI 欄位名稱。
    # 新增 ROI 欄位時，把名字加到這裡就會自動被指令與存檔流程支援。
    ROI_FIELDS = ("ticket_roi", "object_roi", "packing_roi")

@dataclass
class CsvLogCfg:
    """
    訂單 CSV log 的落地位置與切檔規則。

    root_dir 刻意跟 app.py 的 self.save_dir (錄影基準路徑) 指同一個地方,
    這樣 CSV 會跟當天的影片放在同一個日期資料夾底下:
        <root_dir>/<YYYYMMDD>/<folder>/<YYYYMMDD>_<機器ID後4碼>.csv

    max_video_size: 單一份 CSV 所對應的所有影片大小上限,支援 "1GB" / "900MB"
                    這類字串或純位元組數字。留 null 代表不限制 (維持單一檔)。
    """
    root_dir: str = "/mnt/reserved/record/stream"
    folder: str = "csv_logs"
    video_category: str = "event"
    max_video_size: Optional[Any] = None


@dataclass
class DlaGuardCfg:
    """DLA 模型載入失敗時的處置（見 utils/dla_guard.py 的完整說明）。

    現場遇過開機時 APU 沒起來、DLA 模型載不進去，而且不重開機不會恢復；
    但「路徑設錯 / 檔案不存在 / 模型檔壞掉」重開一百次也沒用，所以守衛會
    先體檢檔案，只有「檔案正常卻載入失敗」才重開機。

    這一段在 config.yaml 裡可以完全不寫 —— 全部欄位都有預設值，機器上的
    config.yaml（執行期會被相機參數覆寫，算狀態）因此不需要為這個功能改動。

    enabled:          false 可完全停用自動重開，只留 log。
    max_reboots:      連續重開上限，超過就不再重開、留給人工處理。
                      任何一次全部模型載入成功就會歸零。
    delay_sec:        決定重開後、真正下指令前等幾秒（留給人 SSH 進來攔）。
    reset_after_sec:  上次重開距今超過這麼久就把計數視為過期、歸零。
    """
    enabled: bool = True
    max_reboots: int = 2
    delay_sec: int = 10
    reset_after_sec: int = 6 * 3600


@dataclass
class OrderGuardCfg:
    """訂單層級的防護機制。

    跟 dla_guard 一樣整段可以不寫，全部欄位都有預設值 —— 機器上的
    config.yaml（執行期會被改寫，算狀態）因此不需要為這個功能改動。

    timeout_sec:   一筆訂單從傳票確認（TICKET_READY）起最多進行幾秒，超過就
                   照一般收尾寫 CSV、重置、停錄影。preset_roi 模式唯一的正常出口
                   是護貝板重新露出，作業員一直沒露出時訂單永遠不會結束，之後
                   放進來的傳票全部會被當成同一筆而遺失。0 = 關閉。
    max_item_qty:  傳票上任一品項數量超過這個值就視為 OCR 讀錯，整張傳票作廢、
                   重新偵測。0 = 關閉。
    """
    timeout_sec: float = 600.0
    max_item_qty: int = 20


@dataclass
class CloudUploadCfg:
    """每日上傳到雲端（Dashboard + S3），接口是同仁的 aibox_uploader/。

    跟 dla_guard 一樣整段可以不寫，不寫就是關閉 —— 機器上的 config.yaml
    （執行期會被改寫，算狀態）因此不需要為這個功能改動。

    金鑰與伺服器憑證**不在這裡**，也不在版控裡：由人工放到
    /etc/yoshinoya_ai/cloud_upload.env，由 manager/cloud_uploader.py 自己讀取
    （見 docs/cloud_upload.md）。

    enabled:     false = 完全不碰上傳模組。
    at:          每天幾點上傳「昨天」的資料（本機時間，HH:MM）。
    window_min:  上傳時段長度。到 at + window_min 一律中止，因為之後
                 crontab 的 rclone 會把同一批資料夾搬去 OneDrive。
    store_code:  店代號；留空 = hostname 後 4 碼，跟 CsvWriter 的檔名一致。
    folders:     要上傳的媒體子資料夾（<root_dir>/<YYYYMMDD>/<folder>/）。
    module_dir:  上傳模組的位置，相對於專案根目錄。
    retry_sec:   開機時伺服器連不上，隔多久重試一次。
    """
    enabled: bool = False
    at: str = "01:00"
    window_min: int = 120
    store_code: str = ""
    folders: List[str] = field(default_factory=lambda: ["event"])
    module_dir: str = "aibox_uploader"
    retry_sec: int = 300


def _parse_hhmm(value, default: str) -> str:
    """YAML 1.1 會把沒加引號的 1:00 / 10:30 當成 60 進位整數（= 60 / 630），這裡轉回 HH:MM。
    （有前導零的 01:00 不會被轉，本來就是字串。）"""
    if value is None:
        return default
    if isinstance(value, int):
        hhmm = f"{value // 60:02d}:{value % 60:02d}"
        # 舊版 loader 一讀一寫留下的值（10:23 -> 623）。換算結果正確，但檔案沒人看得懂
        logger.warning(f"[config] 時間設定是數字 {value}，視為 {hhmm}；請在 config.yaml 改寫成 '{hhmm}'")
        return hhmm
    return str(value).strip()


@dataclass
class CameraParmCfg:
    """Camera parameters"""
    GainAuto: str
    Gain: float
    ExposureAuto: str
    ExposureTime: float
    BalanceWhiteAuto: str

@dataclass
class Config:
    """Main configuration object aggregating all settings"""
    mqtt: MqttCfg
    runtime: RuntimeCfg
    thresholds: ThresholdsCfg
    placement: PlacementCfg
    stability: StabilityCfg
    camera_params: CameraParmCfg
    bright_ctrl: dict
    classes: List[ClassInfo]
    menus_ticket: List[str]
    menus_sticker: List[str]
    menus_mapping: List[MappingCasesCfg]
    csv_log: CsvLogCfg = field(default_factory=CsvLogCfg)
    dla_guard: DlaGuardCfg = field(default_factory=DlaGuardCfg)
    order_guard: OrderGuardCfg = field(default_factory=OrderGuardCfg)
    cloud_upload: CloudUploadCfg = field(default_factory=CloudUploadCfg)
    # 該部署「真正會出現」的詞彙。與 menus_ticket/menus_sticker 分開，
    # 因為後者來自 menus_mapping.yaml（吉野家 1235 筆），拿來當小場景的
    # 模糊比對候選會大量錯配。留空則退回舊行為。
    vocab_ticket: List[str] = field(default_factory=list)
    vocab_sticker: List[str] = field(default_factory=list)
    # "tray": 原本綁 tray 盤的狀態機; "single": 單一訂單、不需要 tray 盤的狀態機
    mode: str = "tray"
    preset_roi: Optional[PresetRoiCfg] = None   # 新增:僅 mode == "preset_roi" 時才會有值
    ocr_routes: Dict[str, str] = field(default_factory=dict)
    ticket_parser: str = "table"
    object_class_items: Dict[str, str] = field(default_factory=dict)
    # 傳票名稱 -> 多個貼紙組合變體（同名多列的來源，見 load_config 的說明）。
    # 值一定是 List[List[str]]：即使只有一個變體也包一層，呼叫端不需要分岔。
    menu_variants: Dict[str, List[List[str]]] = field(default_factory=dict)
    # 菜名等價層。menus_ticket / menus_sticker 已經是這兩個索引產出的代表菜名，
    # 但比對時還需要索引本身（把 OCR 讀到的任意寫法折到代表菜名）。
    ticket_index: Optional["MenuIndex"] = None
    sticker_index: Optional["MenuIndex"] = None
    # menu_aliases.yaml 的 distinct 台帳：人工判定「這兩個寫法不是同一道菜」。
    # 除了阻止等價層合併，也用來否決狀態機「同名多變體 -> 自動視為異寫」的
    # 推論（見 TrayStateMachine._expand_parsed_items）。
    menu_distinct: List[List[str]] = field(default_factory=list)
    # 傳票修飾列台帳（menu_modifiers.yaml + 共用的 ja_ticket_grammar.yaml）。
    # 認出「変更しじみ汁」這種「改上一列」的列，並把被換掉的槽位從套餐的
    # 需求裡扣掉。沒有 menu_modifiers.yaml 的品牌會拿到一份空知識，
    # 狀態機所有相關流程都是 no-op。
    modifier_knowledge: Optional["ModifierKnowledge"] = None

# ---------- Configuration Loader ----------

def _resolve(base: Path, p: str | None) -> str | None:
    """Resolves a path relative to the base directory if it's not absolute."""
    return str((base / p).resolve())

def _load_vocab_file(base_dir: str, rel_path) -> List[str]:
    """讀取部署詞彙檔（一行一詞）。找不到檔案時回傳空 list 並警告。"""
    if not rel_path:
        return []
    path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        logger.warning(f"[config] 詞彙檔不存在，忽略: {path}")
        return []
    words: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            w = line.strip()
            if w and not w.startswith("#"):
                words.append(w)
    return words


class ConfigYamlLoader(yaml.SafeLoader):
    """SafeLoader，但不做 YAML 1.1 的「六十進位」數字轉換。

    PyYAML 照 YAML 1.1 把沒加引號的 10:23 讀成整數 623（10×60+23）。
    config.yaml 在執行期會被「讀進來 → 整份寫回」（hikcam 存相機參數、
    preset_roi 自動校正），一讀一寫之後 cloud_upload.at 就從 10:23 變成 623，
    行為雖然沒錯（載入時會換算回來），但檔案裡的值沒人看得懂。
    拿掉這條規則後 10:23 維持字串，寫回時 safe_dump 會自動加上引號。
    其他數字（含 0x / 0o / 小數 / inf）照舊。
    """


_INT_TAG = "tag:yaml.org,2002:int"
_FLOAT_TAG = "tag:yaml.org,2002:float"
ConfigYamlLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in (_INT_TAG, _FLOAT_TAG)]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
# 與 PyYAML 內建的 int / float 規則相同，只少了 `(?::[0-5]?[0-9])+` 那一支
ConfigYamlLoader.add_implicit_resolver(
    _INT_TAG,
    re.compile(r"""^(?:[-+]?0b[0-1_]+
                |[-+]?0[0-7_]+
                |[-+]?(?:0|[1-9][0-9_]*)
                |[-+]?0x[0-9a-fA-F_]+)$""", re.X),
    list("-+0123456789"),
)
ConfigYamlLoader.add_implicit_resolver(
    _FLOAT_TAG,
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def _load_yaml(path: str) -> dict:
    """Loads a YAML file and returns its content.（不做六十進位轉換，見 ConfigYamlLoader）"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=ConfigYamlLoader)
    return data

def load_config(config_path: str, overrides=None) -> Config:
    """
    Loads and parses all YAML configuration files into a structured Config object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dir = Path(config_path).resolve().parent
    from profile_config import merge_settings
    base = merge_settings(_load_yaml(config_path), overrides or {})
    parser = base.get("ticket_parser", "table")
    if parser not in {"table", "legacy"}:
        raise ValueError(f"Unknown ticket_parser: {parser}")

    try:
        # Load Classes
        inc = base["includes"]
        classes: List[ClassInfo] = []
        if inc["classes"]:
            classes_path = os.path.join(base_dir, inc["classes"])
            classes_yaml = _load_yaml(classes_path)
            classes_data = classes_yaml["classes"]
            for c in classes_data:
                classes.append(ClassInfo(
                    id=int(c["id"]),
                    name=str(c["name"])
                ))

        # Load Mapping Case Menus
        mapping_cases: List[MappingCasesCfg] = []
        from logic.known_class_items import CLASS_TO_ITEM_NAME
        object_class_items = dict(CLASS_TO_ITEM_NAME)
        if inc["menus_mapping"]:
            mc_path = os.path.join(base_dir, inc["menus_mapping"])
            mc_yaml = _load_yaml(mc_path)
            object_class_items = mc_yaml.get("object_class_items", object_class_items)
            for mc in mc_yaml["mapping"]:
                mapping_cases.append(MappingCasesCfg(
                    name=str(mc["name"]),
                    sub_items=[str(s) for s in mc["sub_items"]]
                ))

        # ------------------------------------------------------------------
        # 菜名等價層（同菜異寫）
        # ------------------------------------------------------------------
        # 日文同一道菜可能寫成漢字或假名（唐揚げ / から揚げ），而
        # menus_mapping.yaml 只會收錄客戶 xlsx 裡的那一種寫法。這裡把整個
        # 菜名池折疊成「代表菜名」，比對與核對都改用代表菜名，貼紙印哪一種
        # 寫法都能命中。詳見 logic/menu_alias.py 的說明。
        #
        # 兩份知識檔都是選填，路徑可用 includes 覆寫；沒有檔案時退回
        # 「NFKC + 去符號 + 片假名折疊」，仍優於改動前。
        readings_path = _resolve(base_dir, inc.get("readings") or "../ja_readings.yaml")
        aliases_path = _resolve(base_dir, inc.get("menu_aliases") or "menu_aliases.yaml")
        alias_know = load_alias_knowledge(readings_path, aliases_path)

        # 池子刻意保留重複次數：代表菜名的挑選會用到出現頻率
        ticket_pool = [mc.name for mc in mapping_cases]
        sticker_pool = [sub for mc in mapping_cases for sub in mc.sub_items]

        ticket_index = build_menu_index(alias_know, "ticket", ticket_pool)
        # 貼紙側的顯示寫法優先跟傳票一致：作業員手上拿的是傳票
        sticker_index = build_menu_index(
            alias_know, "sticker", sticker_pool, prefer=ticket_pool
        )

        # 比對候選改用代表菜名（原本是 1106 / 381 筆各種寫法混在一起）
        menus_ticket: List[str] = list(ticket_index.canonical_names)
        menus_sticker: List[str] = list(sticker_index.canonical_names)

        # ------------------------------------------------------------------
        # 傳票名稱 -> 多個「貼紙組合」變體
        # ------------------------------------------------------------------
        # menus_mapping.yaml 允許同一個 name 出現多列（同一張傳票在不同 POS
        # 代碼下對到不同的貼紙組合，例如 4切牛ﾀﾝ 有 唐辛子 / たれ / 都沒有
        # 三個版本）。傳票上讀不到 POS 代碼，所以「到底是哪一版」是不可知的，
        # 只能把每一版都當成活著的假設，任一版被滿足就算完成。
        #
        # 舊版把這裡建成 Dict[str, List[str]]，同名後者覆蓋前者，28 組變體
        # 只有最後一列活下來 —— 這就是「傳票寫 から揚げ３個、店家貼紙印
        # 假名卻被判 WRONG_ITEM」的直接原因。
        menu_variants: Dict[str, List[List[str]]] = {}
        for mc in mapping_cases:
            key = ticket_index.canonical_or_self(mc.name)
            combo = [sticker_index.canonical_or_self(sub) for sub in mc.sub_items]
            combos = menu_variants.setdefault(key, [])
            if combo and combo not in combos:   # 同內容的重複列不必留兩份
                combos.append(combo)

        multi = {k: v for k, v in menu_variants.items() if len(v) > 1}
        if multi:
            logger.info(
                f"[config] 傳票名稱有多個貼紙組合變體: {len(multi)} 筆"
                f"（任一變體被滿足即算完成）"
            )

        # ------------------------------------------------------------------
        # 傳票修飾列（換湯 / 換飯…）
        # ------------------------------------------------------------------
        # 文法跨品牌共用（跟 ja_readings.yaml 同一層），軸台帳各品牌一份。
        # 兩份都是選填，缺檔就拿到空知識、整個功能停用。
        #
        # 台帳的 slots 照 xlsx 的寫法寫，這裡過 sticker_index 折成代表菜名，
        # 才跟 menu_variants 裡的 sub_items 對得起來。
        from logic.text_utils import normalize_text as _norm_text
        grammar_path = _resolve(base_dir, inc.get("ticket_grammar") or "../ja_ticket_grammar.yaml")
        modifiers_path = _resolve(base_dir, inc.get("menu_modifiers") or "menu_modifiers.yaml")
        modifier_knowledge = load_modifier_knowledge(
            grammar_path, modifiers_path,
            normalize=lambda s: _norm_text(s, ignore_space=True),
            canonical_sticker=sticker_index.canonical_or_self,
        )
        # 再從菜單自動補上「文法命中 + 貼紙帶変更標記」的修飾列。
        # 現場會為了新的傳票寫法（しじみ汁に変更…）往 menus_mapping.yaml 加列，
        # 要求每一筆都先人工登記等於保證遲早漏掉，而漏掉是靜默的。
        # 推導條件很嚴，詳見 ModifierKnowledge.derive_from_menu。
        if modifier_knowledge.axes:
            modifier_knowledge.derive_from_menu(menu_variants)

        # Load deployment vocabularies (optional).
        # 純文字檔，一行一個詞；支援 UTF-8 / UTF-8-BOM 與 CRLF。
        vocab_ticket = _load_vocab_file(base_dir, inc.get("vocab_ticket"))
        vocab_sticker = _load_vocab_file(base_dir, inc.get("vocab_sticker"))
        if vocab_ticket:
            logger.info(f"[config] vocab_ticket 載入 {len(vocab_ticket)} 筆: {vocab_ticket}")
        if vocab_sticker:
            logger.info(f"[config] vocab_sticker 載入 {len(vocab_sticker)} 筆: {vocab_sticker}")

        # Parse MQTT settings
        m = base["mqtt"]
        mqtt = MqttCfg(
            host=m["host"],
            port=int(m["port"]),
            base_topic=m["base_topic"],
            session=m["session"],
            client_id=m.get("client_id") or None,
        )

        # Parse Runtime settings
        rt = base["runtime"]
        model = rt["model"]
        camera = rt["camera"]
        stream = rt["stream"]

        object_det_path = model["object_det"]
        ocr_det_path = model["ocr_det"]
        ocr_cls_path = model["ocr_cls"]
        ocr_rec_path = model["ocr_rec"]
        text_path = model["text"]

        # depth camera setting
        depth_cam = base["depth_camera"]

        runtime = RuntimeCfg(
            model=ModelCfg(
                object_det=_resolve(base_dir, object_det_path),
                ocr_det=_resolve(base_dir, ocr_det_path),
                ocr_cls=_resolve(base_dir, ocr_cls_path),
                ocr_rec=_resolve(base_dir, ocr_rec_path),
                text=_resolve(base_dir, text_path),
            ),
            camera=CameraCfg(
                source=camera["source"],
                width=camera["width"],
                height=camera["height"],
                device=camera["device"]
            ),
            depth_cam=DepthCamCfg(depth_cam),
            stream=StreamCfg(port=int(stream["port"]), stream_size=stream["stream_size"])
        )

        # Thresholds
        thr = base["thresholds"]
        thresholds = ThresholdsCfg(
            ai_conf=float(thr["ai_conf"]),
            ai_iou=float(thr["ai_iou"])
        )

        # Placement
        pl = base["placement"]
        placement = PlacementCfg(
            roi_strink=float(pl["roi_strink"]),
            iou_assign=float(pl["iou_assign"]),
            iou_candidate = float(pl["iou_candidate"]),
            drift_iou_thresh=float(pl["drift_iou_thresh"]),
        )

        # Stability
        st = base["stability"]
        stability = StabilityCfg(
            n_settle_frame=int(st["n_settle_frame"]),
            k_container_fail=int(st["k_container_fail"]),
            k_container_new=int(st["k_container_new"]),
            tray_missing_frame=int(st["tray_missing_frame"]),
            sticker_missing_frame=int(st["sticker_missing_frame"]),
            ticket_leave_frame=int(st["ticket_leave_frame"]),
            still_shift_ratio=float(st.get("still_shift_ratio") or 0.0),
            wrong_item_confirm=int(st.get("wrong_item_confirm") or 0),
            wrong_item_strong_score=float(st.get("wrong_item_strong_score") or 0.0),
            wrong_item_strong_conf=float(st.get("wrong_item_strong_conf", 0.9) or 0.0),
        )

        # Mode: 決定要啟用哪一種狀態機 ("tray" | "single" | "preset_roi")
        # 舊的 config.yaml 沒有這個欄位也沒關係，預設當作 "tray"，行為完全不變。
        mode = str(base.get("mode", "tray")).strip().lower()
        if mode not in ("tray", "single", "preset_roi"):
            logger.warning(f"設定檔 mode 值不合法: '{mode}'，退回預設值 'tray'")
            mode = "tray"

        # Preset ROI(僅 preset_roi 模式需要;其餘模式底下若 yaml 有寫,仍會解析但不會被使用)
        preset_roi_cfg: Optional[PresetRoiCfg] = None
        if "preset_roi" in base:
            pr = base["preset_roi"]

            class_names = {c.name for c in classes}

            def _parse_auto(roi_name: str, auto_data: dict, source: str) -> Tuple[AutoRoiCfg, str]:
                """
                解析單一 ROI 的 auto 區塊,並回傳(可能被修正過的) source。

                class_name 打錯的話,自動校正會永遠等不到那個類別、白等到逾時,
                現場只會看到 ROI 沒動卻不知道為什麼 —— 所以這裡直接拿
                classes.yaml 對一次,對不到就退回 manual 並大聲記一筆。
                """
                auto_cfg = AutoRoiCfg(
                    class_name=str(auto_data.get("class_name", AutoRoiCfg.class_name)),
                    padding_ratio=float(auto_data.get("padding_ratio", AutoRoiCfg.padding_ratio)),
                    confirm_frames=int(auto_data.get("confirm_frames", AutoRoiCfg.confirm_frames)),
                    stable_iou=float(auto_data.get("stable_iou", AutoRoiCfg.stable_iou)),
                    timeout_frames=int(auto_data.get("timeout_frames", AutoRoiCfg.timeout_frames)),
                    persist=bool(auto_data.get("persist", AutoRoiCfg.persist)),
                )
                if class_names and auto_cfg.class_name not in class_names:
                    logger.error(
                        f"preset_roi.{roi_name}.auto.class_name = '{auto_cfg.class_name}' "
                        f"不在 classes.yaml 裡 (可用的有 {sorted(class_names)})，"
                        f"{roi_name} 退回 manual"
                    )
                    source = "manual"
                return auto_cfg, source

            def _parse_roi(name: str, roi_data: dict) -> RoiCfg:
                source = str(roi_data.get("source", "manual")).strip().lower()
                if source not in ("manual", "auto"):
                    logger.warning(
                        f"preset_roi.{name}.source 值不合法: '{source}'，退回 'manual'"
                    )
                    source = "manual"

                auto_cfg, source = _parse_auto(name, roi_data.get("auto") or {}, source)

                return RoiCfg(
                    name=name,
                    points=[[float(x), float(y)] for x, y in roi_data["points"]],
                    color=roi_data.get("color"),
                    source=source,
                    auto=auto_cfg,
                )

            if "ticket_roi" not in pr:
                raise ValueError("preset_roi 設定缺少必要欄位 'ticket_roi'")

            # 舊版(2026/09)的 source 設定曾經放在 preset_roi 這一層，後來下移到
            # 每個 ROI 底下。留著提醒就好，免得舊設定檔搬過來時默默失效。
            for legacy_key in ("ticket_roi_source", "auto_roi"):
                if legacy_key in pr:
                    logger.warning(
                        f"preset_roi.{legacy_key} 已不再支援，請改寫成 "
                        f"preset_roi.<roi>.source / preset_roi.<roi>.auto"
                    )

            preset_roi_cfg = PresetRoiCfg(
                ticket_roi=_parse_roi("ticket_roi", pr["ticket_roi"]),
                object_roi=_parse_roi("object_roi", pr["object_roi"]) if "object_roi" in pr else None,
                packing_roi=_parse_roi("packing_roi", pr["packing_roi"]) if "packing_roi" in pr else None,
            )

        if mode == "preset_roi" and preset_roi_cfg is None:
            raise ValueError("mode 設為 'preset_roi' 時，設定檔必須提供 'preset_roi.ticket_roi'")

        # CSV log：舊 config 沒這一段也能載入，整段用 dataclass 的預設值
        cl = base.get("csv_log") or {}
        csv_log = CsvLogCfg(
            root_dir=str(cl.get("root_dir", CsvLogCfg.root_dir)),
            folder=str(cl.get("folder", CsvLogCfg.folder)),
            video_category=str(cl.get("video_category", CsvLogCfg.video_category)),
            max_video_size=cl.get("max_video_size", CsvLogCfg.max_video_size),
        )

        # DLA 載入守衛：舊 config 沒這一段也能載入，整段用 dataclass 的預設值
        dg = base.get("dla_guard") or {}
        dla_guard_cfg = DlaGuardCfg(
            enabled=bool(dg.get("enabled", DlaGuardCfg.enabled)),
            max_reboots=int(dg.get("max_reboots", DlaGuardCfg.max_reboots)),
            delay_sec=int(dg.get("delay_sec", DlaGuardCfg.delay_sec)),
            reset_after_sec=int(dg.get("reset_after_sec", DlaGuardCfg.reset_after_sec)),
        )

        # 訂單防護：舊 config 沒這一段也能載入，整段用 dataclass 的預設值
        og = base.get("order_guard") or {}
        order_guard_cfg = OrderGuardCfg(
            timeout_sec=float(og.get("timeout_sec", OrderGuardCfg.timeout_sec)),
            max_item_qty=int(og.get("max_item_qty", OrderGuardCfg.max_item_qty)),
        )

        # 雲端上傳：舊 config 沒這一段也能載入，不寫 = 關閉
        cu = base.get("cloud_upload") or {}
        folders = cu.get("folders", None)
        if isinstance(folders, str):
            folders = [f.strip() for f in folders.split(",") if f.strip()]
        cloud_upload_cfg = CloudUploadCfg(
            enabled=bool(cu.get("enabled", CloudUploadCfg.enabled)),
            at=_parse_hhmm(cu.get("at"), CloudUploadCfg.at),
            window_min=int(cu.get("window_min", CloudUploadCfg.window_min)),
            store_code=str(cu.get("store_code") or ""),
            folders=[str(f) for f in folders] if folders else ["event"],
            module_dir=str(cu.get("module_dir", CloudUploadCfg.module_dir)),
            retry_sec=int(cu.get("retry_sec", CloudUploadCfg.retry_sec)),
        )

        # Camera
        ca = base["camera_params"]
        bright_ctrl = base["birghtness_controller"]
        #camera_params = CameraParmCfg(
        #    GainAuto = str(ca["GainAuto"]),
        #    Gain = float(ca["Gain"]),
        #    ExposureAuto = str(ca["ExposureAuto"]),
        #    ExposureTime = float(ca["ExposureTime"]),
        #    BalanceWhiteAuto = str(ca["BalanceWhiteAuto"])
        #)

        unknown = set(object_class_items) - {c.name for c in classes}
        if base.get("ticket_parser") and unknown:
            raise ValueError(f"Unknown object_class_items: {sorted(unknown)}")
        return Config(
            ticket_parser=parser,
            object_class_items=object_class_items,
            ocr_routes=validate_routes(base.get('ocr_routes', {})),
            mqtt=mqtt,
            runtime=runtime,
            thresholds=thresholds,
            placement=placement,
            stability=stability,
            camera_params=ca,
            classes=classes,
            menus_ticket=menus_ticket,
            menus_sticker=menus_sticker,
            menus_mapping=mapping_cases,
            menu_variants=menu_variants,
            ticket_index=ticket_index,
            sticker_index=sticker_index,
            menu_distinct=alias_know.distinct,
            modifier_knowledge=modifier_knowledge,
            csv_log=csv_log,
            dla_guard=dla_guard_cfg,
            order_guard=order_guard_cfg,
            cloud_upload=cloud_upload_cfg,
            vocab_ticket=vocab_ticket,
            vocab_sticker=vocab_sticker,
            mode=mode,
            bright_ctrl=bright_ctrl,
            preset_roi=preset_roi_cfg,
        )

    except KeyError as e:
        error_msg = f"設定檔缺失必要參數: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        logger.error(f"解析設定檔時發生非預期錯誤: {e}")
        raise


# ---------- Configuration Writer ----------

def save_preset_roi(cfg: Config, roi: RoiCfg) -> None:
    """
    把單一個 preset_roi 區塊寫回這個 profile 的 config.yaml。

    App 只負責送 cmd/preset_roi,不寫檔;現值一律由 AI 這邊落地,App 下次
    啟動時再從 config.yaml 唯讀載入,兩邊才不會各有一份不同的 ROI。

    寫法跟 hikcam 存相機參數一致:先寫 tmp 再 os.replace,避免寫一半斷電
    就毀掉整個設定檔。
    """
    cfg_path = Path(cfg.source_config_path)
    data = _load_yaml(str(cfg_path))

    block = {"points": [[int(round(pt[0])), int(round(pt[1]))] for pt in roi.points]}
    if roi.color:
        block["color"] = [int(c) for c in roi.color]

    preset_block = data.setdefault("preset_roi", {})
    # 保留同一個 ROI 底下其他將來可能新增的欄位,只覆蓋 points / color
    existing = preset_block.get(roi.name)
    if isinstance(existing, dict):
        existing.update(block)
    else:
        preset_block[roi.name] = block

    # 暫存檔名刻意跟 hikcam 存相機參數用的 tmp_<stem> 不同：兩邊寫的是同一個
    # config.yaml，各自用不同的 tmp 檔，才不會在同時存檔時互相蓋掉半個檔案。
    tmp_path = cfg_path.with_name(f"tmp_preset_roi_{cfg_path.name}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, indent=4, allow_unicode=True, sort_keys=False)
    os.replace(tmp_path, cfg_path)
    logger.success(f"[FILE UPDATE] preset_roi.{roi.name} 已寫回 {cfg_path}")

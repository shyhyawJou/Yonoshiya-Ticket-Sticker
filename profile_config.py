"""品牌選擇與品牌設定載入。

設定檔責任刻意分成兩層：

* ``runtime.yaml``：版本庫管理的共用預設品牌。
* ``runtime.local.yaml``：單一機台的品牌選擇，只允許 ``profile``；此檔受
  .gitignore 排除，不保存相機、OCR、MQTT 或其他實際設定。
* ``tasks/ocr/<brand>/config.yaml``：該品牌唯一的完整設定來源。直接修改檔案
  與 UI／MQTT 寫回相機參數，都應落在同一份品牌 config。

舊版曾將每台機器的品牌 override（包含相機參數）寫入 runtime.local.yaml，
導致 local 靜默蓋過 config。新版載入前會由 migrate_legacy_runtime_local()
執行一次性遷移：備份舊 local、把相機參數搬回對應品牌 config，最後讓 local
只留下 profile。如此既保留升級前實際生效的現場相機值，也消除雙重設定來源。
"""
from copy import deepcopy
from pathlib import Path
import time

from loguru import logger
import yaml


PROFILE_NAMES = {"yoshinoya", "suzumo"}


def merge_settings(base, overrides):
    """遞迴合併字典且不修改輸入；僅供舊相機參數遷移使用。"""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict:
    """讀取 YAML 並要求根節點為 mapping，避免後續出現模糊型別錯誤。

    用 config.ConfigYamlLoader：這裡讀到的品牌 config 會被整份寫回，
    不能把 10:23 這種時間先轉成 623（見 ConfigYamlLoader 的說明）。
    """
    from config import ConfigYamlLoader   # 延遲 import：config 會在載入時 import 本模組
    data = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=ConfigYamlLoader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 必須是 YAML mapping")
    return data


def _write_yaml_atomic(path: Path, data, tmp_name: str) -> None:
    """先完整寫入暫存檔再 replace，降低斷電留下半份 YAML 的風險。"""
    temp = path.with_name(tmp_name)
    temp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=4),
        encoding="utf-8",
    )
    temp.replace(path)


def migrate_legacy_runtime_local(root) -> None:
    """把舊版 local override 一次性轉成只含 profile 的品牌選擇檔。

    相機參數會搬進各自品牌的 config，因為 local 值在舊版原本具有最高
    優先權；保留它可避免升級當下相機曝光或 Gain 突然改變。其他 override
    不再套用，因為新版要求所有實際設定都直接維護在品牌 config；它們仍會
    留在帶時間戳的備份中，方便人工比對或追查。

    已經只有 profile 的新格式檔案會直接返回，因此正常啟動不會反覆備份、
    改寫 local 或品牌 config。
    """
    root = Path(root)
    local_path = root / "runtime.local.yaml"
    if not local_path.exists():
        return

    local = _read_yaml(local_path)
    extra_top_level = set(local) - {"profile"}
    if not extra_top_level:
        # 新格式已符合規範，不產生任何啟動期寫入。
        return

    profile = local.get("profile")
    if profile is not None and profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile: {profile!r}")

    profiles = local.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ValueError("runtime.local.yaml 的 profiles 必須是 YAML mapping")

    moved = []
    ignored = []
    for brand, overrides in profiles.items():
        if brand not in PROFILE_NAMES or not isinstance(overrides, dict):
            ignored.append(f"profiles.{brand}")
            continue

        camera_params = overrides.get("camera_params")
        if camera_params is not None:
            # 每個品牌分別搬回自己的 config；即使命令列目前選了另一品牌，
            # 也一併清理 local 內已存在的歷史資料，避免日後切換品牌又復活。
            config_path = root / "tasks" / "ocr" / brand / "config.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = _read_yaml(config_path)
            config["camera_params"] = merge_settings(
                config.get("camera_params", {}), camera_params
            )
            _write_yaml_atomic(
                config_path, config, f"tmp_migrate_{config_path.name}"
            )
            moved.append(f"profiles.{brand}.camera_params")

        ignored.extend(
            # 除 camera_params 外不搬移：讓品牌 config 成為唯一來源。
            # 原值仍存在下方建立的備份，不會無法追查。
            f"profiles.{brand}.{key}"
            for key in overrides
            if key != "camera_params"
        )

    ignored.extend(sorted(extra_top_level - {"profiles"}))

    original = local_path.read_text(encoding="utf-8-sig")
    backup = local_path.with_name(
        f"{local_path.name}.bak-{time.strftime('%Y%m%d_%H%M%S')}"
    )
    backup.write_text(original, encoding="utf-8")

    # local 最終只保留原本機台選擇的品牌；若舊檔連 profile 都沒有，移除
    # 空檔，後續自然退回 runtime.yaml 的版本庫預設品牌。
    normalized = {"profile": profile} if profile is not None else {}
    if normalized:
        _write_yaml_atomic(local_path, normalized, "runtime.local.yaml.tmp")
    else:
        local_path.unlink()

    logger.warning(
        "[profile] 舊版 runtime.local.yaml 已正規化為僅選擇品牌；"
        f"備份於 {backup.name}"
        + (f"；已搬移 {sorted(moved)}" if moved else "")
        + (f"；已停用 {sorted(ignored)}" if ignored else "")
    )


def select_profile(root, explicit=None):
    """只決定品牌與對應 config 路徑，不讀取或合併任何 local 設定值。

    品牌優先順序為：命令列 explicit > runtime.local.yaml > runtime.yaml >
    yoshinoya。local 若出現 profile 以外欄位會明確失敗，避免新的隱藏
    override 再次進入載入流程。
    """
    root = Path(root)
    common = _read_yaml(root / "runtime.yaml")
    local_path = root / "runtime.local.yaml"
    local = _read_yaml(local_path) if local_path.exists() else {}

    unexpected = set(local) - {"profile"}
    if unexpected:
        raise ValueError(
            "runtime.local.yaml 只能包含 profile；"
            f"不支援欄位: {sorted(unexpected)}"
        )

    selected = explicit or local.get("profile") or common.get("profile", "yoshinoya")
    if selected not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile: {selected!r}")

    path = root / "tasks" / "ocr" / selected / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return selected, path


def load_profile_config(root, explicit=None):
    """遷移舊 local、選擇品牌，再從該品牌 config 載入完整設定。

    這裡刻意不把 runtime.local.yaml 傳入 load_config 做 merge；載入完成後
    附加的 profile_name/profile_root/source_config_path 只提供執行期識別與
    相機寫回正確品牌 config，不是第二份設定來源。
    """
    from config import load_config

    root = Path(root).resolve()
    migrate_legacy_runtime_local(root)
    selected, path = select_profile(root, explicit)
    cfg = load_config(str(path))
    cfg.profile_name = selected
    cfg.profile_root = str(root)
    cfg.source_config_path = str(path)
    return cfg

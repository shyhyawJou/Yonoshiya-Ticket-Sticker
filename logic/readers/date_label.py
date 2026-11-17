"""
readers/date_label.py
=====================
吉野家的沙拉日期標籤：點サラダ時，除了一般的品項貼紙，旁邊常會多貼一張小標籤 ——
上面是小字「サラダ」、「保存方法 10℃以下 / 消費期限」，底下是大字的日期（26. 8. 8）
與時間（16:27）。設定在 tasks/ocr/<品牌>/reference_img/salad_date_label/label.yaml。

這張標籤偵測器一樣框成 order_sticker，照一般貼紙比對會出事：只讀到日期時間時
連續兩次就對前端發 Unknown；讀到「サラダ」時又變成多一份サラダ（錯誤品項、擋完成）。

判定與讀取（det 優先讀字，模板確認身份）：
  det  rec 讀到合法的日期（YY.M.D）或時間（HH:MM），而且其他框沒有對到サラダ以外的菜名
       （一般貼紙剛好印了時間也不會被誤認）
  CV   label.yaml 的 cv 打開時，用同一個資料夾的參考圖找模板區（readers/label_cv.py）：
       - det 已判定：模板對到就確認身份；det 缺的欄位用 CV 補讀（[CV補救]），
         兩邊都有但不同時用 det、log 記 [CV不一致]
       - det 沒判定、而且 det 的字也不是任何菜名：模板對到、CV 又讀出合法的日期或時間，
         才判定成日期標籤（CV 單獨不能判定，一定要有讀出來的字）
  require_template（預設開，2026-11-16）：模板沒對到就**不是**日期標籤，只靠 det 讀到日期時間不算。
       現場店內備料罐的保存期限標籤（72時間 / 26.11.16 / 15:58）被 det 讀出 26.11.16，
       就被當成 サラダ、報錯誤品項；日期時間是所有保存期限標籤都有的，認不出是哪一種標籤。
       模板沒對到的照一般貼紙比對（讀不出菜名就是 Unknown，不擋完成）。
       關掉 = 舊行為：模板分數不到門檻時 CV 不參與，結果跟只用 det 一樣。

判定之後怎麼核對在 readers/sticker.py（StickerReader.apply_date_label）。沙拉品項分兩種
（2026-11-16 chris.fan 說明現場）：
  原味（item，サラダ）  日期標籤本身就是它的貼紙，可能只有日期標籤、也可能另有一般貼紙：
                        任一張都能代表那份サラダ，另一張進來就跟它成組、不重複計數（v3.4.1 起）
  口味（salad_items）   ポテトサラダ 等一定有完整名稱的一般貼紙，日期標籤是附屬的、不一定有：
                        日期標籤不能代替它，只能跟它的一般貼紙成組；一般貼紙還沒到就先收著（不計數、不報錯）
口味的清單由菜單推導：貼紙名稱含 salad_items.contain、排除 exclude 與 item。
日期時間只印 log（[沙拉日期]），CSV 不動。

品名小字（サラダ）比日期時間的字小，常被 ocr_engine 的 _filter_sticker_boxes 濾掉，
所以品項不靠讀它：目前只有サラダ會印日期標籤，品項直接取設定的 item。
log 標出品名的根據：「サラダ(模板0.95)」= 參考圖的模板區（含「サラダ」字樣）對到了；
「サラダ(設定)」= 只有 det 讀到日期時間，品名是照設定填的（只在 require_template 關掉時出現）。
"""

from __future__ import annotations

import datetime
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Tuple

import yaml
from loguru import logger

from logic.readers.label_cv import LabelCvConfig, LabelTemplate

_DATE_RE = re.compile(r"^(\d{2})\.(\d{1,2})\.(\d{1,2})$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_MIN_CONF = 0.5          # 跟 StickerMatcher 一樣，信心不到 0.5 的框不採用
_OTHER_MENU_SCORE = 60   # 跟 StickerMatcher 採用比對結果的門檻一致


@dataclass
class DateLabelConfig:
    enabled: bool = False
    item: str = "サラダ"     # 這張標籤代表的品項（貼紙上的名稱）
    cv: LabelCvConfig = field(default_factory=LabelCvConfig)
    folder: Optional[str] = None   # label.yaml 所在資料夾（參考圖也在這裡）
    require_template: bool = True  # 模板沒對到就不是日期標籤（見檔頭）
    # 口味沙拉（見檔頭）：貼紙名稱含 salad_contain、不在 salad_exclude 的（item 本身除外）。空字串 = 沒有口味沙拉
    salad_contain: str = ""
    salad_exclude: Tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "DateLabelConfig":
        """檔案不存在 = 關閉（其他品牌沒有這個資料夾）"""
        if not path or not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        salad = raw.get("salad_items") or {}
        return cls(enabled=bool(raw.get("enabled", False)), item=str(raw.get("item") or cls.item),
                   cv=LabelCvConfig.from_dict(raw.get("cv")), folder=os.path.dirname(os.path.abspath(path)),
                   require_template=bool(raw.get("require_template", cls.require_template)),
                   salad_contain=str(salad.get("contain") or ""),
                   salad_exclude=tuple(str(x) for x in (salad.get("exclude") or ())))


@dataclass
class DateLabelRead:
    date: Optional[str]   # 「26.8.8」，沒讀到為 None
    time: Optional[str]   # 「16:27」
    template: Optional[float] = None   # 參考圖模板分數；None = CV 沒對到（或沒開）

    def describe(self, item: str) -> str:
        src = f"模板{self.template:.2f}" if self.template is not None else "設定"
        return f"{item}({src}):{self.date or '?'}-{self.time or '?'}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def _parse_date(text: str) -> Optional[str]:
    m = _DATE_RE.match(text)
    if not m:
        return None
    yy, mo, dd = (int(g) for g in m.groups())
    try:
        datetime.date(2000 + yy, mo, dd)
    except ValueError:
        return None
    return f"{yy}.{mo}.{dd}"


def _parse_time(text: str) -> Optional[str]:
    m = _TIME_RE.match(text)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return f"{hh}:{mm:02d}"


_PARSERS = {"date": _parse_date, "time": _parse_time}


class DateLabelReader:
    def __init__(self, cfg: DateLabelConfig, sticker_matcher, rec_pred=None):
        self.cfg = cfg
        self.item = cfg.item
        self.sticker_matcher = sticker_matcher
        known = sticker_matcher.vocab_sticker or sticker_matcher.menus_sticker or []
        if self.item not in known:
            logger.warning(f"[沙拉日期] label.yaml 的 item '{self.item}' 不在貼紙菜單裡，比對時會一律判成錯誤品項")
        # 口味沙拉：由貼紙菜單推導（見檔頭）。比對用 NFKC，半形 ｻﾗﾀﾞ 也算
        exclude = {_clean(x) for x in cfg.salad_exclude}
        contain = _clean(cfg.salad_contain)
        self.flavored = frozenset(
            name for name in known
            if contain and contain in _clean(name) and _clean(name) not in exclude and name != self.item
        )
        # 所有沙拉品項（日期標籤可以跟哪些一般貼紙成組）
        self.items = frozenset({self.item}) | self.flavored
        if contain:
            logger.info(f"[沙拉日期] 口味沙拉（日期標籤只能跟它的一般貼紙成組，不能代替）{len(self.flavored)} 種："
                        f"{sorted(self.flavored)}（貼紙名稱含 '{cfg.salad_contain}'，排除 {list(cfg.salad_exclude)}）")
        self.cv: Optional[LabelTemplate] = None
        self.require_template = cfg.require_template
        if cfg.cv.enabled and cfg.folder:
            self.cv = LabelTemplate(cfg.cv, cfg.folder, rec_pred, tag="沙拉日期")
            if not self.cv.ready:
                logger.warning("[沙拉日期] 參考圖輔助沒有可用的參考圖 / rec / 欄位設定"
                               + ("，require_template 開著，日期標籤一律判定不了" if self.require_template else "，只用 det"))
                self.cv = None
        elif self.require_template:
            logger.warning("[沙拉日期] require_template 開著但 cv 關閉，日期標籤一律判定不了（照一般貼紙比對）")
        logger.info(f"[沙拉日期] 日期標籤已啟用：代表品項 '{self.item}'，參考圖輔助 "
                    f"{'開' if self.cv else '關'}，"
                    f"{'一定要模板對到' if self.require_template else '模板沒對到時只靠 det 也算'}（日期時間只印 log）")

    def read(self, rec_res: list) -> Optional[DateLabelRead]:
        """只看 det 的結果：這次 OCR 是不是日期標籤；不是回 None。"""
        date = time_ = None
        others = []
        for text, conf in rec_res or []:
            if float(conf) < _MIN_CONF:
                continue
            t = _clean(text)
            d, tm = _parse_date(t), _parse_time(t)
            if d and not date:
                date = d
            elif tm and not time_:
                time_ = tm
            elif not (d or tm):
                others.append(t)
        if date is None and time_ is None:
            return None
        name = self._other_menu(others)
        if name is not None:
            logger.info(f"[沙拉日期] 讀到日期/時間 {date}/{time_}，但其他框對到 '{name}'，當一般貼紙處理")
            return None
        return DateLabelRead(date=date, time=time_)

    def _menu_of(self, text: str) -> Optional[str]:
        name, score = self.sticker_matcher.resolve(text)
        return name if name is not None and (score or 0) > _OTHER_MENU_SCORE else None

    def _other_menu(self, texts) -> Optional[str]:
        # 口味沙拉的名字也算「別的菜名」：讀得到 ポテトサラダ 的就是它的一般貼紙（或兩張重疊被框在一起），
        # 照一般貼紙核對 —— 口味沙拉一定要有一般貼紙，不能被當成日期標籤收著不計數
        for t in texts:
            name = self._menu_of(t)
            if name is not None and name != self.item:
                return name
        return None

    def _no_template(self, det: Optional[DateLabelRead], why: str) -> Optional[DateLabelRead]:
        """模板沒對到（或沒辦法比）時：require_template 開著就不是日期標籤，否則照 det"""
        if det is None or not self.require_template:
            return det
        logger.info(f"[沙拉日期] 讀到日期/時間 {det.date}/{det.time}，但{why}，不當日期標籤（照一般貼紙比對）")
        return None

    def read_with_cv(self, rec_res: list, frame_crop) -> Optional[DateLabelRead]:
        """det 判定 + 參考圖確認（見檔頭）。require_template 關著時，CV 沒開或模板沒對到等於 read(rec_res)。"""
        det = self.read(rec_res) if rec_res else None
        if self.cv is None or frame_crop is None:
            return self._no_template(det, "參考圖輔助沒開" if self.cv is None else "沒有貼紙影像可比對模板")
        if det is None:
            # det 的字是サラダ以外的菜名就是一般貼紙，不必跑 CV。
            # 只讀到「サラダ」的要跑：可能是一般サラダ貼紙，也可能是 det 只讀到小字品名的日期標籤
            if self._other_menu([_clean(t) for t, c in rec_res or [] if float(c) >= _MIN_CONF]):
                return None
        try:
            res = self.cv.read(frame_crop)
        except Exception as e:  # 輔助功能出錯不能讓貼紙讀不到
            logger.exception(f"[沙拉日期] 參考圖輔助發生例外: {e}")
            return self._no_template(det, "參考圖輔助發生例外")
        if res is None:
            return self._no_template(det, f"模板沒對到（門檻 {self.cv.cfg.min_score:g}）")
        got = {k: _PARSERS[k](_clean(t)) if k in _PARSERS else None for k, (t, _) in res.texts.items()}
        raw = "、".join(f"{k}「{t}」{c:.2f}" for k, (t, c) in res.texts.items())
        note = f"模板 {res.score:.2f}（{res.ref}，轉{res.rotate}）{res.ms:.0f}ms"

        if det is None:
            if not (got.get("date") or got.get("time")):
                logger.info(f"[沙拉日期] det 沒讀到日期時間，模板對到但 CV 也讀不出合法值（{note}：{raw}），當一般貼紙")
                return None
            logger.info(f"[CV補救] [沙拉日期] det 沒讀到日期時間，CV 判定為日期標籤：{note}：{raw}")
            return DateLabelRead(date=got.get("date"), time=got.get("time"), template=res.score)

        out = DateLabelRead(date=det.date, time=det.time, template=res.score)
        for k in ("date", "time"):
            d, c = getattr(det, k), got.get(k)
            if d is None and c is not None:
                setattr(out, k, c)
                logger.info(f"[CV補救] [沙拉日期] det 沒讀到{('日期' if k == 'date' else '時間')}，CV 讀到 {c}（{note}）")
            elif d is not None and c is not None and d != c:
                logger.warning(f"[CV不一致] [沙拉日期] {('日期' if k == 'date' else '時間')} det={d} CV={c}，用 det（{note}：{raw}）")
        return out

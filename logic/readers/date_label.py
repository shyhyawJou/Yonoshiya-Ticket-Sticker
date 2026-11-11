"""
readers/date_label.py
=====================
吉野家的沙拉日期標籤：點サラダ時，除了一般的品項貼紙，旁邊常會多貼一張小標籤 ——
上面是小字「サラダ」、「保存方法 10℃以下 / 消費期限」，底下是大字的日期（26. 8. 8）
與時間（16:27）。設定在 tasks/ocr/<品牌>/reference_img/salad_date_label/label.yaml。

這張標籤偵測器一樣框成 order_sticker，照一般貼紙比對會出事：只讀到日期時間時
連續兩次就對前端發 Unknown；讀到「サラダ」時又變成多一份サラダ（錯誤品項、擋完成）。

判定與讀取（det 優先，CV 輔助，跟大傳票的 CV 輔助同一個分工）：
  det  rec 讀到合法的日期（YY.M.D）或時間（HH:MM），而且其他框沒有對到サラダ以外的菜名
       （一般貼紙剛好印了時間也不會被誤認）
  CV   label.yaml 的 cv 打開時，用同一個資料夾的參考圖找模板區（readers/label_cv.py）：
       - det 已判定：模板對到就確認身份；det 缺的欄位用 CV 補讀（[CV補救]），
         兩邊都有但不同時用 det、log 記 [CV不一致]
       - det 沒判定、而且 det 的字也不是任何菜名：模板對到、CV 又讀出合法的日期或時間，
         才判定成日期標籤（CV 單獨不能判定，一定要有讀出來的字）
       模板分數不到門檻 = CV 不參與，結果跟只用 det 一樣。

判定之後怎麼核對在 readers/sticker.py：一般貼紙與日期標籤任一張都能代表那份サラダ，
另一張進來就跟它成組、不重複計數（StickerReader._join_pair）。日期時間只印 log
（[沙拉日期]），CSV 不動。

品名小字（サラダ）比日期時間的字小，常被 ocr_engine 的 _filter_sticker_boxes 濾掉，
所以品項不靠讀它：目前只有サラダ會印日期標籤，品項直接取設定的 item。
log 標出品名的根據：「サラダ(模板0.95)」= 參考圖的模板區（含「サラダ」字樣）對到了；
「サラダ(設定)」= 只有 det 讀到日期時間，品名是照設定填的。
"""

from __future__ import annotations

import datetime
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

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

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "DateLabelConfig":
        """檔案不存在 = 關閉（其他品牌沒有這個資料夾）"""
        if not path or not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(enabled=bool(raw.get("enabled", False)), item=str(raw.get("item") or cls.item),
                   cv=LabelCvConfig.from_dict(raw.get("cv")), folder=os.path.dirname(os.path.abspath(path)))


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
        self.cv: Optional[LabelTemplate] = None
        if cfg.cv.enabled and cfg.folder:
            self.cv = LabelTemplate(cfg.cv, cfg.folder, rec_pred, tag="沙拉日期")
            if not self.cv.ready:
                logger.warning("[沙拉日期] 參考圖輔助沒有可用的參考圖 / rec / 欄位設定，只用 det")
                self.cv = None
        logger.info(f"[沙拉日期] 日期標籤已啟用：代表品項 '{self.item}'，參考圖輔助 "
                    f"{'開' if self.cv else '關'}（日期時間只印 log）")

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
        for t in texts:
            name = self._menu_of(t)
            if name is not None and name != self.item:
                return name
        return None

    def read_with_cv(self, rec_res: list, frame_crop) -> Optional[DateLabelRead]:
        """det 判定 + 參考圖輔助（見檔頭）。CV 沒開或模板沒對到時，等於 read(rec_res)。"""
        det = self.read(rec_res) if rec_res else None
        if self.cv is None or frame_crop is None:
            return det
        if det is None:
            # det 的字是サラダ以外的菜名就是一般貼紙，不必跑 CV。
            # 只讀到「サラダ」的要跑：可能是一般サラダ貼紙，也可能是 det 只讀到小字品名的日期標籤
            if self._other_menu([_clean(t) for t, c in rec_res or [] if float(c) >= _MIN_CONF]):
                return None
        try:
            res = self.cv.read(frame_crop)
        except Exception as e:  # 輔助功能出錯不能讓貼紙讀不到
            logger.exception(f"[沙拉日期] 參考圖輔助發生例外，只用 det: {e}")
            return det
        if res is None:
            return det
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

"""
readers/sticker.py
==================
餐點貼紙的 OCR 結果怎麼套用到訂單：比對成功 -> 確認進度 / ITEM_CHECKED / TRAY_COMPLETED；
比對失敗 -> 錯誤品項確認次數、強證據、Unknown 通知（WRONG_ITEM_DETECTED）。

貼紙文字怎麼對到菜名是 StickerMatcher 的事（readers/sticker_matcher.py）；完成判定與
確認數交給 OrderRequirements。醬料包（偵測類別即身份）不經 OCR，但比對完同樣走
settle，由 TrayStateMachine.resolve_known_class_items 呼叫。
內容是從 TrayStateMachine 原封不動搬出來的。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from loguru import logger

from logic.models import Tray, TrayState
from logic.order_requirements import OrderRequirements
from logic.readers.base import find_task_item
from logic.readers.debug_hooks import NO_DEBUG


def _is_subsequence(short: str, long: str) -> bool:
    """short 的每個字依序出現在 long 裡（可以不連續）。'黒カレー' 之於 '牛黒カレー'、'豚丼特盛' 之於 '豚丼超特盛'"""
    it = iter(long)
    return len(short) < len(long) and all(ch in it for ch in short)


def blocks_completion(ts) -> bool:
    """這張貼紙會不會擋住訂單完成。

    已成立的錯誤品項會擋（舊行為）。還在確認中、而且至少一次讀到「不在訂單內的
    品名」的也要擋：不然其他品項剛好在這時湊齊，訂單進 COMPLETED 後就不再送
    貼紙 OCR，這張真的放錯的貼紙永遠到不了確認次數，整筆被記成 OK。
    只讀不出菜名（Unknown）的不擋：那多半是字印得差、或根本不是貼紙。
    """
    return ts.is_wrong_item or any(kind == "WRONG" for kind, _ in ts.wrong_reads)


class StickerReader:
    def __init__(self, bus, sticker_matcher, requirements: OrderRequirements,
                 wrong_item_confirm: int = 0, wrong_item_strong_score: float = 0.0,
                 wrong_item_strong_conf: float = 0.9, date_label=None):
        self.bus = bus
        self.sticker_matcher = sticker_matcher
        self.requirements = requirements
        # 沙拉日期標籤（readers/date_label.py::DateLabelReader）。None = 關閉，貼紙全部照一般比對
        self.date_label = date_label

        # 錯誤品項確認次數（見 settle）。0 = 舊行為：第一次判錯就成立
        self.wrong_item_confirm = int(wrong_item_confirm or 0)
        if self.wrong_item_confirm > 0:
            logger.info(
                f"[wrong_confirm] 錯誤品項確認啟用：同一張貼紙連續 {self.wrong_item_confirm} 次"
                f"沒比對成功才通知前端；讀不出菜名的以 Unknown 通知，不擋完成、不寫 CSV"
            )
        # 強證據一次成立：比對分數 >= strong_score 且 rec 信心 >= strong_conf 的錯誤品名不必等滿次數。
        # 歷史 log 裡看錯字（模糊、遮擋）都落在模糊比對（分數 < 100），一字不差讀成別的菜名幾乎都是真的放錯。
        # 0 = 關閉（每次都要等滿 wrong_item_confirm 次）
        self.wrong_item_strong_score = float(wrong_item_strong_score or 0)
        self.wrong_item_strong_conf = float(wrong_item_strong_conf or 0)
        if self.wrong_item_confirm > 0 and self.wrong_item_strong_score > 0:
            logger.info(
                f"[wrong_confirm] 強證據一次成立：比對分數 >= {self.wrong_item_strong_score:g}"
                f" 且 rec 信心 >= {self.wrong_item_strong_conf:g}"
            )
        # 除錯總覽圖的收集 hooks，由 TrayStateMachine 注入；沒注入時是 no-op（見 readers/debug_hooks.py）
        self.debug = NO_DEBUG

    def apply_sticker_result(self, tray: Tray, tray_id: str, dt_boxes, rec_res, is_flip, task_bbox,
                             target=None, frame_crop=None):
        target_ts = find_task_item(tray.stickers, task_bbox, target)
        if target_ts is not None:
            target_ts.is_ocr_busy = False
            target_ts.last_result_mono = time.monotonic()

        # det 優先、參考圖輔助（readers/date_label.py）；det 一個字都沒讀到時也讓 CV 看一次
        label = self.date_label.read_with_cv(rec_res, frame_crop) if self.date_label is not None else None
        if label is not None:
            self.apply_date_label(tray, tray_id, target_ts, label, dt_boxes, rec_res, frame_crop)
            return

        matched_item = None
        match_status = "UNRECOGNIZED"

        if len(rec_res) > 0:
            matched_item, match_status = self.sticker_matcher.match(
                rec_res, dt_boxes, is_flip, tray.expected_items, tray.checked_items
            )

        # 一般サラダ貼紙：先看有沒有缺一般貼紙的日期標籤可以成組，有就併進去、不再計數。
        # 要在判 MATCHED / WRONG_ITEM 之前：日期標籤先到時サラダ已經滿額，這張會被判成多的
        if (self.date_label is not None and target_ts is not None
                and match_status in ("MATCHED", "WRONG_ITEM")
                and self._report_name(tray, matched_item) == self._report_name(tray, self.date_label.item)):
            primary = self._open_pair(tray, {self._report_name(tray, matched_item)}, want_label=True, exclude=target_ts)
            if primary is not None:
                self._join_pair(tray, primary, target_ts)
                logger.info(f"[沙拉日期] 一般貼紙 {matched_item} 跟已核對的日期標籤 {primary.date_label} "
                            f"成組（不重複計數）tray={tray.id} 單號={tray.order_number}")
                self.debug.sticker(tray, tray_id, target_ts, frame_crop, dt_boxes, rec_res, matched_item, "MATCHED")
                return

        if match_status in ("MATCHED", "WRONG_ITEM") and self.moved_here(tray, target_ts, matched_item):
            return

        raw_text = getattr(self.sticker_matcher, "last_raw_text", None) if len(rec_res) > 0 else None
        strong = match_status == "WRONG_ITEM" and self.is_strong_wrong(tray, matched_item)
        self.settle(tray, tray_id, target_ts, matched_item, match_status,
                    confirm=self.wrong_item_confirm, raw_text=raw_text, strong=strong)
        if self.date_label is not None and target_ts is not None and match_status == "MATCHED":
            # 口味沙拉的一般貼紙照常核對（上面的 settle），有先到、收著等它的日期標籤就配成一組
            self._attach_held_label(tray, target_ts)
        self.debug.sticker(tray, tray_id, target_ts, frame_crop, dt_boxes, rec_res, matched_item, match_status)

    def moved_here(self, tray: Tray, ts, name, label: bool = False) -> bool:
        """ts 讀到訂單裡的 name：是不是那張已核對的同名貼紙被移到這裡來了。是就把位置接回
        那張（不計數、不判錯），移除 ts，回 True。醬料包（TrayStateMachine.resolve_known_class_items）也走這裡。

        貼紙被快速移動時，tracker 對不上舊框，新位置會建一個新的追蹤物件；舊的那張要
        sticker_missing_frame 幀（15 幀 ≈ 1.3 s）沒看到才算離開，這段期間它還算「在畫面上」：
          - 滿額時：新框在舊的離開前就讀完 wrong_item_confirm 次，被判成多一份
            （2026-11-06 實驗機單號 494，チーズ 1 份：移動後 0.9 s 內錯誤品項成立，舊框 1.3 s 後才離開）
          - 沒滿時：新框比對成功，checked_items 同時有新舊兩張，confirmed_counts 就被算成 2
        sticker_settle_frame 調小後新框讀得快，這兩種都更容易發生。
        計數規則是「同時在畫面上的份數」（confirmed_counts 取 high-water mark，兩份要同時出現
        才算 2），所以同名貼紙從沒同時出現過，不管是移動還是「包走一份、再放下一份」，都只算一份。
        判斷跟傳票一樣（readers/ticket.py::apply_extra_ticket_result）：一張紙同時只有一個框，
        兩張同名貼紙從沒在同一幀一起被偵測到（seen_with，由 tracker 記錄），就是同一張。
        """
        if ts is None or not name or ts.is_wrong_item:
            return False
        rname = self._report_name(tray, name)
        if name not in tray.expected_items and rname not in tray.expected_items:
            return False
        moved_from = [
            t for t in tray.stickers
            if t is not ts and t.is_checked and t.item_name == rname
            and (t.date_label is not None) == label
            and t.missing_count > 0 and not t.has_notified_missing   # 這幀沒看到、還沒判定離開
            and id(t) not in ts.seen_with
        ]
        if not moved_from:
            return False
        self._reattach(tray, ts, min(moved_from, key=lambda t: t.missing_count),   # 剛被拿起來的那張消失最短
                       f"'{name}'，跟已核對的同名貼紙")
        return True

    def _reattach(self, tray: Tray, ts, anchor, what: str) -> None:
        """ts 是 anchor 被移動到新位置：位置接回 anchor，移除 ts（不計數、不判錯）"""
        logger.info(
            f"[貼紙移動] tray={tray.id} 新位置讀到 {what}從沒同時出現"
            f"（舊框 {anchor.missing_count} 幀沒看到），判定為同一張被移動："
            f"位置接回已核對的那張，不加計數、不判錯誤品項"
        )
        anchor.bbox, anchor.xywhr = ts.bbox, ts.xywhr
        anchor.missing_count = 0
        anchor.is_missing = False
        anchor.seen_with |= ts.seen_with
        tray.stickers.remove(ts)
        # set_ocr_busy 送出時改成 CHECKING_STICKERS，settle 沒跑到，要自己改回來
        if tray.state != TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS

    # ------------------------------------------------------------------
    # 沙拉日期標籤（readers/date_label.py）
    #
    # 沙拉品項分兩種（2026-11-16 現場說明）：
    #   原味 サラダ（date_label.item）：日期標籤本身就是它的貼紙。一般貼紙與日期標籤任一張都能
    #     代表那份サラダ（現場不保證兩張都會貼）：先到的照一般品項核對（ITEM_CHECKED），後到的
    #     另一種跟它成組，不再計數、不再通知前端（v3.4.1 起）。
    #   口味沙拉（date_label.flavored，ポテトサラダ 等）：一定有完整名稱的一般貼紙，日期標籤是附屬的。
    #     日期標籤不能代替它：先到的話「收著」（已處理、不計數、不報錯），一般貼紙照常核對後跟它成組。
    # 日期標籤進來時依序：
    #   1. 有已核對、還沒成組的沙拉一般貼紙（原味或口味）-> 成組
    #   2. 原味 サラダ 還沒到齊 -> 以日期標籤核對 サラダ
    #   3. 口味沙拉還有沒配到日期標籤的份數 -> 收著
    #   4. 都不是 -> 多的，照錯誤品項（サラダ）
    # 1 排在 2 前面是刻意的：同一筆同時點原味與口味沙拉時，日期標籤分不出是誰的。先成組的錯法是
    # 「原味只貼了日期標籤、卻被配給口味的一般貼紙」-> サラダ 顯示沒到（店員看得到）；反過來的錯法是
    # 「原味根本沒放、口味附的日期標籤被當成 サラダ」-> 訂單完成，缺餐被靜默放過。
    # 成組只看「同一筆訂單裡缺另一半」，不看兩張貼紙的位置（份數對就好）。
    # 成組 / 收著的那張是 pair_partner：is_checked（不再送 OCR）但不進 checked_items；
    # 離開畫面時的處理見 TrayStateMachine._leave_pair。
    # ------------------------------------------------------------------
    @staticmethod
    def _report_name(tray: Tray, item):
        """settle 會把純異寫改成主變體寫法再記到貼紙上，比對成組時兩邊都用對外的寫法"""
        return getattr(tray, "report_alias", {}).get(item, item)

    def _salad_names(self, tray: Tray, flavored_only: bool = False) -> set:
        names = self.date_label.flavored if flavored_only else self.date_label.items
        return {self._report_name(tray, n) for n in names}

    def _open_pair(self, tray: Tray, names, want_label: bool, exclude=None):
        """找一張已核對、還沒成組、還在畫面上、名稱在 names 裡的貼紙；want_label=True 找日期標籤，False 找一般貼紙"""
        for t in tray.stickers:
            if (t is exclude or not t.is_checked or t.is_missing or t.pair_with is not None
                    or t.pair_partner or t.item_name not in names):
                continue
            if (t.date_label is not None) == want_label:
                return t
        return None

    @staticmethod
    def _join_pair(tray: Tray, primary, ts, item_name=None) -> None:
        """ts 併進 primary 那一組（primary=None = 收著等一般貼紙）：標成已核對（不再送 OCR），
        但不進 checked_items、不發 ITEM_CHECKED"""
        ts.is_checked = True
        ts.item_name = primary.item_name if primary is not None else item_name
        ts.pair_partner = True
        ts.pair_with = primary
        if primary is not None:
            primary.pair_with = ts
        ts.wrong_reads.clear()
        ts.is_unknown_item = False
        ts.is_wrong_item = False
        # set_ocr_busy 送出時改成 CHECKING_STICKERS，settle 沒跑到，要自己改回來
        if tray.state != TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS

    @staticmethod
    def _held_labels(tray: Tray, exclude=None):
        """收著、還沒配到一般貼紙的日期標籤（還在畫面上）"""
        return [t for t in tray.stickers
                if t is not exclude and t.date_label is not None and t.pair_partner
                and t.pair_with is None and not t.is_missing]

    def _flavored_waiting(self, tray: Tray, exclude=None) -> Optional[str]:
        """口味沙拉還有沒配到日期標籤的份數時，回傳其中一個品名（收著的日期標籤記在它名下）；沒有回 None"""
        names = self._salad_names(tray, flavored_only=True)
        need = Counter({n: c for n, c in Counter(tray.expected_items).items() if n in names})
        if not need:
            return None
        used = 0
        for t in tray.stickers:
            if t is exclude or t.date_label is None or not t.pair_partner or t.is_missing:
                continue
            owner = t.pair_with.item_name if t.pair_with is not None else t.item_name
            if t.pair_with is None or owner in names:
                used += 1
        if sum(need.values()) - used <= 0:
            return None
        return sorted(need)[0]

    def _attach_held_label(self, tray: Tray, ts) -> None:
        """剛核對成功的沙拉一般貼紙：有收著的日期標籤就配成一組（一般貼紙計數，日期標籤不計數）"""
        if (ts.date_label is not None or ts.pair_with is not None or ts.pair_partner
                or not ts.is_checked or ts.item_name not in self._salad_names(tray)):
            return
        held = self._held_labels(tray, exclude=ts)
        if not held:
            return
        label = held[0]
        label.pair_with, ts.pair_with = ts, label
        label.item_name = ts.item_name
        logger.info(f"[沙拉日期] 一般貼紙 {ts.item_name} 跟先前收著的日期標籤 {label.date_label} 成組"
                    f"（日期標籤不計數）tray={tray.id} 單號={tray.order_number}")

    def _label_moved_here(self, tray: Tray, ts) -> bool:
        """不計數的日期標籤（成組的另一半 / 收著的）被移到這裡：位置接回，回 True。判斷同 moved_here"""
        moved_from = [
            t for t in tray.stickers
            if t is not ts and t.date_label is not None and t.is_checked and t.pair_partner
            and t.missing_count > 0 and not t.has_notified_missing and id(t) not in ts.seen_with
        ]
        if not moved_from:
            return False
        self._reattach(tray, ts, min(moved_from, key=lambda t: t.missing_count), "日期標籤，跟已處理的日期標籤")
        return True

    def apply_date_label(self, tray: Tray, tray_id: str, ts, label, dt_boxes, rec_res, frame_crop=None):
        item = self.date_label.item
        text = label.describe(item)
        if ts is not None:
            ts.date_label = text
            if self._label_moved_here(tray, ts):
                return

        # 1. 已核對、還沒成組的沙拉一般貼紙
        primary = self._open_pair(tray, self._salad_names(tray), want_label=False, exclude=ts) if ts is not None else None
        if primary is not None:
            self._join_pair(tray, primary, ts)
            logger.info(f"[沙拉日期] {text}：跟已核對的一般貼紙 {primary.item_name} 成組（不重複計數）"
                        f"tray={tray.id} 單號={tray.order_number}")
            self.debug.sticker(tray, tray_id, ts, frame_crop, dt_boxes, rec_res, primary.item_name, "MATCHED")
            return

        # 2. 原味 サラダ 還沒到齊：日期標籤代表它
        matched_item, match_status = self.sticker_matcher.match_known_item(
            item, tray.expected_items, tray.checked_items)
        if match_status in ("MATCHED", "WRONG_ITEM") and self.moved_here(tray, ts, item, label=True):
            return
        if match_status != "MATCHED" and ts is not None:
            # 3. 口味沙拉還有沒配到日期標籤的份數：收著，等它的一般貼紙
            flavor = self._flavored_waiting(tray, exclude=ts)
            if flavor is not None:
                self._join_pair(tray, None, ts, item_name=flavor)
                logger.info(f"[沙拉日期] {text}：口味沙拉 {flavor} 附的日期標籤，一般貼紙還沒到，先收著"
                            f"（不計數、不報錯）tray={tray.id} 單號={tray.order_number}")
                self.debug.sticker(tray, tray_id, ts, frame_crop, dt_boxes, rec_res, flavor, "MATCHED")
                return

        # 4. 多的
        logger.info(
            f"[沙拉日期] {text}：" + ("還沒有一般貼紙，先以日期標籤核對" if match_status == "MATCHED"
                                     else f"這筆訂單沒有待核對的沙拉，照錯誤品項（{item}）處理")
            + f" tray={tray.id} 單號={tray.order_number}")
        # 上一張貼紙比對留下的原始文字，不能被當成這張的 matched_text 送給前端
        self.sticker_matcher.last_matched_text = None
        self.settle(tray, tray_id, ts, matched_item, match_status,
                    confirm=self.wrong_item_confirm, raw_text=text)
        self.debug.sticker(tray, tray_id, ts, frame_crop, dt_boxes, rec_res, matched_item, match_status)

    def is_strong_wrong(self, tray: Tray, matched_item) -> bool:
        """這次讀到的錯誤品名是不是強證據，可以不等滿 wrong_item_confirm 次就成立。

        分數、信心夠高之外，還要排除三種「讀得很清楚但其實不是放錯」的情況
        （2026-10-20 重放 S12~S14，都是 K 次確認本來擋得住、強證據會直接誤報成 NGtoOK）：
          1. 不是真的一字不差：ねぎ 前綴邏輯會把漏字的 'ネギ塩丼小' 以 100 分對到 'ねぎラー油'
          2. 品名本來就在訂單裡（只是數量已滿）：已核對的餐點被推開、tracker 換了新框，
             舊框還沒判定消失前新框就讀到同名 —— 真的多放一份同名則多等一次
          3. 品名跟還沒核對的品項互為子序列：'牛黒カレー' 左邊被遮住讀成 '黒カレー'、
             '豚丼超特盛' 漏一字成 '豚丼特盛'，遮擋造成的剛好是另一個真菜名
        """
        sm = self.sticker_matcher
        if self.wrong_item_strong_score <= 0 or not matched_item:
            return False
        if getattr(sm, "last_match_score", 0) < self.wrong_item_strong_score:
            return False
        if getattr(sm, "last_match_conf", 0) < self.wrong_item_strong_conf:
            return False
        if not getattr(sm, "last_match_exact", False):
            return False
        if matched_item in tray.expected_items:
            return False
        fold = getattr(sm, "_fold", None) or (lambda s: s)
        got = fold(matched_item)
        remaining = Counter(tray.expected_items) - Counter(tray.checked_items)
        for item in remaining:
            want = fold(item)
            if _is_subsequence(got, want) or _is_subsequence(want, got):
                return False
        return True

    def confirm_wrong_read(self, ts, kind: str, text: str, confirm: int, strong: bool = False):
        """累積一次沒比對成功的讀取，回傳 (成立種類, 名稱)；還沒到確認次數回 (None, None)。

        品名不要求每次相同：rec 信心在 0.5 上下時，同一張貼紙每次對到的品名不一定一樣。
        確認次數內只要有一次讀到真的品名就算錯誤品項，名稱取出現最多的（同票取最後一次）；
        全部都讀不出菜名才是 Unknown，名稱帶最後一次讀到的字。
        strong = 這次是強證據（一字不差讀成別的菜名），直接以這次的品名成立。
        """
        ts.wrong_reads.append((kind, text))
        del ts.wrong_reads[:-confirm]
        if strong and kind == "WRONG":
            return "WRONG", text
        if len(ts.wrong_reads) < confirm:
            return None, None
        names = [t for k, t in ts.wrong_reads if k == "WRONG"]
        if names:
            counts = Counter(names)
            best = max(counts.values())
            return "WRONG", [n for n in names if counts[n] == best][-1]
        return "UNKNOWN", f"Unknown:{ts.wrong_reads[-1][1]}"

    def settle(self, tray: Tray, tray_id: str, ts, matched_item, match_status,
               confirm: int = 0, raw_text: Optional[str] = None, strong: bool = False):
        """
        比對結果的共用收尾:不管身份是 OCR 得來或分類器直讀得來,
        「核對成功/失敗後該做什麼」都是同一套業務邏輯。

        confirm = 錯誤品項要連續幾次才成立（0 = 第一次就成立，舊行為；醬料包固定走 0，
        它看的是偵測類別，已經有 wrong_item_margin 多等幾幀）。成立前不通知前端、
        不記 CSV。raw_text 是這次讀到的原始文字，讀不出菜名時拿來組 Unknown 名稱。
        strong = 這次讀到的是強證據（見 wrong_item_strong_score），不必等滿次數。
        """
        wrong_alert = None  # 這次要對前端發 WRONG_ITEM_DETECTED 的名稱（None = 不發）
        # 純異寫群組（牛丼小 / 牛丼小盛）統一用主變體的寫法對外，否則前端會
        # 收到一個它沒在 TICKET_READY 列出來的品名，進度對不起來。
        # 比對本身已經完成（用的是含所有寫法的聯集），這裡只改對外的名字。
        if matched_item:
            renamed = getattr(tray, "report_alias", {}).get(matched_item)
            if renamed and renamed != matched_item:
                logger.info(
                    f"[tray_state_machine] 貼紙比到 '{matched_item}'，"
                    f"對外統一回報為主變體寫法 '{renamed}'"
                )
                matched_item = renamed
        if ts is not None:
            if match_status == "MATCHED":
                ts.is_checked = True
                ts.item_name = matched_item
                if (ts.wrong_reads or ts.is_unknown_item) and not ts.is_wrong_item:
                    # 還沒成立就比對成功 = 先前是一時看錯字（模糊、遮擋），什麼都不記
                    logger.info(
                        f"[wrong_confirm] tray={tray.id} 貼紙先前 {len(ts.wrong_reads)} 次沒比對成功"
                        f" {ts.wrong_reads}，這次比對成功 '{matched_item}'，視為看錯字、不記錯誤"
                    )
                ts.wrong_reads.clear()
                ts.is_unknown_item = False
                if ts.is_wrong_item:
                    # 這個實體之前被判為多餘/錯誤，現在重新比對成功了 ——
                    # 一定要解除警示，否則訂單永遠無法完成。
                    #
                    # 2026-09-22 實測的卡單（傳票 519，現場放了兩份牛丼並）：
                    #   1. 兩份牛丼並 -> A 判 MATCHED、B 判 WRONG_ITEM
                    #   2. 拿走多餘的一份，tracker 把「消失」歸給 A（兩張貼紙
                    #      幾乎同位置，IoU 配對無法保證配到哪一個），
                    #      handle_missing_items() 把牛丼並從 checked_items 扣掉
                    #   3. B 還在畫面上會被重新送 OCR，此時 remaining 又空出
                    #      牛丼並 -> B 這次判 MATCHED
                    #   4. 舊版只設 is_checked，is_wrong_item 留著沒清
                    #   5. B 實體還在，tracker 每幀把 is_missing 歸零（見
                    #      single_order_tracker 的貼紙更新），所以
                    #      resolve_wrong_item_removal() 那條「實體離開才解除」
                    #      的路徑永遠碰不到它 -> has_unresolved_wrong_item
                    #      永遠是 True -> 永久卡在 WAITING_STICKERS，
                    #      而且 heartbeat_wrong_items() 會一直對前端重發
                    #      WRONG_ITEM_DETECTED
                    #
                    # 只有「這個品項確實屬於本訂單、而且還有沒填滿的需求」時
                    # StickerMatcher 才會回 MATCHED（它看的是
                    # expected_items - checked_items），所以這裡解除警示不會
                    # 放過真正的錯餐點：不屬於訂單的貼紙下一次仍然判 WRONG_ITEM。
                    #
                    # tray.ever_had_wrong_item / wrong_item_history 刻意不清 ——
                    # 這筆訂單確實發生過錯誤，那是要寫進 CSV 的稽核紀錄。
                    logger.info(
                        f"[tray_state_machine] 原本被判定為錯誤/多餘的 "
                        f"'{matched_item}' 重新比對成功，解除警示 tray={tray.id}"
                    )
                    ts.is_wrong_item = False
            else:
                ts.stable_frames = 0
                if confirm <= 0:
                    # 舊行為：第一次判錯就成立
                    if match_status == "WRONG_ITEM":
                        if not ts.is_wrong_item:
                            # 只在「第一次」被判定為 wrong item 時記一筆，
                            # 避免同一個物件之後重複送 OCR 或 heartbeat 通知造成重複計數
                            tray.ever_had_wrong_item = True
                            tray.wrong_item_history.append(matched_item)
                        ts.is_wrong_item = True
                        ts.item_name = matched_item
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = matched_item
                elif ts.is_wrong_item:
                    # 已成立：照舊重發警示，名稱固定用成立時那個（之後每次讀到的品名可能不同）
                    if match_status == "WRONG_ITEM":
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = ts.item_name
                else:
                    # 確認中：讀到不在訂單內的品名，或讀出了字但比對不到菜名，都算一次。
                    # 一個字都沒讀到不算（沒有證據），也不歸零。
                    kind, text = None, None
                    if match_status == "WRONG_ITEM":
                        kind, text = "WRONG", matched_item
                    elif raw_text:
                        kind, text = "UNKNOWN", raw_text
                    verdict, name = (self.confirm_wrong_read(ts, kind, text, confirm, strong=strong)
                                     if kind else (None, None))
                    if verdict == "WRONG":
                        # 只在成立的那一次記一筆（一張貼紙只記一次，跟舊行為一樣）
                        tray.ever_had_wrong_item = True
                        tray.wrong_item_history.append(name)
                        ts.is_wrong_item = True
                        ts.is_unknown_item = False
                        ts.item_name = name
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = name
                        reason = ("強證據（一字不差、信心夠高）" if strong
                                  else f"連續 {confirm} 次沒比對成功")
                        logger.warning(
                            f"[wrong_confirm] tray={tray.id} {reason} {ts.wrong_reads}"
                            f" -> 錯誤品項成立 '{name}'"
                        )
                    elif verdict == "UNKNOWN":
                        if not ts.is_unknown_item:
                            logger.warning(
                                f"[wrong_confirm] tray={tray.id} 連續 {confirm} 次讀不出菜名 {ts.wrong_reads}"
                                f" -> 通知 '{name}'（不擋完成、不寫 CSV）"
                            )
                        ts.is_unknown_item = True
                        ts.item_name = name
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = name
                    elif kind:
                        logger.info(
                            f"[wrong_confirm] tray={tray.id} 疑似錯誤 {len(ts.wrong_reads)}/{confirm}"
                            f" {ts.wrong_reads}，先不通知"
                        )
        elif match_status == "WRONG_ITEM":
            if confirm <= 0:
                wrong_alert = matched_item  # 舊行為：對不到追蹤中的貼紙也照發
            else:
                logger.warning(f"[wrong_confirm] tray={tray.id} OCR 結果對不到追蹤中的貼紙，無法累積確認次數: '{matched_item}'")

        if match_status == "MATCHED":
            # --------------------------------------------------
            # 1. checked_items = 現在畫面上存在的已確認餐點
            # --------------------------------------------------
            tray.checked_items.append(matched_item)

            # --------------------------------------------------
            # 2. confirmed_counts = 歷史最高確認進度
            #    只有真的突破歷史最高值，才通知 frontend
            # --------------------------------------------------
            before = self.requirements.report_checked_items(tray)
            confirmed_increased = self.requirements.update_confirmed_count(
                tray,
                matched_item
            )
            items_list = self.requirements.report_checked_items(tray)
            if confirmed_increased and items_list == before:
                # 換湯 show_removed: optional 多貼的 みそ汁：貼了也收，但不在前端清單上，不通知
                logger.info(f"[tray_state_machine] '{matched_item}' 是換湯多貼的那張（不在前端清單），"
                            f"不送 ITEM_CHECKED tray={tray.id}")
                confirmed_increased = False

            if confirmed_increased:

                payload = {
                    "tray_id": tray_id,
                    "status": "ITEM_CHECKED",
                    "items": items_list
                }
                # 新增欄位：這次實際從貼紙上讀到的原始文字。顯示名是折疊後
                # 選出的代表菜名（跟傳票同一種寫法），但店家貼紙可能印的是
                # 另一種寫法 —— 兩個都給，前端才能顯示得跟現場一致。
                raw_text = getattr(self.sticker_matcher, "last_matched_text", None)
                if raw_text and raw_text != matched_item:
                    payload["matched_text"] = raw_text
                self.bus.publish_det_status(payload)

            # --------------------------------------------------
            # 3. 判斷整份訂單是否完成
            #    多重假設：任一個貼紙組合變體被湊齊就算完成，不需要把
            #    「所有變體的聯集」都湊滿（那會等一份永遠不會來的品項）。
            # --------------------------------------------------
            all_completed = self.requirements.is_order_satisfied(tray, tray.confirmed_counts)

            has_unresolved_wrong_item = any(
                blocks_completion(t)
                for t in tray.stickers
            )

            # 已經完成就不再送：換湯 show_removed: optional 時，完成後才貼上的那張 みそ汁 也會走到這裡，
            # 不擋的話前端會收到第二次 TRAY_COMPLETED（完成後不會退回，只有合併第二張傳票會，那時 state 已改掉）
            if all_completed and not has_unresolved_wrong_item and tray.state != TrayState.COMPLETED:
                tray.state = TrayState.COMPLETED

                items_list = self.requirements.report_checked_items(tray)

                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "TRAY_COMPLETED",
                    "items": items_list
                })

        elif wrong_alert is not None:
            self.bus.publish_det_status({
                "tray_id": tray_id,
                "status": "WRONG_ITEM_DETECTED",
                "items": [{wrong_alert: 1}]
            })

        if tray.state != TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS

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
                 wrong_item_strong_conf: float = 0.9):
        self.bus = bus
        self.sticker_matcher = sticker_matcher
        self.requirements = requirements

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
        matched_item = None
        match_status = "UNRECOGNIZED"

        if len(rec_res) > 0:
            matched_item, match_status = self.sticker_matcher.match(
                rec_res, dt_boxes, is_flip, tray.expected_items, tray.checked_items
            )

        target_ts = find_task_item(tray.stickers, task_bbox, target)
        if target_ts is not None:
            target_ts.is_ocr_busy = False
            target_ts.last_result_mono = time.monotonic()

        raw_text = getattr(self.sticker_matcher, "last_raw_text", None) if len(rec_res) > 0 else None
        strong = match_status == "WRONG_ITEM" and self.is_strong_wrong(tray, matched_item)
        self.settle(tray, tray_id, target_ts, matched_item, match_status,
                    confirm=self.wrong_item_confirm, raw_text=raw_text, strong=strong)
        self.debug.sticker(tray, tray_id, target_ts, frame_crop, dt_boxes, rec_res, matched_item, match_status)

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
            confirmed_increased = self.requirements.update_confirmed_count(
                tray,
                matched_item
            )

            if confirmed_increased:
                items_list = [
                    {item: count}
                    for item, count in tray.confirmed_counts.items()
                ]

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

            if all_completed and not has_unresolved_wrong_item:
                tray.state = TrayState.COMPLETED

                items_list = [
                    {item: count}
                    for item, count in tray.confirmed_counts.items()
                ]

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

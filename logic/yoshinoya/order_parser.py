from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from loguru import logger
from rapidfuzz import fuzz, process

from logic.menu_alias import FoldedMatcher
from logic.text_utils import FuzzyMatcher
from utils.skew_corrector import SkewCorrector
from utils.quad_validator import validate_and_fix_boxes, find_angle_outliers

ParsedItem = Tuple[str, int]  # (menu_name, quantity)

# 幽靈品項防護的分數界線（詳細理由見 parse() 內 [幽靈列防護] 那一段）。
# 名稱比對只是「勉強像」某道菜（分數低於這個值），而且整列完全找不到數量
# 證據時，判定為版面文字而不是品項。等價層/完全命中會得 100 分，一定安全。
WEAK_MATCH_SCORE = 70.0

# 訂單編號的形狀：一段數字，後面可以用連字號接更多段數字（4、480、234-01、1234-5…）。
# 刻意不限位數與段數 —— 現場會出現哪些組合不知道，真正分辨「是不是編號」的是
# 位置（右上角、「数」標題上方，見 parse()），格式只負責排除時間 18:14、金額、品名。
# OCR 常把「-」讀成長音「ー」或其他橫線，全部視為同一個連字號。
_ORDER_DASHES = "-ー―‐‑−–—~〜"
ORDER_NO_RE = re.compile(r"\d+(?:[" + re.escape(_ORDER_DASHES) + r"]\d+)*")


def normalize_order_number(text: str) -> Optional[str]:
    """文字框看起來是訂單編號就回傳正規化後的字串（連字號一律半形 '-'），否則 None。

    NFKC 先把全形數字/符號轉半形；空白一律去掉（OCR 常在 '-' 兩側多切空白）。
    """
    s = unicodedata.normalize("NFKC", text).replace(" ", "")
    if not ORDER_NO_RE.fullmatch(s):
        return None
    return re.sub("[" + re.escape(_ORDER_DASHES) + "]", "-", s)


# ----------------------------------------------------------------------------
# 菜單區域下界（表格底線）
# ----------------------------------------------------------------------------
# 傳票的品項一律印在「数 / メニュー名 / 金額」那張表格裡，表格以下是小計、合計、
# テーブルNO 與店家自訂的說明文字。原本品項區只有上界（單號往下推），沒有下界；
# 2026-10 現場傳票多了「牛丼フェス・21日15時まで…」一段說明，det 框得到，
# 於是被模糊比對成 牛丼・アタマ ×7、焼肉さら ×8 這種幽靈品項
# （幽靈列防護擋不到：方法三對亂碼列裁數量欄一樣讀得出 7、8）。
#
# 下界靠表格框線找，不靠文字：det 模型刻意 overfit 傳票版面，小計/合計/日期這些
# 字常常根本沒被框出來，拿來當錨點不可靠。框線在 crop 上又淡又細、常斷，
# 所以用兩個彼此獨立的訊號互相印證：
#   1. 長橫線：灰階 blackhat（比局部背景暗、法線方向很細）再做長度 25% 寬的開運算
#   2. 長直線的下端：欄線（数｜メニュー名｜金額）從表格頂端一路畫到底線，
#      下端正好落在底線上；條碼的直條太短（< 3.5 字高）會被濾掉
# 候選 = 「有直線下端落在上面的橫線」或「兩條以上左右分開、下端同高的直線」。
# 只接受在最上面那個「数」下方至少 2 個字高的候選 —— 這排除表頭分隔線
# （在「数」下方不到 1 個字高）與背景裡別張傳票的線（在本張單號上方）。
# 都找不到就回 None，呼叫端退回文字錨點、再不行就維持沒有下界的舊行為。
MENU_BOTTOM_MIN_GAP = 2.0      # 底線至少要在「数」下方幾個字高
_RULE_H_FRAC = 0.25            # 橫線最短長度（crop 寬的比例）
_RULE_V_MULT = 3.5             # 直線最短長度（字高的倍數）
_RULE_BH_THR = 8               # blackhat 開運算後視為線條的最低灰階差
# 表格外的版面文字。CV 找不到底線時，最上面一列命中的就當下界（備援，常常沒被框到）。
_BELOW_TABLE_RE = re.compile(r"小\s*計|合\s*計|テーブル|\d{4}\s*年|\d{1,2}\s*[:：]\s*\d{2}")


def _deskew_matrix(skew: SkewCorrector) -> np.ndarray:
    """skew.correct_point 的影像版：把 crop 轉到 corrected_dt_boxes 的座標系。"""
    if skew.theta == 0.0:
        return np.float32([[1, 0, 0], [0, 1, 0]])
    c, s = np.cos(-skew.theta), np.sin(-skew.theta)
    px, py = skew.pivot
    return np.float32([[c, -s, px - c * px + s * py], [s, c, py - s * px - c * py]])


def _rule_stats(gray: np.ndarray, horizontal: bool, length: int, thick: int):
    normal = (1, thick) if horizontal else (thick, 1)
    along = (length, 1) if horizontal else (1, length)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, normal))
    op = cv2.morphologyEx(bh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, along))
    mask = np.where(op >= _RULE_BH_THR, 255, 0).astype(np.uint8)
    # 補回印刷斷點
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1) if horizontal else (1, 7)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    return [(stats[i], cents[i]) for i in range(1, n)]


def find_menu_table_bottom(frame_crop: np.ndarray, skew: SkewCorrector, text_h: float,
                           anchor_y: float) -> Tuple[Optional[float], str]:
    """回傳 (表格底線的 y（擺正座標）, 判斷依據)；找不到為 (None, 原因)。frame_crop 必須是正向。"""
    H, W = frame_crop.shape[:2]
    gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY) if frame_crop.ndim == 3 else frame_crop
    gray = cv2.warpAffine(gray, _deskew_matrix(skew), (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    thick = max(5, int(text_h * 0.35)) | 1

    h_len = max(10, int(W * _RULE_H_FRAC))
    h_rules = [(float(c[1]), float(s[0]), float(s[0] + s[2]))
               for s, c in _rule_stats(gray, True, h_len, thick) if s[2] >= h_len]
    v_len = max(10, int(text_h * _RULE_V_MULT))
    edge = W * 0.05  # 紙張邊緣、背景物件
    v_rules = [(float(c[0]), float(s[1] + s[3]))
               for s, c in _rule_stats(gray, False, v_len, thick)
               if s[3] >= v_len and edge < c[0] < W - edge]

    tol = max(4.0, text_h * 0.6)
    min_y = anchor_y + MENU_BOTTOM_MIN_GAP * text_h
    cands = []  # (支持數, 有橫線, y)
    for y, x0, x1 in h_rules:
        n = sum(1 for vx, vy1 in v_rules if abs(vy1 - y) <= tol and x0 - tol <= vx <= x1 + tol)
        if n and y > min_y:
            cands.append((n + 1, True, y))
    for _, vy1 in v_rules:
        grp = [(vx, y1) for vx, y1 in v_rules if abs(y1 - vy1) <= tol]
        xs = [vx for vx, _ in grp]
        y = float(np.median([y1 for _, y1 in grp]))
        if len(grp) >= 2 and max(xs) - min(xs) >= 0.2 * W and y > min_y:
            cands.append((len(grp), False, y))
    if not cands:
        return None, f"找不到框線 (橫線 {len(h_rules)} 條、直線 {len(v_rules)} 條)"
    # 同分取較下面那條：選低了只是少濾幾列（等於舊行為），選高了會砍掉真品項
    support, has_h, y = max(cands)
    return y, f"框線 (支持 {support}{'，橫線+直線下端' if has_h else '，直線下端'})"


class OrderParser:
    def __init__(
        self,
        menus_ticket: List[str],
        menus_with_digits: Set[str],
        fuzzy_matcher: FuzzyMatcher,
        rec_pred,  # StandaloneRecDLA
        menu_index=None,
        cv_assist: bool = False,
    ):
        self.menus_ticket = menus_ticket
        self.menus_with_digits = menus_with_digits
        self.fuzzy_matcher = fuzzy_matcher
        self.rec_pred = rec_pred
        self.abc = 0
        # 傳票側的菜名等價層。傳票是 POS 印的，寫法比貼紙穩定，但同一份
        # menus_mapping.yaml 裡本來就有 牛丼･並 / 牛丼並 這類差異（實測 37 組），
        # 用同一套折疊比對可以一併吸收掉。None 時退回舊行為。
        self.folded = FoldedMatcher(menu_index)
        # 最近一次 parse() 找到的菜單區域下界，crop 座標的兩個端點；給畫面 overlay 用
        self._menu_bottom_line = None
        # 最近一次 parse() 每一列的比對細節（原始文字 / 比對到的菜名 / 分數 / 數量與來源 / 被丟棄的原因 /
        # 該列用到的 det 框 index），給 utils/debug_sheet 的除錯總覽圖用。只讀，不影響解析結果。
        self.last_debug_rows: List[dict] = []
        # CV 輔助（config 的 ticket_cv_assist，實驗中）：版面交給 CV、字交給 det，det 漏框才用 CV 框補讀。
        # 找不到表格時 parse() 照舊走下面的 det 流程。見 logic/yoshinoya/ticket_cv_assist.py
        self.cv_assist = None
        if cv_assist:
            from logic.yoshinoya.ticket_cv_assist import CvAssist
            self.cv_assist = CvAssist(self.folded, self.menus_ticket, self.fuzzy_matcher, self.rec_pred)
        self.last_cv_result = None   # 最近一次 CV 輔助的結果（含版面），沒走 CV 時為 None

    def pop_menu_bottom_line(self):
        """取出最近一次 parse() 的菜單區域下界 ((x0, y0), (x1, y1))，沒找到為 None。

        取出後清空，所以這次 OCR 結果沒有經過 parse() 時，不會拿到上一張票的線。
        """
        line, self._menu_bottom_line = self._menu_bottom_line, None
        return line

    def parse(
        self,
        frame_crop: np.ndarray,
        dt_boxes: list,
        rec_res: list,
        is_flip: bool,
    ) -> Tuple[List[ParsedItem], Optional[str]]:
        """
        解析一張訂單影像的 OCR 結果 (已整合傾斜校正與菜名數字防禦機制)。
        """
        parsed_items: List[ParsedItem] = []
        order_number: Optional[str] = None
        self._menu_bottom_line = None
        self.last_debug_rows = []
        self.last_cv_result = None

        # is_flip 時不走：AsyncOCR 交給 parser 前已經把 crop 轉正，is_flip 實際上恆為 False
        if self.cv_assist is not None and not is_flip:
            try:
                res = self.cv_assist.parse(frame_crop, dt_boxes, rec_res)
            except Exception as e:  # 實驗功能出錯不能讓整張傳票讀不到
                logger.exception(f"[CV輔助] 發生例外，改用原本的解析: {e}")
                res = None
            if res is not None:
                self.last_cv_result = res
                self.last_debug_rows = res.debug_rows
                self._menu_bottom_line = res.menu_bottom_line
                return res.items, res.order_number
        nn_x = None  # 擺正後的數量欄中心 X 座標
        nn_w = None  # 擺正後的數量欄寬度

        flip_res = list(reversed(rec_res)) if is_flip else rec_res
        flip_dt = list(reversed(dt_boxes)) if is_flip else dt_boxes

        # 傾斜校正 1：初始化校正器
        skew = SkewCorrector.from_dt_boxes(flip_dt)
        logger.info(f"[傾斜校正] 估計角度: {skew.theta_deg:.2f} 度")

        # 傾斜校正 2：一次性批次將所有 Bbox 擺正，避免在後續迴圈重複運算消耗 CPU
        corrected_dt_boxes = skew.correct_boxes(flip_dt)

        # 檢察 bbox 有沒有變成菱形或斗笠形狀
        corrected_dt_boxes, bad_box_indices = validate_and_fix_boxes(
            corrected_dt_boxes, log_fn=logger.warning
        )
        bad_box_indices = set(bad_box_indices)
        # 角度離群值檢測（順序正確但形狀是菱形/怪異四邊形）
        angle_outlier_indices = find_angle_outliers(corrected_dt_boxes, max_dev_deg=20.0)
        if angle_outlier_indices:
            logger.warning(f"[角度異常框] 偵測到 {len(angle_outlier_indices)} 個角度離群框，index={angle_outlier_indices}")
        bad_box_indices |= set(angle_outlier_indices)

        # 1. 找出標題、外帶框、數量欄，同時收集純數字框
        takeout_y = None
        takeout_x = None
        digit_boxes = []  # 儲存結構: (corrected_cy, digit_value, text)
        order_candidates = []  # 訂單編號候選框: (cx, cy, text, x_coords)，格式見 ORDER_NO_RE
        header_cys = []        # 「数」欄標題的 cy；編號一定在標題列的上方
        header_xs = []         # 與 header_cys 對應的 (數量欄中心 x, 寬)
        layout_marker_indices = set()  # 版面標記框（「数」欄標題、「ティクアウト」標籤）的 index

        zipped_data = list(zip(flip_dt, corrected_dt_boxes, flip_res))

        for idx, (orig_box, corr_box, (text, score)) in enumerate(zipped_data):
            if score < 0.5:
                continue
            if idx in bad_box_indices:
                continue

            cy = np.mean([pt[1] for pt in corr_box])
            x_coords = [pt[0] for pt in corr_box]
            cx = np.mean(x_coords)
            cleaned_text = text.strip()

            if re.fullmatch(r'[1-9]\d*', cleaned_text):
                digit_boxes.append((cy, int(cleaned_text), cleaned_text))

            # 訂單編號候選：形狀像編號（數字，可帶連字號）的框都先收集，不限位數。
            # 稍後用「在数標題上方 + 離傳票角落最近」挑出真正的編號，而不是靠掃描順序。
            order_no = normalize_order_number(cleaned_text)
            if order_no is not None:
                order_candidates.append((cx, cy, order_no, x_coords))

            if cleaned_text == "枚":
                cleaned_text = "数"
            nn_match = process.extractOne(cleaned_text, ["数"], scorer=fuzz.token_set_ratio)
            if nn_match and nn_match[1] > 60:
                nn_x = np.mean(x_coords)
                nn_w = max(x_coords) - min(x_coords)
                header_cys.append(cy)
                header_xs.append((nn_x, nn_w))
                # 版面標記，不是品項：留著給方法三定位數量欄，但不可進入菜名比對
                layout_marker_indices.add(idx)

            take_out_match = process.extractOne(cleaned_text, ["ティクアウト"], scorer=fuzz.token_set_ratio)
            if take_out_match and take_out_match[1] > 60:
                takeout_y = cy
                takeout_x = (min(x_coords), max(x_coords))
                # 同上。標籤自己的 cy 就是 takeout_y，而下面的過濾條件是嚴格
                # 不等式（cy < takeout_y），等於的那一列濾不掉 —— 所以標籤框
                # 一定要靠 index 排除，否則「外帶」字樣本身就有機會模糊比對到
                # 菜名，變成一筆預設數量 1 的幽靈品項。
                layout_marker_indices.add(idx)

        # 1.5 菜單區域下界（表格底線，見 find_menu_table_bottom）。
        # 錨點用最上面那個「数」：真正的標題，或偶爾被讀成「数」的單號（在更上面，
        # 一樣在表格上方）。表格下方的「テーブルNO」常被讀成「テーブ数」，
        # 所以不能像下面挑編號界線那樣取最下面的。
        # is_flip 時不做：AsyncOCR 交給 parser 前已經把 crop 轉正，is_flip 實際上恆為 False。
        menu_bottom = None
        if not is_flip and header_cys:
            text_h = float(np.median([abs(b[3][1] - b[0][1]) for b in corrected_dt_boxes]))
            anchor_y = min(header_cys)
            menu_bottom, how = find_menu_table_bottom(frame_crop, skew, text_h, anchor_y)
            if menu_bottom is None:
                text_ys = [
                    float(np.mean([pt[1] for pt in corr_box]))
                    for idx, (_, corr_box, (text, score)) in enumerate(zipped_data)
                    if score >= 0.5 and idx not in bad_box_indices and _BELOW_TABLE_RE.search(text)
                ]
                text_ys = [y for y in text_ys if y > anchor_y + MENU_BOTTOM_MIN_GAP * text_h]
                if text_ys:
                    # 錨點文字本身的上緣才是界線
                    menu_bottom = min(text_ys) - 0.5 * text_h
                    how = f"{how}；改用表格外文字錨點"
            if menu_bottom is None:
                logger.warning(f"[菜單區域] 找不到表格底線，不設下界: {how}")
            else:
                logger.info(f"[菜單區域] 表格底線 y={menu_bottom:.0f}（「数」y={anchor_y:.0f}，字高 {text_h:.0f}）依據: {how}")
                # menu_bottom 在擺正座標系，轉回 crop 座標，overlay 才能跟 dt_boxes 用同一個矩陣投到全畫面
                crop_w = float(frame_crop.shape[1] - 1)
                self._menu_bottom_line = (
                    skew.inverse_correct_point(0.0, menu_bottom),
                    skew.inverse_correct_point(crop_w, menu_bottom),
                )
                # 表格下方被讀成「数」的框（テーブ数…）不可當數量欄標題，否則
                # 編號界線與方法三的數量欄 x 都會被拉到表格外
                kept = [i for i, cy in enumerate(header_cys) if cy < menu_bottom]
                if len(kept) < len(header_cys):
                    logger.info(f"[菜單區域] 忽略表格下方的「数」框 {len(header_cys) - len(kept)} 個")
                    header_cys = [header_cys[i] for i in kept]
                    nn_x, nn_w = header_xs[kept[-1]]

        # 訂單編號改用座標挑選：找離「傳票角落」最近的候選框。
        # is_flip=False → 右上角；is_flip=True（整張顛倒）→ 左下角。
        #
        # 先排除「数」標題列以下的候選：那裡是品項區，右側一整排是數量欄的 1、2…，
        # 離右上角也很近。實機 log 有 44 筆「編號: 1」—— 真正的編號 OCR 信心太低
        # 被濾掉（例 '4' 0.44），於是數量欄第一列的「1」被挑成了編號。
        # 寧可回報未確認（None → 狀態機記 "000"），也不要拿數量冒充單號。
        if order_candidates and header_cys:
            # 有多個「数」時取最靠近品項區的那一個（正向取最下面）：編號框偶爾會被
            # 誤讀成「数」，它在上方，拿它當界線會把真正的編號也排除掉。
            header_cy = min(header_cys) if is_flip else max(header_cys)
            above = [c for c in order_candidates if (c[1] > header_cy if is_flip else c[1] < header_cy)]
            if len(above) < len(order_candidates):
                logger.info(
                    f"[訂單編號] 排除「数」標題列以下的數字框（品項區/數量欄）: "
                    f"{[c[2] for c in order_candidates if c not in above]}"
                )
            order_candidates = above
            if not order_candidates:
                logger.warning("[訂單編號] 「数」標題上方沒有可用的編號（可能 OCR 信心太低或沒框到），編號未確認")

        if order_candidates:
            frame_h, frame_w = frame_crop.shape[0], frame_crop.shape[1]
            corner = (0.0, float(frame_h)) if is_flip else (float(frame_w), 0.0)

            # 可選：先過濾掉明顯不在「上方/下方區域」的候選，避免品項列裡剛好也是
            # 1~3 碼數字的文字框（例如份量、桌號）干擾角落距離的判斷。
            # 這個 0.35 是抓「傳票上方/下方 35%」，實際比例建議用真實影像微調。
            TOP_REGION_RATIO = 0.35
            if is_flip:
                region_filtered = [
                    c for c in order_candidates
                    if c[1] > frame_h * (1 - TOP_REGION_RATIO)
                ]
            else:
                region_filtered = [
                    c for c in order_candidates
                    if c[1] < frame_h * TOP_REGION_RATIO
                ]
            candidates_for_pick = region_filtered or order_candidates

            best_cx, best_cy, best_text, best_x_coords = min(
                candidates_for_pick,
                key=lambda c: (c[0] - corner[0]) ** 2 + (c[1] - corner[1]) ** 2,
            )
            order_number = best_text

            # 品項區域的邊界：以「ティクアウト」標籤為優先，找不到標籤時才用
            # 訂單編號往下推。
            #
            # det 模型改版後已經很少框出傳票上方的「外帶」字樣，所以這裡幾乎
            # 都會走編號 fallback；但標籤那條路留著當保險（未來要靠它對位、
            # 或偶發誤偵測時），所以不能像先前那樣無條件覆蓋掉它 ——
            # 編號挑選搬到迴圈外之後，它的寫入時機從「掃到編號的那一刻」變成
            # 「整個迴圈跑完」，會蓋掉標籤那支設好的較精準窗口（標籤版的 x 範圍
            # 只有欄寬，編號版是 (80, max_x-60) 這種寬鬆值）。
            if takeout_y is None:
                if not is_flip:
                    takeout_x = (80, max(best_x_coords) - 60)
                    takeout_y = best_cy + 100
                else:
                    takeout_x = (min(best_x_coords) + 60, frame_w - 80)
                    takeout_y = best_cy - 100

        # 2. 過濾有效文字區塊 (排除外帶框外的雜訊)
        valid_items = []
        below_table = []  # 被菜單區域下界排除的文字（只為了 log）
        for idx, (orig_box, corr_box, (text, score)) in enumerate(zipped_data):
            min_score = 0.3 if text.startswith("ねぎ") else 0.5
            if score < min_score:
                continue
            if idx in bad_box_indices:
                continue
            if idx in layout_marker_indices:
                continue

            cy = np.mean([pt[1] for pt in corr_box])
            cx = np.mean([pt[0] for pt in corr_box])
            h = abs(min(corr_box[3][1], corr_box[2][1]) - max(corr_box[0][1], corr_box[1][1]))

            if takeout_y is not None and takeout_x is not None:
                if not is_flip:
                    if cy < takeout_y: continue
                    if cx < (takeout_x[0] - 30) or cx > (takeout_x[1] + 40): continue
                else:
                    if cy > takeout_y: continue
                    if cx < (takeout_x[0] - 40) or cx > (takeout_x[1] + 30): continue

            if menu_bottom is not None and cy > menu_bottom:
                below_table.append(text)
                continue

            valid_items.append({
                'idx': idx,   # zipped_data 的 index；is_flip 時要換算回 dt_boxes 的原始 index（除錯圖用）
                'text': text,
                'score': score,
                'orig_box': orig_box,
                'cx': cx,
                'cy': cy,
                'h': h,
            })

        if below_table:
            logger.info(f"[菜單區域] 排除表格下方的文字框 {len(below_table)} 個: {below_table}")

        # 靜默失敗診斷: 全部文字框都被濾掉時要留下痕跡,否則後面 step 4 直接空轉,
        # log 只會看到一行空的解析結果,無從判斷是被 takeout 濾掉還是根本沒比對到菜名。
        if not valid_items and zipped_data:
            logger.warning(
                f"[訂單解析] 全部 {len(zipped_data)} 個文字框都被濾掉 -> 無品項可比對 "
                f"(takeout_y={takeout_y}, takeout_x={takeout_x}, is_flip={is_flip}, "
                f"bad_box={sorted(bad_box_indices)})"
            )

        # 3. 將同個高度的文字合併成同一列 (校正後座標讓 Y 軸對齊極度精準)
        rows = []
        for item in valid_items:
            placed = False
            for row in rows:
                avg_cy = sum(r['cy'] for r in row) / len(row)
                avg_h = sum(r['h'] for r in row) / len(row)
                if abs(item['cy'] - avg_cy) < avg_h * 0.5:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        # 3.5 把列**依傳票的上下順序**排好。
        #
        # rows 原本是「valid_items 的走訪順序」，也就是 OCR det 框的輸出順序，
        # 跟傳票上下無關。以前沒人在意，因為下游只把它當成一袋品項；但
        # 2026-09-21 起「修飾列」（変更しじみ汁 = 把上方最近那個套餐的湯換掉）
        # 要靠列的先後決定錨定對象，順序錯了就會錨到別的套餐。
        #
        # is_flip=True 代表傳票是顛倒的，傳票的第一列在影像上 cy 最大 ——
        # 這跟下面 row.sort(key=cx, reverse=is_flip) 是同一個道理。
        rows.sort(key=lambda r: float(np.mean([i['cy'] for i in r])), reverse=is_flip)

        # 4. 合併內容並與 menus_ticket 比對，進行三重機制判斷數量
        for row in rows:
            row.sort(key=lambda x: x['cx'], reverse=is_flip)
            merged_text = "".join([item['text'] for item in row])
            row_cy = np.mean([item['cy'] for item in row])
            avg_h = np.mean([item['h'] for item in row])

            # 嘗試拆出前綴數字
            prefix_m = re.match(r'^([1-9]\d*)(.+)', merged_text.strip())
            leading_digit = int(prefix_m.group(1)) if prefix_m else None

            # 比對品項名稱：先問等價層，沒命中才在折疊座標系上做模糊比對
            matched_str, match_score = self.folded.resolve(
                merged_text, self.menus_ticket, self.fuzzy_matcher
            )

            n_boxes = len(flip_dt)
            dbg_row = {
                "raw": merged_text,
                "name": matched_str,
                "score": float(match_score) if matched_str is not None else 0.0,
                "qty": None,
                "qty_src": "",
                "status": "no_match" if matched_str is None else "low_score",
                "box_idx": [(n_boxes - 1 - it['idx']) if is_flip else it['idx'] for it in row],
            }
            self.last_debug_rows.append(dbg_row)

            if matched_str is not None:
                if match_score > 40:
                    match_num = 1  # 預設初始化
                    qty_source = None  # 三個方法中哪一個真的讀到數量（None = 都沒有）

                    # --------------------------------------------------------
                    # [方法一：前綴數字]
                    # --------------------------------------------------------
                    if leading_digit is not None and matched_str not in self.menus_with_digits:
                        match_num = leading_digit
                        qty_source = "前綴數字"
                        logger.info(f"[數量判斷-方法一 前綴數字]: '{merged_text}' → 數量={match_num}")
                    else:
                        # --------------------------------------------------------
                        # [方法二：獨立數字框 + 菜名內建數字防禦機制]
                        # --------------------------------------------------------
                        if digit_boxes:
                            nearest_cy, nearest_val, nearest_raw_text = min(
                                digit_boxes, key=lambda x: abs(x[0] - row_cy)
                            )
                            
                            if abs(nearest_cy - row_cy) < avg_h * 0.8:
                                is_part_of_menu = False
                                
                                # 🛡️ 防禦檢查：如果品項名稱本來就含有數字 (例如 "18切牛舌")
                                if matched_str in self.menus_with_digits:
                                    # 檢查撈出來的數字框是否其實是品項字串的一部分 (例如抓到 1 或 8)
                                    if nearest_raw_text in matched_str:
                                        logger.warning(
                                            f"[數量判斷-防禦機制觸發]: 數字框 '{nearest_raw_text}' 疑似為內建品項 '{matched_str}' 的一部分。放棄方法二。"
                                        )
                                        is_part_of_menu = True
                                
                                if not is_part_of_menu:
                                    match_num = nearest_val
                                    qty_source = "獨立數字框"
                                    logger.info(f"[數量判斷-方法二 digit_boxes]: '{merged_text}' → 最近純數字框 val={nearest_val} → 數量={match_num}")

                        # --------------------------------------------------------
                        # [方法三：數量欄交叉定位 Crop 圖 (Fallback 最終防線)]
                        # --------------------------------------------------------
                        if match_num == 1:
                            # 效能優化：不再對 row 內每個元素重跑模糊比對，直接以最長文字區塊作為主要 row 計算高低的基準
                            menu_item = max(row, key=lambda x: len(x['text']))

                            if nn_x is not None:
                                # 1. 在擺正的邏輯座標系中，精準找出交叉點 (數量欄中心 X , 當前列中心 Y)
                                logical_cx = nn_x
                                logical_cy = menu_item['cy']

                                # 2. 透過逆向旋轉，將邏輯交點還原成原始傾斜影像上的真實像素座標 (real_cx, real_cy)
                                real_cx, real_cy = skew.inverse_correct_point(logical_cx, logical_cy)

                                # 3. 丟入真實影像座標進行切圖 (將 w 設為 menu_item['h']，修正長方形切圖問題)
                                alone_res, _ = self.rec_pred.predict_quantity(
                                    frame_crop,
                                    cx=real_cx,
                                    cy=real_cy,
                                    h=menu_item['h'],
                                    w=menu_item['h'],  # 強迫 predict_quantity 內部以正方形比例進行裁切
                                    is_flip=is_flip,
                                )
                                m2 = re.search(r'[1-9]\d*', alone_res)
                                if m2:
                                    match_num = int(m2.group())
                                    qty_source = "數量欄crop"
                                    logger.info(f"[數量判斷-方法三 rec_pred]: rec_pred 辨識結果='{alone_res}' → 數量={match_num}")
                                else:
                                    logger.warning(f"[數量判斷-方法三 rec_pred]: rec_pred 辨識結果='{alone_res}' → 無法取得數量，預設=1")
                            else:
                                logger.warning(f"[數量判斷-方法三 rec_pred]: nn_x 為 None，無法定位數量欄位，預設=1")

                    # --------------------------------------------------------
                    # [幽靈列防護]
                    # 傳票上方的描述字樣（「外帶」之類）或 det 切壞的殘框，偶爾
                    # 會勉強模糊比對到某道菜 —— 傳票菜名池裡有 並/小/大/特/・並
                    # 這種 1~2 字的名字（實測 747 筆裡有 36 筆），是誤配磁鐵：
                    # '小計' 對 '小' 有 66.7 分、'お持ち帰り' 對 'お新香(持帰り)'
                    # 有 61.5 分，都過得了上面 > 40 的門檻，於是憑空多一筆數量 1
                    # 的品項，貼紙永遠湊不到它，整單卡住。
                    #
                    # 判準用兩個彼此獨立的弱訊號取交集，才不會誤殺正常列：
                    #   1. 名稱只是「勉強像」（分數 < WEAK_MATCH_SCORE）。等價層
                    #      或完全命中都是 100 分，不受影響。
                    #   2. 三個方法都沒讀到數量。實測機上 log：方法三只要
                    #      「数」欄定位得到（nn_x 不是 None），126 次全部讀到
                    #      數字、0 次失敗 —— 真正的品項列幾乎不可能沒有數量證據。
                    # 正常列只要任一條不成立就會保留（有前導數量、有數字框、
                    # 數量欄讀得到、或名稱命中等價層），行為跟修改前一致。
                    if qty_source is None and match_score < WEAK_MATCH_SCORE:
                        dbg_row.update(status="ghost_dropped", qty=match_num)
                        logger.warning(
                            f"[幽靈列防護]: 丟棄疑似版面文字的列 -> '{merged_text}' "
                            f"(勉強對到 '{matched_str}' score={match_score:.1f} "
                            f"< {WEAK_MATCH_SCORE}，且三個方法都沒讀到數量)。"
                            f"若這其實是真品項，請調低 WEAK_MATCH_SCORE。"
                        )
                        continue

                    logger.info(
                        f"[訂單餐點確認]: OCR合併結果 -> '{merged_text}' 對應 -> "
                        f"'{matched_str}' (score={match_score:.1f}) ... "
                        f"數量: {match_num} (來源={qty_source or '無，預設1'})"
                    )
                    dbg_row.update(status="ok", qty=match_num, qty_src=qty_source or "無，預設1")
                    parsed_items.append((matched_str, match_num))

        logger.info(f"[訂單解析結果]: {parsed_items}, 編號: {order_number}")
        return parsed_items, order_number
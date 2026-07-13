from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from rapidfuzz import fuzz, process

from logic.text_utils import FuzzyMatcher
from utils.skew_corrector import SkewCorrector

ParsedItem = Tuple[str, int]  # (menu_name, quantity)


class OrderParser:
    def __init__(
        self,
        menus_ticket: List[str],
        menus_with_digits: Set[str],
        fuzzy_matcher: FuzzyMatcher,
        rec_pred,  # StandaloneRecDLA
    ):
        self.menus_ticket = menus_ticket
        self.menus_with_digits = menus_with_digits
        self.fuzzy_matcher = fuzzy_matcher
        self.rec_pred = rec_pred

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
        nn_x = None  # 擺正後的數量欄中心 X 座標
        nn_w = None  # 擺正後的數量欄寬度

        flip_res = list(reversed(rec_res)) if is_flip else rec_res
        flip_dt = list(reversed(dt_boxes)) if is_flip else dt_boxes

        # 傾斜校正 1：初始化校正器
        skew = SkewCorrector.from_dt_boxes(flip_dt)
        logger.info(f"[傾斜校正] 估計角度: {skew.theta_deg:.2f} 度")

        # 傾斜校正 2：一次性批次將所有 Bbox 擺正，避免在後續迴圈重複運算消耗 CPU
        corrected_dt_boxes = skew.correct_boxes(flip_dt)

        # 1. 找出標題、外帶框、數量欄，同時收集純數字框
        takeout_y = None
        takeout_x = None
        digit_boxes = []  # 儲存結構: (corrected_cy, digit_value, text)

        zipped_data = list(zip(flip_dt, corrected_dt_boxes, flip_res))

        for orig_box, corr_box, (text, score) in zipped_data:
            if score < 0.5:
                continue

            # 全面使用已校正的擺正座標進行幾何判斷
            cy = np.mean([pt[1] for pt in corr_box])
            x_coords = [pt[0] for pt in corr_box]
            cleaned_text = text.strip()

            # 收集純數字框 (放寬 Regex 至多位數，如 10)
            if re.fullmatch(r'[1-9]\d*', cleaned_text):
                digit_boxes.append((cy, int(cleaned_text), cleaned_text))

            # 判斷訂單編號
            if order_number is None:
                match_order_number = re.search(r'\d{3}$', cleaned_text)
                if match_order_number:
                    order_number = match_order_number.group()
                    if not is_flip:
                        takeout_x = (80, max(x_coords) - 60)
                        takeout_y = cy + 100
                    else:
                        takeout_x = (min(x_coords) + 60, frame_crop.shape[1] - 80)
                        takeout_y = cy - 100

            if cleaned_text == "枚":
                cleaned_text = "数"
            nn_match = process.extractOne(cleaned_text, ["数"], scorer=fuzz.token_set_ratio)
            if nn_match and nn_match[1] > 60:
                nn_x = np.mean(x_coords)
                nn_w = max(x_coords) - min(x_coords)

            take_out_match = process.extractOne(cleaned_text, ["ティクアウト"], scorer=fuzz.token_set_ratio)
            if take_out_match and take_out_match[1] > 60:
                takeout_y = cy
                takeout_x = (min(x_coords), max(x_coords))

        # 2. 過濾有效文字區塊 (排除外帶框外的雜訊)
        valid_items = []
        for orig_box, corr_box, (text, score) in zipped_data:
            min_score = 0.3 if text.startswith("ねぎ") else 0.5
            if score < min_score:
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

            valid_items.append({
                'text': text,
                'score': score,
                'orig_box': orig_box,
                'cx': cx,
                'cy': cy,
                'h': h,
            })

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

        # 4. 合併內容並與 menus_ticket 比對，進行三重機制判斷數量
        for row in rows:
            row.sort(key=lambda x: x['cx'], reverse=is_flip)
            merged_text = "".join([item['text'] for item in row])
            row_cy = np.mean([item['cy'] for item in row])
            avg_h = np.mean([item['h'] for item in row])

            # 嘗試拆出前綴數字
            prefix_m = re.match(r'^([1-9]\d*)(.+)', merged_text.strip())
            leading_digit = int(prefix_m.group(1)) if prefix_m else None

            # 模糊比對品項名稱
            match_result = self.fuzzy_matcher.match(merged_text, self.menus_ticket)

            if match_result:
                matched_str, match_score = match_result
                if match_score > 40:
                    match_num = 1  # 預設初始化

                    # --------------------------------------------------------
                    # [方法一：前綴數字]
                    # --------------------------------------------------------
                    if leading_digit is not None and matched_str not in self.menus_with_digits:
                        match_num = leading_digit
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
                                    logger.info(f"[數量判斷-方法三 rec_pred]: rec_pred 辨識結果='{alone_res}' → 數量={match_num}")
                                else:
                                    logger.warning(f"[數量判斷-方法三 rec_pred]: rec_pred 辨識結果='{alone_res}' → 無法取得數量，預設=1")
                            else:
                                logger.warning(f"[數量判斷-方法三 rec_pred]: nn_x 為 None，無法定位數量欄位，預設=1")

                    logger.info(f"[訂單餐點確認]: OCR合併結果 -> '{merged_text}' 對應 -> '{matched_str}' ... 數量: {match_num}")
                    parsed_items.append((matched_str, match_num))

        logger.info(f"[訂單解析結果]: {parsed_items}, 編號: {order_number}")
        return parsed_items, order_number
import cv2
import os
import numpy as np
import time
import math
import copy
from shapely.geometry import Polygon
from loguru import logger
import pyclipper
import threading
import queue
import concurrent.futures
from datetime import datetime
from utils.skew_corrector import affine_to_homogeneous, rotate90_cw_matrix, rotate180_matrix, rotate90_ccw_matrix

# 1. 取得當下時間，並格式化成字串 (例如: 20260702_105541)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
from AI import PaddleOCR_Det, PaddleOCR_Cls, PaddleOCR_Rec
from utils import dla_guard
from utils.perf_stats import ocr_begin_line
from logic.orientation import (
    AREA_RATIO_THRESHOLD as ORIENT_AREA_THRESHOLD,
    VERTICAL_ASPECT as ORIENT_VERTICAL_ASPECT,
    needs_quarter_turn,
)

def draw_det_bboxes(frame_crop, dt_boxes, save_path="det_debug.jpg"):
    """
    把 OCR det 到的 bboxes 畫在影像上並存檔
    
    Args:
        frame_crop: 原始影像 (numpy array)
        dt_boxes: OCR det 輸出的 bounding boxes，格式通常是 list of polygon points
        save_path: 輸出路徑
    """
    vis = frame_crop.copy()
    
    if dt_boxes is not None:
        for box in dt_boxes:
            box = np.array(box).astype(np.int32)
            
            if box.ndim == 2 and box.shape[1] == 2:
                # Polygon 格式 (4個點或更多點)
                cv2.polylines(vis, [box.reshape((-1, 1, 2))], 
                              isClosed=True, color=(0, 255, 0), thickness=2)
            elif box.ndim == 1 and len(box) == 4:
                # [x1, y1, x2, y2] 格式
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(save_path, vis)
    print(f"Saved to {save_path}, found {len(dt_boxes) if dt_boxes else 0} boxes")
    return vis

REC_INDEX_MAPPING = {
            4338: 16,   4339: 17,   4340: 18,   4341: 19,   4342: 20,   4343: 21,   
            4344: 22,   4345: 23,   4346: 24,   4347: 25,   4353: 32,   4354: 33,   
            4355: 34,   4356: 35,   4357: 36,   4358: 37,   4359: 38,   4360: 39,   
            4361: 40,   4362: 41,   4363: 42,   4364: 43,   4365: 44,   4366: 45,   
            4367: 46,   4368: 47,   4369: 49,   4370: 50,   4371: 51,   4372: 52,   
            4373: 53,   4374: 54,   4375: 55,   4376: 57,   4377: 62,   4378: 64,   
            4379: 65,   4380: 66,   4381: 67,   4382: 69,   4383: 70,   4384: 71,
            4385: 72,   4386: 73,   4387: 74,   4388: 75,   4389: 76,   4390: 77,   
            4391: 79,   4392: 80,   4393: 81,   4394: 82,   4395: 86,   4396: 87,
            420: 395,   # 井 -> 丼
            731: 2373,  # 千 -> 牛
            733: 2373,  # 午 -> 牛
            730: 735,   # 十 -> 半
            2424: 2423, # 王 -> 玉
            985: 4309,  # 墨 -> 黒
            684: 296,   # 力 -> カ
            1593: 2381, # 持 -> 特
            1691: 1696  # 教 -> 数
        }

_ROTATE_MATRIX_FN = {
    cv2.ROTATE_90_CLOCKWISE: rotate90_cw_matrix,
    cv2.ROTATE_180: rotate180_matrix,
    cv2.ROTATE_90_COUNTERCLOCKWISE: rotate90_ccw_matrix,
}

class TextSystemDLA:
    def __init__(self, det_path, cls_path, rec_path, dict_path):

        # 1. 初始化 DLA
        # 經 dla_guard：檔案有問題只記 log，檔案正常卻載不進去（APU 沒起來）才重開機
        self.det_model = dla_guard.load(PaddleOCR_Det, det_path, "OCR det")
        self.cls_model = dla_guard.load(PaddleOCR_Cls, cls_path, "OCR cls")
        self.rec_model = dla_guard.load(PaddleOCR_Rec, rec_path, "OCR rec")

        # 2. 載入字典
        self.character_str = ["<BLANK>"]  
        with open(dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        self.character_str.append(" ")

        # Detection 參數 (參考 Paddle 預設值)      
        self.det_thresh = 0.3
        self.det_box_thresh = 0.6
        self.det_unclip_ratio = 1.5
        self.det_max_candidates = 1000      # 官方 DBPostProcess 預設值
        self.det_score_mode = "fast"        # "fast" 或 "slow"
        self.det_box_type = "quad"          # "quad" 或 "poly"
        self.min_size = 3
        self.det_image_shape = [640, 640]
        
        # Classification 參數
        self.cls_thresh = 0.9
        self.cls_image_shape = [3, 48, 192]

        # Recognition 參數
        self.rec_image_shape = [3, 48, 320]

        # 90 度方向判斷的補救參數
        # ------------------------------------------------------------
        # _check_boxes_need_rotate_90 是純幾何判斷：框的 h/w >= 1.4 才算直式。
        # 兩個字的直式貼紙(玉子 / 豚汁)det 框近乎正方形 —— 實測長短比 1.03~1.05,
        # 過不了 1.4, 於是整張圖不轉正, 直立文字直接進 rec 只讀到一個垃圾字
        # (實測信心 0.15~0.22; 同一張轉 90 度後讀出「豚汁」信心 1.000)。
        # 三個字的「お新香」是 1.47, 也只擦過門檻 0.07, 這個判準對短字貼紙
        # 本來就站在邊緣。
        #
        # 正方形框在幾何上無解: 轉或不轉, 長短比都還是 ~1.0, 沒有任何幾何線索。
        # 所以改用 rec 信心當裁判 —— 對是 1.000、錯是 0.15~0.22, 差 5 倍,
        # 訊號比幾何強得多。只在「幾何說不用轉」+「框落在正方形區間」+
        # 「讀出來信心偏低」三個條件同時成立時才多跑一次, 正常 case 完全不受影響。
        #
        # 刻意不動 VERTICAL_ASPECT(1.4): 那是傳票方向調校(images/order_ticket
        # 99 張, 48.5% -> 100%)依賴的同一顆旋鈕, 往下調會換成傳票那邊崩掉。
        # orient_retry_conf_thresh 的取值依據(20260915 實測 179 張正方形框樣本):
        #   A) 轉 90 度後較好(真的需要補救) 176 張：原方向 mean conf
        #      min 0.000 / max 0.814 / p95 0.635
        #   B) 原方向本來就對(不該補救)       3 張：min 0.662 / max 0.959
        #   兩群在 0.662~0.814 重疊 —— 錯的方向也可能讀得很有自信(「数」0.814,
        #   同一張轉 90 度是「豚汁」1.000), 所以「信心高」不等於「方向對」,
        #   不存在一個乾淨的門檻。
        #
        #   門檻=0.5 -> 漏補救 24 張; 0.7 -> 4 張; 0.9 -> 0 張但只省下 2 次重跑。
        #   而「多跑一次」是無害的(predict 是比分數取高者, 錯方向不會被採用),
        #   「漏補救」則直接吐垃圾字 —— 代價完全不對稱。
        #   因此預設取 1.0: 正方形框一律比對, 只有讀到滿分(conf == 1.0)
        #   才省下轉 90 度那一組。傳票與明顯直式的框完全不受影響。
        #   若之後要省這段延遲, 把它調到 0.9 是有實測依據的折衷。
        #
        # 2026-10-02: 由「原方向 vs 順轉 90」兩方向改為四方向全比。
        #   mmr 的旋轉框分不出貼紙頭朝上或朝下, 所以正方形框實際上四個方向都可能。
        #   舊版的 180/270 只能靠 cls 補, 但兩個字的短框 cls 從來過不了
        #   cls_thresh(0.9)(當天 log 每次都是 0/1 -> is_flip=False), 等於只比了
        #   0 與 90 兩個方向; 貼紙倒著進來時兩個都是錯的, 只是從兩個錯的挑高分:
        #   實測讀出「た玉」0.68、「玉並」0.51~0.85, 而正確的「玉子」也有 0.75~0.83,
        #   分數區間重疊, 門檻擋不住, 只能真的把四個方向都讀過。
        #   180 度不重跑 det(框座標直接換算), 所以四方向只要 2 次 det + 4 次單框 rec。
        self.orient_ambiguous_aspect = 0.7      # 長短比落在 [0.7, 1.4) 視為方向不明
        self.orient_ambiguous_area_ratio = 0.5  # 方向不明的框需佔總面積多少才啟動補救
        self.orient_retry_conf_thresh = 1.0     # 原樣/轉180 的最佳信心低於此值, 才再比順轉/逆轉 90

        self._prof_reset()
        logger.info("模型初始化完成。")

    def _prof_reset(self):
        """
        這次讀取的計時（utils/perf_stats.py 的 [OCR#n ■] 那行）。det / rec 的次數不固定：
        轉 90、翻 180 會重跑 det，正方形貼紙會讀 2~4 個方向的 rec，所以次數跟耗時一起記，
        DLA 推論（model.run）另外拆出來，才分得出慢在模型還是 Python 的前後處理。
        """
        self.prof = {
            "path": [],
            "det_n": 0, "det_ms": 0.0, "det_pre_ms": 0.0, "det_infer_ms": 0.0,
            "cls_n": 0, "cls_ms": 0.0,
            "rec_rounds": 0, "rec_boxes": 0, "rec_ms": 0.0, "rec_infer_ms": 0.0,
        }

    def _save_debug_image(self, img, filename):
        """儲存 debug 圖片"""

        if len(img.shape) == 3 and img.shape[0] == 3:
            img_vis = img.transpose(1, 2, 0)
        else:
            img_vis = img.copy()

        if img_vis.dtype == np.float32 or img_vis.dtype == np.float64:
            img_min, img_max = img_vis.min(), img_vis.max()
            if img_max - img_min > 0:
                img_vis = (img_vis - img_min) / (img_max - img_min) * 255
            img_vis = img_vis.astype(np.uint8)

        if len(img_vis.shape) == 2:
            img_vis = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)

        cv2.imwrite(filename, img_vis)
        print(f"  Saved: {filename}")

    # ==========================
    # 1. Detection (文字偵測)
    # ==========================
    def _resize_image_det(self, img):
        """
        Detection 的圖片 resize：letterbox 等比例縮放
        - 縮放比例 ratio = min(target_w/ori_w, target_h/ori_h)，確保整張圖都能塞進目標尺寸，不被裁切
        - 縮放後貼齊左上角，右側/下側不足的部分補 0 (黑色)
        """
        target_h, target_w = self.det_image_shape
        ori_h, ori_w = img.shape[:2]

        ratio = min(target_w / ori_w, target_h / ori_h)
        new_w = int(round(ori_w * ratio))
        new_h = int(round(ori_h * ratio))

        resized = cv2.resize(img, (new_w, new_h))

        if img.ndim == 3:
            canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        canvas[:new_h, :new_w] = resized  # 貼左上角，不做置中

        # 回傳 ratio 而非 [ratio_h, ratio_w]：因為是等比例縮放，兩個方向縮放比例相同，
        # 且貼左上角沒有 offset，後處理還原座標時直接除以 ratio 即可。
        return canvas, ratio

    def _normalize_det(self, img):
        """Detection 的標準化"""
        scale = 1.0 / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype('float32')
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)).astype('float32')
        img = img.astype('float32')
        img = img * scale
        img = (img - mean) / std
        return img

    def _det_preprocess(self, img):
        """Detection 前處理"""
        ori_h, ori_w = img.shape[:2]
        resized_img, ratio = self._resize_image_det(img)
        if resized_img is None:
            return None, None

        normalized_img = self._normalize_det(resized_img)
        # shape 內容改成 [ori_h, ori_w, ratio]，取代原本的 [ori_h, ori_w, ratio_h, ratio_w]
        return normalized_img, [ori_h, ori_w, ratio]

    def unclip(self, box, unclip_ratio):
        """擴展文字框"""
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def _get_mini_boxes(self, contour):
        """取得最小外接矩形"""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _box_score_fast(self, bitmap, _box):
        """score_mode='fast'：用 box 的最小外接矩形範圍算平均分數"""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        從 bitmap 中提取文字框（對齊官方 DBPostProcess.boxes_from_bitmap，box_type='quad'）
        dest_width/dest_height: 還原座標用的目標尺寸（這裡傳入原圖的 ori_w/ori_h）
        """
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            contours = outs[1]
        elif len(outs) == 2:
            contours = outs[0]

        # 官方有 max_candidates 上限，避免雜訊輪廓過多拖慢速度
        num_contours = min(len(contours), self.det_max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]

            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            if self.det_score_mode == "fast":
                score = self._box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self._box_score_slow(pred, contour)
            if score < self.det_box_thresh:
                continue

            box = self.unclip(points, self.det_unclip_ratio)
            if len(box) == 0 or len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            # 還原回原始圖片尺寸
            # 注意：因為前處理是「letterbox 等比例縮放 + 左上貼齊」，
            # 所以這裡是用同一個 ratio 還原寬高，而不是官方原版的
            # 「x/width*dest_width、y/height*dest_height」各自獨立縮放公式
            # （官方那版是假設前處理用拉伸 resize，寬高縮放比例不同）。
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)

            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def _det_postprocess(self, pred, shape_list):
        """
        Detection 後處理（box_type 固定為 'quad'）
        shape_list 內容: [ori_h, ori_w, ratio]
        """
        if len(pred.shape) == 4:
            pred = pred[:, :, :, 0]

        segmentation = pred > self.det_thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            ori_h, ori_w, ratio = shape_list[batch_index]
            mask = segmentation[batch_index]

            # letterbox 前處理時整張圖貼在左上角，且推論圖（self.det_image_shape）
            # 跟 letterbox 後的 canvas 同尺寸，所以這裡傳入 dest_width/height 為
            # letterbox 後內容區域大小 (ori_w*ratio, ori_h*ratio)，
            # _boxes_from_bitmap 內部會再除以 ratio 換算回原圖尺寸的等效效果。
            # 簡化做法：直接把 dest_width/dest_height 設為原圖尺寸，
            # 並讓 width/height 改用「letterbox 內容區大小」而非整個 640x640，
            # 這樣 box[:, 0]/width*dest_width 就等於 box[:, 0]/ratio。
            content_w = int(round(ori_w * ratio))
            content_h = int(round(ori_h * ratio))

            # 把預測圖裁掉 padding 區域，只保留 letterbox 貼圖的內容範圍
            mask = mask[:content_h, :content_w]
            pred_crop = pred[batch_index][:content_h, :content_w]

            boxes, scores = self._boxes_from_bitmap(pred_crop, mask, ori_w, ori_h)
            boxes_batch.append({"points": boxes})
        return boxes_batch

    def order_points_clockwise(self, pts):
        """排序四個點為順時針順序"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        """裁剪檢測結果到圖片範圍內"""
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        """
        過濾檢測結果
        1. 排序為順時針 (統一為: [左上, 右上, 右下, 左下])
        2. 裁剪到圖片範圍內
        3. 過濾太小的框
        """
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 10 or rect_height <= 10:
                continue
   
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def get_rotate_crop_image(self, img, points):
        """根據 Detection 的座標，從原圖裁剪出每個文字區域"""
        assert len(points) == 4, "shape of points must be 4*2"

        # 計算目標裁剪區域的寬高
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])
            )
        )

        # 定義目標矩形 (正立的矩形)
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )

        # 透視變換 - 把傾斜的文字框拉正
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]

        # 如果高度 > 寬度 1.3 倍，代表是豎排文字，旋轉 90 度
        if dst_img_height * 1.0 / dst_img_width >= 1.3:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array): detected text boxes with shape [N, 4, 2]
        return:
            sorted boxes(array) with shape [N, 4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    # ====================
    def run_det(self, img):
        """執行文字檢測 (DLA 版)"""
        st1 = time.time()
        st1_pc = time.perf_counter()
        ori_im = img.copy()
        data, shape = self._det_preprocess(ori_im)
        if data is None:
            return np.array([]), 0

        # DLA 推論
        input_data = np.expand_dims(data, axis=0).astype(np.float16) 
        t_inf = time.perf_counter()
        self.prof["det_pre_ms"] += 1000 * (t_inf - st1_pc)   # resize 到 640x640 + 正規化
        flat_output = self.det_model.run(input_data)
        self.prof["det_infer_ms"] += 1000 * (time.perf_counter() - t_inf)
        pred = flat_output.reshape(1, self.det_image_shape[0], self.det_image_shape[1], 1)
        
        # 後處理
        post_result = self._det_postprocess(pred, [shape])
        dt_boxes = post_result[0]["points"]
        
        if len(dt_boxes) > 0:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et1 = time.time()
        self.prof["det_n"] += 1
        self.prof["det_ms"] += 1000 * (et1 - st1)
        return dt_boxes, et1 - st1


    # ==========================
    # 2. Classification (方向分類)
    # ==========================
    def _resize_norm_img_cls(self, img):
        """Classification 的圖片 resize 和標準化"""
        imgC, imgH, imgW_max = self.cls_image_shape
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = min(resized_w, imgW_max)

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        
        padding_im = np.zeros((imgH, imgW_max, imgC), dtype=np.float32)
        padding_im[:, 0:resized_w, :] = resized_image

        return padding_im

    def _cls_postprocess(self, prob_out):
        """Classification 後處理"""
        pred_idxs = prob_out.argmax(axis=1)
        decode_out = []
        label_list = ["0", "180"]  
        for i, idx in enumerate(pred_idxs):
            decode_out.append((label_list[idx], prob_out[i, idx]))
        return decode_out

    # ====================
    def run_cls(self, img_list):
        """執行方向分類"""
        st2 = time.time()
        
        flip_count = 0
        img_num = len(img_list)
        cls_res = [["", 0.0]] * img_num

        for idx, img in enumerate(img_list):

            norm_img = self._resize_norm_img_cls(img)
            norm_img = np.expand_dims(norm_img, axis=0).astype(np.float16)

            # DLA 推論
            flat_output = self.cls_model.run(norm_img)
            prob_out = flat_output.reshape(1, 2)

            cls_result = self._cls_postprocess(prob_out)
            label, score = cls_result[0]
            cls_res[idx] = [label, score]

            if "180" in label and score > self.cls_thresh:
                img_list[idx] = cv2.rotate(img, 1)
                flip_count += 1
        
        et2 = time.time()
        self.prof["cls_n"] += img_num
        self.prof["cls_ms"] += 1000 * (et2 - st2)
        return img_list, cls_res, flip_count, et2 - st2

    def _decide_flip(self, img_crop_list, cls_res, min_aspect=3.0):
        """
        判斷整張圖是否上下顛倒 (180 度)。

        為什麼不用「全體多數決」:
            單字框 (例如數量欄的 '1'、欄位標題 '数'、單號 '483') 旋轉 180 度後
            近似對稱,cls 幾乎不可能在 cls_thresh(0.9) 之上給出穩定判斷,
            所以這些框永遠投不出翻轉票。傳票上這種短框往往佔一半以上
            (實測 9 框中有 5 框),把它們算進分母會把比例稀釋到永遠不過半
            (4/9 = 0.44),結果整張圖不翻轉。

            但 run_cls 已經把每個框「各自」轉正了,所以文字仍然讀得正確,
            只有 dt_boxes / rec_res 的順序是顛倒的 —— 這會讓 order_parser
            以影像下緣的單號去算 takeout_y,進而把所有文字框濾光,
            最後靜默回傳空品項清單。

        因此改成只採計「夠長的文字框」(寬高比 >= min_aspect) 的票。
        若一個長框都沒有 (例如 sticker 只有名稱+編號兩個短框),
        則退回原本的全體多數決,維持舊行為。

        回傳 (is_flip, unanimous)。unanimous=False 代表長框的票有分歧(例如 1/2、
        3/4),呼叫端不該採信這次的 is_flip,改用 _decide_flip_by_rec 兩個方向
        都讀過再決定。沒有長框、退回全體多數決時一律回 unanimous=True(舊行為)。
        """
        long_votes = []
        for img, (label, score) in zip(img_crop_list, cls_res):
            h, w = img.shape[:2]
            if h <= 0:
                continue
            # 180 度旋轉不改變寬高比,所以用 run_cls 轉正後的 crop 計算也是安全的
            if (w / float(h)) < min_aspect:
                continue
            long_votes.append("180" in label and score > self.cls_thresh)

        all_flip = sum(
            1 for label, score in cls_res
            if "180" in label and score > self.cls_thresh
        )

        if not long_votes:
            ratio = all_flip / len(cls_res) if cls_res else 0.0
            is_flip = ratio > 0.5
            logger.warning(
                f"[翻轉投票] 無長文字框(寬高比>={min_aspect})可投票,退回全體多數決: "
                f"{all_flip}/{len(cls_res)} ratio={ratio:.2f} -> is_flip={is_flip}"
            )
            return is_flip, True

        ratio = sum(long_votes) / len(long_votes)
        is_flip = ratio > 0.5
        unanimous = ratio in (0.0, 1.0)
        logger.info(
            f"[翻轉投票] 長文字框 {sum(long_votes)}/{len(long_votes)} ratio={ratio:.2f} "
            f"-> {f'is_flip={is_flip}' if unanimous else '票數分歧，改比兩個方向的辨識信心'} "
            f"(全體 {all_flip}/{len(cls_res)}="
            f"{all_flip / len(cls_res) if cls_res else 0:.2f})"
        )
        return is_flip, unanimous

    @staticmethod
    def _long_box_rec_conf(rec_res, dt_boxes, min_aspect=3.0) -> float:
        """
        只取長文字框(寬高比 >= min_aspect)的平均 rec 信心。

        單字框('1'、'数'、單號)正著讀倒著讀信心都差不多,算進來只會把兩個
        方向的差距稀釋掉;倒著的日文長句才讀得出明顯落差。沒有長框就退回全部框。
        """
        confs = []
        for box, (_, conf) in zip(dt_boxes, rec_res):
            b = np.asarray(box, dtype=np.float64).reshape(-1, 2)
            w = float(np.linalg.norm(b[0] - b[1]))
            h = float(np.linalg.norm(b[0] - b[3]))
            if h > 0 and w / h >= min_aspect:
                confs.append(float(conf))
        if not confs:
            return TextSystemDLA._mean_rec_conf(rec_res)
        return sum(confs) / len(confs)

    def _decide_flip_by_rec(self, ori_im, dt_boxes, obj_type):
        """
        cls 的翻轉投票有分歧時:原樣與轉 180 度各讀一次,取長框辨識信心高的那個。

        2026-10-13 實驗機 log:長框票數平手(1/2、2/4)的 11 次裡 10 次判成
        「不翻」而解析出空品項 —— 這台的傳票九成是倒著進來的,而 cls 信心不到
        cls_thresh 的框一律被算成「不翻」票,平手時就偏向錯的那邊。
        cls 拿不定主意的時候,直接看哪個方向讀得出字比較可靠。

        跟 _predict_ambiguous 共用 _read_both_flips(不經 cls、180 度不重跑 det)。
        **只回傳方向**,不回傳這次讀到的字:這兩次讀取是為了比較而省略了 cls 逐框
        轉正、翻轉時也沒重跑 det,小框(單號)會讀得比正式流程差 —— 實測同一張票
        方向判對了,單號卻從 '4' 變成讀不到。方向定了之後照一般流程重讀。
        """
        reads = self._read_both_flips(ori_im, dt_boxes, obj_type)
        scored = [(self._long_box_rec_conf(rec, boxes), rec, flip) for rec, boxes, flip in reads]
        # max 遇到同分取先出現的,也就是原樣
        _, _, is_flip = max(scored, key=lambda s: s[0])

        summary = " | ".join(
            f"{'轉180' if flip else '原樣'} 長框信心 {conf:.3f}"
            f"「{'/'.join(t for t, _ in rec)[:40] or '∅'}」"
            for conf, rec, flip in scored
        )
        logger.info(f"[翻轉投票] 方向比對: {summary} -> is_flip={is_flip}")
        return is_flip


    # ==========================
    # 3. Recognition (文字識別)
    # ==========================
    def _resize_norm_img_rec(self, img):
        """單張圖片前處理"""
        imgC, imgH, imgW_max = self.rec_image_shape
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = min(resized_w, imgW_max)
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5

        padding_im = np.zeros((imgH, imgW_max, imgC), dtype=np.float32)
        padding_im[:, 0:resized_w, :] = resized_image

        return padding_im[np.newaxis, :]

    def _decode(self, preds_idx, preds_prob):
        """解碼"""
        result_list = []     
        for seq_idx, seq_prob in zip(preds_idx, preds_prob):
            char_list, conf_list = [], []
            prev_token = None
            for token, prob in zip(seq_idx, seq_prob):             
                if token != 0 and token != prev_token:
                    if token in REC_INDEX_MAPPING:
                        token = REC_INDEX_MAPPING[token]
                    char_list.append(self.character_str[token])
                    conf_list.append(prob)
                prev_token = token
            text = ''.join(char_list)
            score = sum(conf_list) / len(conf_list) if conf_list else 0.0
            result_list.append([text, score])
        return result_list

    # ====================
    def run_rec(self, img_list):

        st3 = time.time()       
        img_num = len(img_list)
        rec_res = [["", 0.0]] * img_num

        for idx, img in enumerate(img_list):
            norm_img = self._resize_norm_img_rec(img)
            norm_img = np.expand_dims(norm_img, axis=0).astype(np.float16)

            # DLA 推論
            t_inf = time.perf_counter()
            flat_output = self.rec_model.run(norm_img)
            self.prof["rec_infer_ms"] += 1000 * (time.perf_counter() - t_inf)
            # preds = flat_output.reshape(1, 40, 4401)
            preds = flat_output.reshape(1, 40, -1)

            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            rec_result = self._decode(preds_idx, preds_prob)
            rec_res[idx] = rec_result[0]

        et3 = time.time()
        self.prof["rec_rounds"] += 1
        self.prof["rec_boxes"] += img_num
        self.prof["rec_ms"] += 1000 * (et3 - st3)
        return rec_res, et3 - st3

    def _check_boxes_need_rotate_90(self, dt_boxes, ratio_thresh=1.4, vertical_ratio=None):
        """
        檢查文字框是否被誤判為直式框，代表整張圖被轉成 90 或 270 度。

        2026-09-10：判準由「直式框**數量**比例 >= 0.2」改為
        「直式框**面積加權**比例 >= 0.35」。原判準在正立票據上會被下半部
        金額區的一整排直式「円」與單字框推過門檻，images/order_ticket 99 張
        實測只有 48.5% 正確（51 張誤轉 90 度）；面積加權為 100% (99/99)，
        兩類分佈為 0.021~0.147 vs 0.529~0.743，門檻兩側各約 0.2 餘裕。
        判準與量測見 logic/orientation.py 與 tools/eval_orientation.py。

        vertical_ratio 保留為參數僅為相容舊呼叫端；傳入數值時會被當成
        面積加權門檻使用。
        """
        if dt_boxes is None or len(dt_boxes) == 0:
            return False
        thr = ORIENT_AREA_THRESHOLD if vertical_ratio is None else float(vertical_ratio)
        return needs_quarter_turn(dt_boxes,
                                  area_ratio_threshold=thr,
                                  vertical_aspect=ratio_thresh)
    
    def _get_box_dims(self, box):
        """
        計算四邊形 bbox 的寬高與字體大小估計值(short_side)。
        此計算方式只依賴 box 四個頂點的邊長關係,與整張圖片目前是否
        已經轉正(90/270)無關 —— 不論文字是橫排還是直排,
        short_side 恆代表「字體粗細/大小」這個維度。

        box: shape (4, 2), 依序為 [左上, 右上, 右下, 左下]
        """
        p0, p1, p2, p3 = box
        top = np.linalg.norm(p1 - p0)
        bottom = np.linalg.norm(p2 - p3)
        left = np.linalg.norm(p3 - p0)
        right = np.linalg.norm(p2 - p1)

        width = (top + bottom) / 2.0     # 文字排列方向長度
        height = (left + right) / 2.0    # 垂直於排列方向的長度(字體大小)
        short_side = min(width, height)
        long_side = max(width, height)
        return width, height, short_side, long_side

    def _filter_sticker_boxes(self, dt_boxes, height_ratio_threshold=0.75, min_keep=1):
        """
        Sticker 專用過濾:去除字體明顯較小的描述性文字 bbox,
        只保留代表「菜單名稱/編號」的大字 bbox。

        放置時機建議:det 完成(含 90/270 校正後的 det)之後、
        cls 方向分類之前 —— 這樣可以避免大量小字 bbox 干擾
        cls 的翻轉比例判斷(is_flip),進而避免後續 y 軸最小值
        判斷時取錯資訊。

        Args:
            dt_boxes: np.ndarray, shape (N, 4, 2)
            height_ratio_threshold: 保留門檻比例,預設 0.75
                (實務上描述小字大約只有主標題的 0.5~0.7 倍高度,
                設 0.75 可以在保留同高度的「名稱+編號」的同時濾掉小字。
                這個值目測而來,建議之後用實際 sticker 樣本統計調整)
            min_keep: 最少保留幾個 box,避免過濾後一個都不剩

        Returns:
            np.ndarray, shape (M, 4, 2), M <= N
        """
        if dt_boxes is None or len(dt_boxes) <= 2:
            # box 數量太少(通常就是名稱+編號兩框),不需要過濾避免誤殺
            return dt_boxes

        short_sides = np.array([
            self._get_box_dims(box)[2] for box in dt_boxes
        ])

        max_short_side = short_sides.max()
        if max_short_side <= 0:
            return dt_boxes

        keep_mask = short_sides >= (max_short_side * height_ratio_threshold)

        if keep_mask.sum() < min_keep:
            top_idx = np.argsort(-short_sides)[:min_keep]
            keep_mask[:] = False
            keep_mask[top_idx] = True

        filtered_boxes = dt_boxes[keep_mask]

        logger.warning(
            f"[OCR_ENGINE] filter dt_boxes: {len(dt_boxes)} -> {len(filtered_boxes)} "
            f"[OCR_ENGINE] (max_short_side={max_short_side:.1f}, threshold={height_ratio_threshold})"
        )
        return filtered_boxes
    
    def _remove_table_lines(self, img):
        """
        針對 ticket 類型,偵測表格中的長直線/橫線格線並塗白,
        避免格線被誤判成文字(例如跟 '1' 混淆)。
        只作用在傳入的 img 副本上,不會影響外部的 ori_im。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        h, w = gray.shape[:2]

        # 水平線: kernel 寬度跟圖寬成比例,只留下夠長的橫線
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(w // 20, 15), 1)
        )
        horizontal_lines = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )

        # 垂直線: kernel 高度跟圖高成比例
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(h // 20, 15))
        )
        vertical_lines = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1
        )

        lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

        # 膨脹一點,確保線條邊緣殘影也被蓋住
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines_mask = cv2.dilate(lines_mask, dilate_kernel, iterations=1)

        result = img.copy()
        if result.ndim == 3:
            result[lines_mask > 0] = (255, 255, 255)
        else:
            result[lines_mask > 0] = 255

        return result

    def _postprocess_and_crop(self, img, dt_boxes, obj_type):
        """
        共用邏輯: sticker 過濾 -> 排序 -> 裁切
        (det 呼叫時機在 predict 裡不同分支不一樣，所以不包在這個 helper 裡)
        """
        if obj_type == "sticker":
            dt_boxes = self._filter_sticker_boxes(dt_boxes)
        # if obj_type == "ticket":
        #     img = self._remove_table_lines(img)  # 只塗白,不動 box 座標
        #     cv2.imwrite("wihteeee.jpg", img)

        dt_boxes = self.sorted_boxes(dt_boxes)
        img_crop_list = [
            self.get_rotate_crop_image(img, box.astype(np.float32))
            for box in dt_boxes
        ]
        return dt_boxes, img_crop_list

    # ========================= #
    # Main Pipeline             #
    # ========================= #
    def _orientation_is_ambiguous(self, dt_boxes) -> bool:
        """
        文字框近乎正方形 —— 幾何上無法判斷這張圖是不是被轉了 90 度。

        直式(h/w >= VERTICAL_ASPECT)已由 _check_boxes_need_rotate_90 處理掉,
        明顯橫式(h/w < orient_ambiguous_aspect)本來就是正立的; 剩下中間那段
        是正方形區間, 轉與不轉在幾何上完全等價。
        跟 needs_quarter_turn 一樣用面積加權, 避免一堆小的單字框主導判斷。
        """
        if dt_boxes is None or len(dt_boxes) == 0:
            return False

        area_ambiguous = 0.0
        area_total = 0.0
        for box in dt_boxes:
            b = np.asarray(box, dtype=np.float64).reshape(-1, 2)
            w = float(np.linalg.norm(b[0] - b[1]))
            h = float(np.linalg.norm(b[0] - b[3]))
            if w <= 0.0 or h <= 0.0:
                continue
            area = w * h
            area_total += area
            if self.orient_ambiguous_aspect <= (h / w) < ORIENT_VERTICAL_ASPECT:
                area_ambiguous += area

        if area_total <= 0.0:
            return False
        return (area_ambiguous / area_total) >= self.orient_ambiguous_area_ratio

    @staticmethod
    def _mean_rec_conf(rec_res) -> float:
        """這次讀出來的平均信心,當作「這個方向對不對」的分數"""
        if not rec_res:
            return 0.0
        confs = [float(c) for _, c in rec_res]
        return sum(confs) / len(confs)

    def _recognize(self, ori_im, dt_boxes, obj_type):
        """
        det 之後的共用流程: sticker 過濾 + 裁切 -> cls 方向分類 ->
        (必要時轉 180 度重新 det) -> rec。

        抽成獨立方法是為了讓 predict 的「原方向」與「轉 90 度」兩條路徑
        跑完全相同的一段邏輯,不要複製兩份而各自漂移。
        (正方形框不走這裡, 見 _predict_ambiguous —— 那邊四個方向都明確讀過,
        不需要 cls 判斷 180 度。)
        """
        # 2. sticker 過濾 + 排序 + 裁剪 (共用邏輯)
        raw_dt_boxes = dt_boxes   # 方向比對要從原始 det 結果重新過濾/裁切, 見 _decide_flip_by_rec
        dt_boxes, img_crop_list = self._postprocess_and_crop(ori_im, dt_boxes, obj_type)

        # 3. 方向分類
        img_crop_list, cls_res, flip_count, elapse2 = self.run_cls(img_crop_list)
        # 只採計長文字框的方向票,避免單字框(1/数/483)把翻轉比例稀釋到永遠不過半
        is_flip, unanimous = self._decide_flip(img_crop_list, cls_res)
        if not unanimous:
            # 票數分歧時 cls 的結論不可靠,兩個方向都讀過、比辨識信心決定方向;
            # 之後照一般流程讀(翻轉就重跑 det + cls)
            self.prof["path"].append("翻轉比對")
            is_flip = self._decide_flip_by_rec(ori_im, raw_dt_boxes, obj_type)

        # 4. 若偵測到翻轉，旋轉影像 180 度後重新執行偵測與識別
        if is_flip:
            self.prof["path"].append("轉180重讀")
            rotated_im = cv2.rotate(ori_im, cv2.ROTATE_180)

            dt_boxes, _ = self.run_det(rotated_im)

            if len(dt_boxes) == 0:
                return [], [], is_flip

            # 180 度重新 det 後,同樣需要再過濾一次 sticker,理由相同
            dt_boxes, img_crop_list = self._postprocess_and_crop(rotated_im, dt_boxes, obj_type)

            img_crop_list, cls_res, _, _ = self.run_cls(img_crop_list)

        # 5. 文字識別
        rec_res, elapse3 = self.run_rec(img_crop_list)

        return rec_res, dt_boxes, is_flip

    @staticmethod
    def _rotate_boxes_180(dt_boxes, img_w, img_h):
        """
        把 det 框換算到 cv2.rotate(img, ROTATE_180) 後的座標系, 省下重跑一次 det。
        (x, y) -> (w-1-x, h-1-y), 與 utils.skew_corrector.rotate180_matrix 一致。
        轉完後原本的左上角落在右下, 所以頂點往後挪兩格, 維持
        [左上, 右上, 右下, 左下] 的順序 —— get_rotate_crop_image 靠這個順序決定裁切方向。
        """
        boxes = np.asarray(dt_boxes).reshape(-1, 4, 2).copy()
        boxes[..., 0] = (img_w - 1) - boxes[..., 0]
        boxes[..., 1] = (img_h - 1) - boxes[..., 1]
        return np.roll(boxes, 2, axis=1)

    def _read_both_flips(self, img, dt_boxes, obj_type):
        """
        對同一份 det 結果讀「原樣」與「轉 180 度」兩個方向, 不經過 cls。
        回傳 [(rec_res, dt_boxes, is_flip), ...] 兩筆; is_flip=True 那筆的
        dt_boxes 落在 cv2.rotate(img, ROTATE_180) 的座標系, 與 _recognize
        翻轉時回傳的座標系語意相同。
        """
        h, w = img.shape[:2]
        reads = []
        for is_flip in (False, True):
            if is_flip:
                src = cv2.rotate(img, cv2.ROTATE_180)
                boxes = self._rotate_boxes_180(dt_boxes, w, h)
            else:
                src, boxes = img, dt_boxes
            boxes, crops = self._postprocess_and_crop(src, boxes, obj_type)
            rec_res, _ = self.run_rec(crops)
            reads.append((rec_res, boxes, is_flip))
        return reads

    def _predict_ambiguous(self, ori_im, dt_boxes, obj_type):
        """
        框近乎正方形(幾何無解)時, 四個方向都讀過, 取平均 rec 信心最高者。
        cls 在兩個字的短框上過不了 cls_thresh, 所以 180 度不交給 cls,
        而是跟 90 度一樣明確讀一次來比。

        回傳 (rec_res, dt_boxes, is_flip, is_rotated_90), 四種組合對應
        _worker_loop 的 rot_code: 原樣 / ROTATE_180 / ROTATE_90_CLOCKWISE /
        ROTATE_90_COUNTERCLOCKWISE(= 順轉 90 再轉 180)。
        """
        names = {(False, False): "原樣", (False, True): "轉180",
                 (True, False): "順轉90", (True, True): "逆轉90"}

        # (is_rotated_90, rec_res, dt_boxes, is_flip); max 遇到同分取先出現的, 所以平手時偏好原樣
        cands = [(False,) + r for r in self._read_both_flips(ori_im, dt_boxes, obj_type)]
        best = max(cands, key=lambda c: self._mean_rec_conf(c[1]))

        if self._mean_rec_conf(best[1]) < self.orient_retry_conf_thresh:
            self.prof["path"].append("含90度")
            rotated_im = cv2.rotate(ori_im, cv2.ROTATE_90_CLOCKWISE)
            r_dt_boxes, _ = self.run_det(rotated_im)
            if len(r_dt_boxes) > 0:
                cands += [(True,) + r for r in self._read_both_flips(rotated_im, r_dt_boxes, obj_type)]
                best = max(cands, key=lambda c: self._mean_rec_conf(c[1]))

        summary = " | ".join(
            f"{names[(rot90, flip)]} "
            f"{'/'.join(t for t, _ in rec) or '∅'}({self._mean_rec_conf(rec):.3f})"
            for rot90, rec, _, flip in cands
        )
        is_rotated_90, rec_res, dt_boxes, is_flip = best
        logger.info(
            f"[方向補救] 文字框近乎正方形, 方向比對: {summary} "
            f"-> 採用 {names[(is_rotated_90, is_flip)]}"
        )
        return rec_res, dt_boxes, is_flip, is_rotated_90

    def predict_oriented(self, image_input, rotate_cw: int, obj_type="unknown"):
        """方向已知時的 det -> rec：轉正後只跑一次 det、一輪 rec。

        給吉野家大傳票開 CV 輔助時用 —— 方向由表格版面決定（logic/yoshinoya/ticket_cv_layout.py，
        数欄在左 = 正向），不做 90 度檢查、cls 翻轉投票、翻轉比對與轉 180 重讀
        （傳票常倒著放，原本幾乎每張都走「翻轉比對+轉180重讀」：det 2 次、rec 3 輪）。
        cls 也不跑：整張已經轉正，cls 只會偶爾把單一框誤翻。

        rotate_cw: 原圖要順時針轉幾度才是正向（0/90/180/270）。
        回傳格式同 predict；(is_flip, is_rotated_90) 依 AsyncOCR 的慣例表示這個轉法，
        dt_boxes 是轉正後影像上的座標，AsyncOCR 照舊依這兩個旗標把 crop 轉正。
        """
        st = time.time()
        self._prof_reset()
        self.prof["path"].append("CV定向")
        rotate_cw %= 360
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate_cw)
        img = image_input if code is None else cv2.rotate(image_input, code)
        is_flip, is_rotated_90 = rotate_cw in (180, 270), rotate_cw in (90, 270)

        dt_boxes, _ = self.run_det(img)
        if len(dt_boxes) == 0:
            return [], [], is_flip, is_rotated_90, time.time() - st
        dt_boxes, img_crop_list = self._postprocess_and_crop(img, dt_boxes, obj_type)
        rec_res, _ = self.run_rec(img_crop_list)
        return rec_res, dt_boxes, is_flip, is_rotated_90, time.time() - st

    def predict(self, image_input, obj_type="unknown"):
        """主要預測流程"""
        st = time.time()
        self._prof_reset()

        ori_im = image_input.copy()
        img_to_detect = ori_im
        is_rotated_90 = False

        # 1. Detection
        dt_boxes, elapse1 = self.run_det(img_to_detect)
        # logger.info(f"dt_boxes num: {len(dt_boxes)}, elapsed: {elapse1:.3f}s")

        if len(dt_boxes) == 0:
            return [], [], False, is_rotated_90, time.time() - st

        # 1.5 檢查是否整張圖被轉成 90 / 270 度（大量直式文字框）
        if self._check_boxes_need_rotate_90(dt_boxes):
            self.prof["path"].append("轉90")
            ori_im = cv2.rotate(ori_im, cv2.ROTATE_90_CLOCKWISE)
            is_rotated_90 = True

            dt_boxes, elapse1 = self.run_det(ori_im)
            if len(dt_boxes) == 0:
                return [], [], False, is_rotated_90, time.time() - st

            rec_res, dt_boxes, is_flip = self._recognize(ori_im, dt_boxes, obj_type)
            return rec_res, dt_boxes, is_flip, is_rotated_90, time.time() - st

        # 1.6 方向補救：框近乎正方形（幾何無解）時，四個方向都讀過、取信心最高者。
        #     是否落在正方形區間要用「轉正前」的框判斷。
        if self._orientation_is_ambiguous(dt_boxes):
            self.prof["path"].append("正方形多向")
            rec_res, dt_boxes, is_flip, is_rotated_90 = self._predict_ambiguous(
                ori_im, dt_boxes, obj_type
            )
            return rec_res, dt_boxes, is_flip, is_rotated_90, time.time() - st

        # 幾何判定「不用轉」：照原方向讀，180 度交給 cls。
        rec_res, dt_boxes, is_flip = self._recognize(ori_im, dt_boxes, obj_type)

        et = time.time()

        return rec_res, dt_boxes, is_flip, is_rotated_90, et - st


class StandaloneRecDLA:
    def __init__(self, rec_path, dict_path, rec_image_shape=[3, 48, 320]):

        self.rec_image_shape = rec_image_shape
        self.count = 0
        # 從開機起累計的呼叫次數 / 耗時；app.py 在 apply_ocr_result 前後相減，
        # 印在 [OCR#n ■ ticket] 那行的「數量欄 rec」
        self.prof_calls = 0
        self.prof_ms = 0.0
        self.rec_model = dla_guard.load(PaddleOCR_Rec, rec_path, "數量欄 rec (standalone)")

        # 載入字典
        self.character_str = ["<BLANK>"]  
        with open(dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        self.character_str.append(" ")

    def _save_debug_image(self, img, filename):
        """儲存 debug 圖片"""

        if len(img.shape) == 3 and img.shape[0] == 3:
            img_vis = img.transpose(1, 2, 0)
        else:
            img_vis = img.copy()
        
        if img_vis.dtype == np.float32 or img_vis.dtype == np.float64:
            img_min, img_max = img_vis.min(), img_vis.max()
            if img_max - img_min > 0:
                img_vis = (img_vis - img_min) / (img_max - img_min) * 255
            img_vis = img_vis.astype(np.uint8)

        if len(img_vis.shape) == 2:
            img_vis = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)
  
        cv2.imwrite(filename, img_vis)
        print(f"  Saved: {filename}")
  

    def get_rotate_crop_image(self, img, points):
        """根據 Detection 的座標，從原圖裁剪出每個文字區域"""

        img_crop_width = int(
            max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
        )
        img_crop_height = int(
            max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
        )

        pts_std = np.float32(
            [[0, 0],
             [img_crop_width, 0],
             [img_crop_width, img_crop_height],
             [0, img_crop_height]]
        )

        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )

        return dst_img

    def _resize_norm_img(self, img):
        """單張圖片前處理"""
        imgC, imgH, imgW_max = self.rec_image_shape
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = min(resized_w, imgW_max)
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5

        padding_im = np.zeros((imgH, imgW_max, imgC), dtype=np.float32)
        padding_im[:, 0:resized_w, :] = resized_image

        return padding_im[np.newaxis, :]

    def _decode(self, preds_idx, preds_prob):
        """解碼"""
        char_list = []
        conf_list = []
        prev_token = None

        for token, prob in zip(preds_idx, preds_prob):
            if token == 0:
                prev_token = token
                continue
            
            if token != prev_token:
                char_list.append(self.character_str[token])
                conf_list.append(prob)
                
            prev_token = token

        text = ''.join(char_list)
        score = sum(conf_list) / len(conf_list) if conf_list else 0.0

        return text, score

    def predict_text(self, img: np.ndarray, points) -> tuple:
        """一般文字辨識（不限數字），回傳 (文字, 信心)。

        給吉野家大傳票的 CV 輔助補讀 det 漏框的品名 / 單號用（logic/yoshinoya/ticket_cv_assist.py）。
        predict / predict_quantity 只解碼 1~9，讀不了菜名。
        解碼跟主流程 TextSystemDLA._decode 一樣套 REC_INDEX_MAPPING（井→丼 這類易混字修正）。
        """
        t_prof = time.perf_counter()
        crop = self.get_rotate_crop_image(img, np.asarray(points, dtype=np.float32))
        preds = self.rec_model.run(self._resize_norm_img(crop)).reshape(1, 40, -1)
        char_list, conf_list, prev_token = [], [], None
        for token, prob in zip(preds.argmax(axis=2)[0], preds.max(axis=2)[0]):
            if token != 0 and token != prev_token:
                if token in REC_INDEX_MAPPING:
                    token = REC_INDEX_MAPPING[token]
                char_list.append(self.character_str[token])
                conf_list.append(prob)
            prev_token = token
        self.count += 1
        self.prof_calls += 1
        self.prof_ms += 1000 * (time.perf_counter() - t_prof)
        return ''.join(char_list), (float(sum(conf_list) / len(conf_list)) if conf_list else 0.0)

    def predict(self, frame_crop, dt_box, is_flip, nn_x=None):
        """
        傳入單張事先裁切好的圖片
        """
        t_prof = time.perf_counter()
        if not nn_x:
            if not is_flip:
                dt_box[0][0] -= 35
                dt_box[1][0] = dt_box[0][0] + 45
                dt_box[3][0] -= 35
                dt_box[2][0] = dt_box[3][0] + 45
            else:
                dt_box[1][0] += 35
                dt_box[0][0] = dt_box[1][0] - 45
                dt_box[2][0] +=35
                dt_box[3][0] = dt_box[2][0] - 45
        else:
            if not is_flip:
                dt_box[1][0] = dt_box[0][0]
                dt_box[2][0] = dt_box[3][0]
                dt_box[0][0] = nn_x - 10
                dt_box[3][0] = nn_x - 10
            else:
                dt_box[0][0] = dt_box[1][0]
                dt_box[3][0] = dt_box[2][0]
                dt_box[1][0] = nn_x + 10
                dt_box[2][0] = nn_x + 10

        dt_box[0][1] -= 2
        dt_box[1][1] -= 2
        dt_box[2][1] += 2
        dt_box[3][1] += 2

        dt_crop_img = self.get_rotate_crop_image(frame_crop, dt_box)

        if is_flip:
            dt_crop_img = cv2.rotate(dt_crop_img, 1)

        # cv2.imwrite(f"dt_crop_{self.count}.jpg", dt_crop_img)    

        norm_img = self._resize_norm_img(dt_crop_img)
        # self._save_debug_image(norm_img[0], f"dt_crop_resize_{self.count}.jpg")
        
        flat_output = self.rec_model.run(norm_img)
        # preds = flat_output.reshape(1, 40, 4401)
        preds = flat_output.reshape(1, 40, -1)

        allowed_indices = [0] + list(range(17, 26))
        allowed_preds = preds[0][:, allowed_indices]
        allowed_indices_arr = np.array(allowed_indices)

        local_preds_idx = allowed_preds.argmax(axis=1)
        preds_prob = allowed_preds.max(axis=1) 
        preds_idx = allowed_indices_arr[local_preds_idx]

        decode_out = self._decode(preds_idx, preds_prob)

        self.count += 1
        self.prof_calls += 1
        self.prof_ms += 1000 * (time.perf_counter() - t_prof)

        return decode_out

    def predict_quantity(self, frame_crop: np.ndarray, cx: float, cy: float, h: float, w: float, is_flip: bool):
        t_prof = time.perf_counter()
        half_h = int(h / 2 * 1.1)
        half_w = int(w / 2 * 1.0)
        cx = int(cx) + 10  # 往右偏移
        cy = int(cy)

        x1 = max(cx - half_w, 0)
        x2 = min(cx + half_w, frame_crop.shape[1])
        y1 = max(cy - half_h, 0)
        y2 = min(cy + half_h, frame_crop.shape[0])

        debug_crop = frame_crop[y1:y2, x1:x2]

        if is_flip:
            debug_crop = cv2.rotate(debug_crop, 1)

        # 存 debug 圖
        try:
            debug_dir_path = f"debug_imgs/crop_quantity_{current_time}"
            debug_path = os.path.join(debug_dir_path, f"debug_qty_{self.count}.jpg")
            # os.makedirs(debug_dir_path, exist_ok=True)
            # cv2.imwrite(debug_path, debug_crop)
            print(f"[predict_quantity] 已存 crop 圖 → {debug_path}")
        except Exception as e:
            print(f"[predict_quantity] 存圖失敗: {e}")

        norm_img = self._resize_norm_img(debug_crop)

        flat_output = self.rec_model.run(norm_img)
        # preds = flat_output.reshape(1, 40, 4401)
        preds = flat_output.reshape(1, 40, -1)

        allowed_indices = [0] + list(range(17, 26))
        allowed_preds = preds[0][:, allowed_indices]
        allowed_indices_arr = np.array(allowed_indices)

        local_preds_idx = allowed_preds.argmax(axis=1)
        preds_prob = allowed_preds.max(axis=1)
        preds_idx = allowed_indices_arr[local_preds_idx]

        decode_out = self._decode(preds_idx, preds_prob)
        self.count += 1
        self.prof_calls += 1
        self.prof_ms += 1000 * (time.perf_counter() - t_prof)

        return decode_out


class AsyncOCR:
    def __init__(self, det_path, cls_path, rec_path, dict_path, result_callback=None, ocr_routes=None,
                 ticket_cv_orient: bool = False):
        """
        Args:
            result_callback: 處理完成後的回調函數，格式: func(rec_res, dt_boxes, time_cost, metadata)
            ticket_cv_orient: 大傳票（ticket / ticket_extra）先用 CV 版面決定方向再 det
                              （吉野家 ticket_cv_assist，見 _cv_ticket_layout）
        """
        self.ocr_sys = TextSystemDLA(det_path, cls_path, rec_path, dict_path)
        from logic.ocr_routing import validate_routes
        self.ocr_routes = validate_routes(ocr_routes)
        self.ticket_cv_orient = bool(ticket_cv_orient)
        self.input_queue = queue.Queue(maxsize=1)
        
        # 狀態控制
        self.running = False
        self._busy = False            
        self._lock = threading.Lock() 
        
        self.result_callback = result_callback
        self.worker_thread = None
        self._job_seq = 0   # [OCR#n] 的 n，本次開機的流水號

    @property
    def is_busy(self):
        """外部查詢：目前 OCR 是否正在忙碌"""
        with self._lock:
            return self._busy

    def start(self):
        """啟動 OCR 線程"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("OCR Service Started (Waiting for requests...)")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()

    def request_ocr(self, frame_crop, metadata=None):
        """
        請求進行 OCR。
        如果 OCR 目前是閒置的 -> 接收請求，回傳 True
        如果 OCR 目前是忙碌的 -> 拒絕請求，回傳 False (主程式可以直接略過)
        """
        with self._lock:
            if self._busy:
                return False 
            self._busy = True
        if metadata is not None:
            metadata.setdefault("trace", {})["req_mono"] = time.monotonic()
        try:
            self.input_queue.put((frame_crop.copy(), metadata), block=False)
            return True
        except queue.Full:
            with self._lock:
                self._busy = False
            return False

    @staticmethod
    def _cv_ticket_layout(frame_crop):
        """吉野家大傳票的 CV 版面（見 logic/yoshinoya/ticket_cv_layout.py），回傳 (版面, 給 [OCR#n] 路徑的註記)。

        方向可信才拿來跳過 det 的翻轉比對；找不到表格或方向不可信時照舊流程，
        版面仍交給解析器（找不到表格的話解析器直接走 legacy，不用再算一次）。
        """
        t = time.perf_counter()
        try:
            from logic.yoshinoya.ticket_cv_layout import analyze
            layout = analyze(frame_crop)
        except Exception as e:  # 實驗功能出錯不能讓整張傳票讀不到
            logger.exception(f"[CV定向] 版面分析發生例外，照舊流程: {e}")
            return None, "CV版面例外"
        ms = 1000 * (time.perf_counter() - t)
        if not layout.ok:
            return layout, f"CV找不到表格{ms:.0f}ms"
        if not layout.orientation_ok:
            logger.info(f"[CV定向] 方向不可信（{layout.orientation_note}），照舊流程")
            return layout, f"CV方向不可信{ms:.0f}ms"
        return layout, f"CV版面{ms:.0f}ms(轉{layout.rotate})"

    def _worker_loop(self):
        """後台工作迴圈"""
        while self.running:
            try:
                input_data = self.input_queue.get(timeout=0.1)
                frame_crop, metadata = input_data
                from logic.ocr_routing import backend_for
                backend = backend_for(self.ocr_routes, metadata['type'])
                metadata['text_backend'] = backend
                self._job_seq += 1
                trace = metadata.setdefault('trace', {})
                trace['job'] = self._job_seq
                trace['start_mono'] = time.monotonic()
                trace['crop'] = f"{frame_crop.shape[1]}x{frame_crop.shape[0]}"
                logger.info(ocr_begin_line(metadata))
                if backend != 'det':
                    from logic.szm.cv_text_regions import extract_regions
                    self.ocr_sys._prof_reset()
                    self.ocr_sys.prof["path"].append(backend)
                    t0 = time.perf_counter()
                    upright, boxes, transform = extract_regions(frame_crop, backend)
                    crops = [self.ocr_sys.get_rotate_crop_image(upright, b.copy()) for b in boxes]
                    # Table orientation is resolved geometrically; ink regions retain CLS.
                    if crops and backend == 'cv_ink':
                        crops, _, _, _ = self.ocr_sys.run_cls(crops)
                    rec_res, _ = self.ocr_sys.run_rec(crops)
                    metadata['M_inv_full'] = np.linalg.inv(transform @ affine_to_homogeneous(metadata['M']))
                    trace['end_mono'] = time.monotonic()
                    trace['prof'] = dict(self.ocr_sys.prof, path=list(self.ocr_sys.prof["path"]))
                    if self.result_callback:
                        self.result_callback(upright, rec_res, boxes, False,
                                             time.perf_counter()-t0, metadata)
                    self.input_queue.task_done()
                    with self._lock:
                        self._busy = False
                    continue
                M_inv = metadata['M_inv']
                M = metadata['M']
                h0, w0 = frame_crop.shape[:2]

                t0 = time.perf_counter()
                layout, cv_note = None, None
                if self.ticket_cv_orient and metadata['type'] in ('ticket', 'ticket_extra'):
                    layout, cv_note = self._cv_ticket_layout(frame_crop)
                if layout is not None and layout.ok and layout.orientation_ok:
                    rec_res, dt_boxes, is_flip, is_rotated_90, time_cost = self.ocr_sys.predict_oriented(
                        frame_crop, layout.rotate, obj_type=metadata['type'])
                else:
                    rec_res, dt_boxes, is_flip, is_rotated_90, time_cost = self.ocr_sys.predict(frame_crop, obj_type=metadata['type'])
                if cv_note:
                    self.ocr_sys.prof["path"].insert(0, cv_note)
                t1 = time.perf_counter()
                trace['end_mono'] = time.monotonic()
                trace['prof'] = dict(self.ocr_sys.prof, path=list(self.ocr_sys.prof["path"]))

                if is_rotated_90 and is_flip:
                    rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                elif is_rotated_90:
                    rot_code = cv2.ROTATE_90_CLOCKWISE
                elif is_flip:
                    rot_code = cv2.ROTATE_180
                else:
                    rot_code = None

                T_total = affine_to_homogeneous(M)

                if rot_code is not None:
                    T_total = _ROTATE_MATRIX_FN[rot_code](w0, h0) @ T_total
                    frame_crop = cv2.rotate(frame_crop, rot_code)

                if layout is not None:
                    # 版面交給解析器（OrderParser 的 CV 輔助），不用再算一次；
                    # 版面是對原始 crop 算的，換算成對轉正後的 frame_crop
                    from logic.yoshinoya.ticket_cv_layout import hand_off, rebase
                    applied = {None: 0, cv2.ROTATE_90_CLOCKWISE: 90, cv2.ROTATE_180: 180,
                               cv2.ROTATE_90_COUNTERCLOCKWISE: 270}[rot_code]
                    hand_off(frame_crop, rebase(layout, applied, w0, h0))

                is_rotated_90 = False
                is_flip = False

                t2 = time.perf_counter()
                T_total_inv = np.linalg.inv(T_total)
                metadata['M_inv_full'] = T_total_inv
                t3 = time.perf_counter()

                if self.result_callback:
                    try:
                        self.result_callback(frame_crop, rec_res, dt_boxes, is_flip, time_cost, metadata)
                    except Exception as e:
                        logger.error(f"Callback Error: {e}")
                t4 = time.perf_counter()

                # logger.info(
                #     f"predict={1000*(t1-t0):.2f}ms rotate+matrix={1000*(t2-t1):.2f}ms "
                #     f"inv={1000*(t3-t2):.2f}ms callback={1000*(t4-t3):.2f}ms rot_code={rot_code}"
                # )

                with self._lock:
                    self._busy = False
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"OCR Loop Error: {e}")
                with self._lock:
                    self._busy = False 


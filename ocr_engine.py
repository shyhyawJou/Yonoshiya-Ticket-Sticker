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

# 1. 取得當下時間，並格式化成字串 (例如: 20260702_105541)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
from AI import PaddleOCR_Det, PaddleOCR_Cls, PaddleOCR_Rec

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


class TextSystemDLA:
    def __init__(self, det_path, cls_path, rec_path, dict_path):

        # 1. 初始化 DLA
        self.det_model = PaddleOCR_Det(det_path)
        self.cls_model = PaddleOCR_Cls(cls_path)
        self.rec_model = PaddleOCR_Rec(rec_path)

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

        logger.info("模型初始化完成。")

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
        ori_im = img.copy()
        data, shape = self._det_preprocess(ori_im)
        if data is None:
            return np.array([]), 0

        # DLA 推論
        input_data = np.expand_dims(data, axis=0).astype(np.float16) 
        flat_output = self.det_model.run(input_data)
        pred = flat_output.reshape(1, self.det_image_shape[0], self.det_image_shape[1], 1)
        
        # 後處理
        post_result = self._det_postprocess(pred, [shape])
        dt_boxes = post_result[0]["points"]
        
        if len(dt_boxes) > 0:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et1 = time.time()
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
        return img_list, cls_res, flip_count, et2 - st2


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
            flat_output = self.rec_model.run(norm_img)
            # preds = flat_output.reshape(1, 40, 4401)
            preds = flat_output.reshape(1, 40, -1)

            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            rec_result = self._decode(preds_idx, preds_prob)
            rec_res[idx] = rec_result[0]

        et3 = time.time()
        return rec_res, et3 - st3

    def _check_boxes_need_rotate_90(self, dt_boxes, ratio_thresh=1.4, vertical_ratio=0.2):
        """
        檢查文字框是否被誤判為直式框，代表整張圖被轉成 90 或 270 度。
        用「比例」而非「絕對數量」判斷，這樣框數很少（例如貼紙只有 1 個框）也能適用。
        """
        if len(dt_boxes) == 0:
            return False

        vertical_count = 0
        for box in dt_boxes:
            box = np.array(box, dtype=np.float32)
            rect_width = np.linalg.norm(box[0] - box[1])
            rect_height = np.linalg.norm(box[0] - box[3])
            if rect_width > 0 and (rect_height / rect_width) >= ratio_thresh:
                vertical_count += 1

        return (vertical_count / len(dt_boxes)) >= vertical_ratio

    # ========================= #
    # Main Pipeline             #
    # ========================= #
    def predict(self, image_input):
        """主要預測流程"""
        st = time.time()
 
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
            ori_im = cv2.rotate(ori_im, cv2.ROTATE_90_CLOCKWISE)
            is_rotated_90 = True

            dt_boxes, elapse1 = self.run_det(ori_im)
            if len(dt_boxes) == 0:
                return [], [], False, is_rotated_90, time.time() - st

        # 2. 裁剪文字區域
        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)
        for box in dt_boxes:
            tmp_box = box.astype(np.float32)
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 3. 方向分類
        img_crop_list, cls_res, flip_count, elapse2 = self.run_cls(img_crop_list)
        is_flip = flip_count / len(img_crop_list) > 0.5 # 改用比例判斷
        # logger.info(f"cls_res num: {len(cls_res)}, elapsed: {elapse2:.3f}s")

        # 4. 若偵測到翻轉，旋轉影像 180 度後重新執行偵測與識別
        if is_flip:
            rotated_im = cv2.rotate(ori_im, cv2.ROTATE_180)

            dt_boxes, _ = self.run_det(rotated_im)

            if len(dt_boxes) == 0:
                return [], [], is_flip, is_rotated_90, time.time() - st

            img_crop_list = []
            dt_boxes = self.sorted_boxes(dt_boxes)
            for box in dt_boxes:
                tmp_box = box.astype(np.float32)
                img_crop = self.get_rotate_crop_image(rotated_im, tmp_box)
                img_crop_list.append(img_crop)

            img_crop_list, cls_res, _, _ = self.run_cls(img_crop_list)

        # 4. 文字識別
        rec_res, elapse3 = self.run_rec(img_crop_list)
        # logger.info(f"rec_res num: {len(rec_res)}, elapsed: {elapse3:.3f}s")

        et = time.time()

        return rec_res, dt_boxes, is_flip, is_rotated_90, et - st


class StandaloneRecDLA:
    def __init__(self, rec_path, dict_path, rec_image_shape=[3, 48, 320]):

        self.rec_image_shape = rec_image_shape
        self.count = 0
        self.rec_model = PaddleOCR_Rec(rec_path)

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

    def predict(self, frame_crop, dt_box, is_flip, nn_x=None):
        """
        傳入單張事先裁切好的圖片
        """
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

        return decode_out

    def predict_quantity(self, frame_crop: np.ndarray, cx: float, cy: float, h: float, w: float, is_flip: bool):
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

        return decode_out


class AsyncOCR:
    def __init__(self, det_path, cls_path, rec_path, dict_path, result_callback=None):
        """
        Args:
            result_callback: 處理完成後的回調函數，格式: func(rec_res, dt_boxes, time_cost, metadata)
        """
        self.ocr_sys = TextSystemDLA(det_path, cls_path, rec_path, dict_path)
        self.input_queue = queue.Queue(maxsize=1)
        
        # 狀態控制
        self.running = False
        self._busy = False            
        self._lock = threading.Lock() 
        
        self.result_callback = result_callback
        self.worker_thread = None

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
        try:
            self.input_queue.put((frame_crop.copy(), metadata), block=False)
            return True
        except queue.Full:
            with self._lock:
                self._busy = False
            return False

    def _worker_loop(self):
        """後台工作迴圈"""
        while self.running:
            try:
                input_data = self.input_queue.get(timeout=0.1)   
                frame_crop, metadata = input_data
                M_inv = metadata['M_inv']
                
                # --- 執行 OCR ---
                rec_res, dt_boxes, is_flip, is_rotated_90, time_cost = self.ocr_sys.predict(frame_crop)
                # 若被判定為 90/270 度誤轉，先把來源圖轉正 90 度
                if is_rotated_90:
                    frame_crop = cv2.rotate(frame_crop, cv2.ROTATE_90_CLOCKWISE)
                    is_rotated_90 = False

                # 若 OCR 內部偵測到翻轉，將來源 frame_crop 也一併旋轉
                if is_flip:
                    is_flip = False
                    frame_crop = cv2.rotate(frame_crop, cv2.ROTATE_180)
                         
                draw_det_bboxes(frame_crop, dt_boxes, save_path="det_debug.jpg")
                # 觸發 Callback，並將 metadata 原封不動送回
                if self.result_callback:
                    try:
                        self.result_callback(frame_crop, rec_res, dt_boxes, is_flip, time_cost, metadata)
                    except Exception as e:
                        logger.error(f"Callback Error: {e}")

                with self._lock:
                    self._busy = False
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"OCR Loop Error: {e}")
                with self._lock:
                    self._busy = False 


import cv2
import numpy as np
from AI import RTMDET
from schema import Detection  
from typing import List, Tuple

# ---------------------------
# main class
# ---------------------------

class Rotated_RTMDET:
    def __init__(self,
                 path: str,
                 classes: List[str],
                 conf_thresh=0.5,
                 iou_thresh=0.35,
                 partial_agnostic_ids: Tuple[int, ...] = (3, 4, 5, 6, 7, 8)) -> None:

        self.model = RTMDET(path)
        self.input_wh = [448, 448]
        self.mean = np.asarray([103.53, 116.28, 123.675], 'float32')
        self.std = np.asarray([57.375, 57.12, 58.395], 'float32')     
        self.classes = classes
        self.nc = len(self.classes)
        self.fix_color = True
        self.colors = []
        if self.fix_color:
            self.colors = [
                (225, 195, 129), # tray
                (145, 132, 224), # ticket
                (115, 184, 205)  # sticker
            ]
            if self.nc > len(self.colors):
                self.colors = [self.colors[i % len(self.colors)] for i in range(self.nc)]
        else:
            golden_ratio = 0.618033988749895
            hue_start = np.random.rand()
            for i in range(self.nc):
                hue = (hue_start + i * golden_ratio) % 1.0
                saturation = np.random.uniform(0.4, 0.5)
                value = np.random.uniform(0.8, 0.9)
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                self.colors.append((int(r * 255), int(g * 255), int(b * 255)))

        if isinstance(conf_thresh, dict):
            base_thresh = conf_thresh.get('default', 0.5) 
            self.conf_thresh = np.full(self.nc, base_thresh, dtype=np.float32)
            for i, cls_name in enumerate(self.classes):
                if cls_name in conf_thresh:
                    self.conf_thresh[i] = conf_thresh[cls_name]
        else:
            self.conf_thresh = float(conf_thresh)
            
        self.iou_thresh = float(iou_thresh)
        self.strides = [8, 16, 32]
        self.agnostic_nms = False
        self.partial_agnostic_ids = np.array(partial_agnostic_ids, dtype=np.float32) if partial_agnostic_ids else None
        self.grids = self._make_grids()
        self._warmup()

    def _warmup(self):
        x = np.random.randn(1, self.input_wh[1], self.input_wh[0], 3).astype('float32')
        for _ in range(2):
            self.model.run(x)

    # --------
    # preprocess
    # --------
    def _preprocess(self, img):
        dst_w, dst_h = self.input_wh
        h, w = img.shape[:2]
        scale_w, scale_h = dst_w / w, dst_h / h
        scale = min(scale_w, scale_h)
        new_h, new_w = int(h * scale), int(w * scale)

        if scale_w < scale_h:
            dx, dy = [0, (dst_h - new_h) // 2]  # x, y
        else:
            dx, dy = [(dst_w - new_w) // 2, 0]  # x, y
        
        img = cv2.resize(img, (new_w, new_h))
        x = np.full((dst_h, dst_w, 3), 114, 'uint8')
        x[dy : dy + new_h, dx : dx + new_w] = img
        x = (x - self.mean) / self.std
        x = x[None].astype('float32')
        return x

    # --------
    # postprocess
    # --------
    def _postprocess(self, y, origin_shape):
        y = self._non_max_suppression(y,
                                      self.conf_thresh,
                                      self.iou_thresh,
                                      agnostic=self.agnostic_nms,
                                      max_det=300,
                                      nc=self.nc,
                                      partial_agnostic_ids=self.partial_agnostic_ids)[0]
        rboxes = np.concatenate([y[:, :4], y[:, -1:]], axis=-1)
        rboxes[:, :4] = self._scale_boxes(self.input_wh[::-1], rboxes[:, :4], origin_shape, xywh=True)
        obb = np.concatenate([rboxes, y[:, 4:6]], axis=-1)  # xywhr, conf, cls
        xywhr = obb[:, :5]
        boxes = self.xywhr2xyxyxyxy(xywhr).astype('float32')
        confs = obb[:, 5].astype('float32')
        cls_ids = obb[:, 6].astype('int32')

        return xywhr, boxes, confs, cls_ids


    # --------
    # plot
    # --------
    def _plot(self, boxes, scores, cls_ids, img_bgr, show_box):
        out = img_bgr.copy()
        order = np.argsort(scores)
        h, w = img_bgr.shape[:2]

        for i in order:
            pts = np.array(boxes[i], dtype=np.int32)
            sc = float(scores[i])
            cid = int(cls_ids[i])

            if show_box:
                color = self.colors[cid % self.nc]
                cv2.polylines(out, [pts.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=5)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 5
                name = self.classes[cid] if 0 <= cid < self.nc else str(cid)
                label = f"{name} {sc:.2f}"

                anchor_x = int(np.min(pts[:, 0]))
                anchor_y = int(np.min(pts[:, 1]))
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                rect_top_left = (anchor_x, anchor_y - text_height - baseline)
                rect_bottom_right = (anchor_x + text_width, anchor_y)
                rect_top_left = (rect_top_left[0], max(0, rect_top_left[1]))
                text_origin = (anchor_x, anchor_y - baseline)

                # 繪製文字背景框
                cv2.rectangle(out, rect_top_left, rect_bottom_right, color, -1)            
                cv2.putText(out, label, text_origin, font, font_scale,
                            (255, 255, 255), font_thickness, cv2.LINE_AA)

        return out


    # --------
    # api
    # --------
    def detect(self, img, show_box: bool):
        x = self._preprocess(img)[0]
        boxes = self.model.run(x).reshape(1, -1, self.grids.shape[0])
        scores, boxes, angles = np.split(boxes, [boxes.shape[1] - 5, boxes.shape[1] - 1], 1)
        boxes = np.concatenate((boxes[0].T, angles[0].T), 1)
        boxes = self._distance2obb(self.grids, boxes, angle_version='le90').T[None]
        y = np.concatenate((boxes[:, :4], scores, angles), 1)
        xywhr, boxes, scores, cls_ids = self._postprocess(y, img.shape)

        out = self._plot(boxes, scores, cls_ids, img, show_box)

        detections: List[Detection] = []
        for r, b, s, c in zip(xywhr, boxes, scores, cls_ids):          
            cls_name = self.classes[c] if 0 <= c < len(self.classes) else f"id{c}"
            detections.append(Detection(xyxy=tuple(b.flatten()),
                                        xywhr=tuple(r.flatten()),
                                        cls_id=c,
                                        cls_name=cls_name,
                                        conf=s))
        return out, detections

    # ---------------------------
    # utils: letterbox & geometry
    # ---------------------------
    def _make_grids(self):
        w, h = self.input_wh
        grids = []
        for s in self.strides:
            x_ticks = np.float32(np.linspace(0, w, w // s, False))
            y_ticks = np.float32(np.linspace(0, h, h // s, False))
            x, y = np.meshgrid(x_ticks, y_ticks)
            grids.append(np.c_[x.ravel(), y.ravel()])
        grids = np.concatenate(grids, axis=None).reshape(-1, 2)
        return grids

    def _distance2obb(self, points, distance, angle_version='le90'):
        assert points.shape[0] == distance.shape[0]
        assert points.shape[-1] == 2
        assert distance.shape[-1] == 5

        distance, angle = np.split(distance, [4], axis=-1)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)

        rot_matrix = np.concatenate([cos_angle, -sin_angle, sin_angle, cos_angle], axis=-1)
        rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

        wh = distance[..., :2] + distance[..., 2:]
        offset_t = (distance[..., 2:] - distance[..., :2]) / 2
        offset = np.matmul(rot_matrix, offset_t[..., None]).squeeze(-1)
        ctr = points[..., :2] + offset

        angle_regular = self._norm_angle(angle, angle_version)
        return np.concatenate([ctr, wh, angle_regular], axis=-1)

    def _norm_angle(self, angle, angle_range):
        if angle_range == 'oc':
            return angle
        elif angle_range == 'le135':
            return (angle + np.pi / 4) % np.pi - np.pi / 4
        elif angle_range == 'le90':
            angle = (angle + np.pi / 2) % np.pi - np.pi / 2
            angle = np.where(angle < 0, angle + 2 * np.pi, angle)
            return angle
        elif angle_range == 'r360':
            return (angle + np.pi) % (2 * np.pi) - np.pi
        else:
            print('Not yet implemented.')
 
    def _scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        if ratio_pad is None: 
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            ) 
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  
            boxes[..., 1] -= pad[1]  
            if not xywh:
                boxes[..., 2] -= pad[0]  
                boxes[..., 3] -= pad[1]  
        boxes[..., :4] /= gain
        return self._clip_boxes(boxes, img0_shape)

    def _clip_boxes(self, boxes, shape):
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  
        return boxes

    def _non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=True,
        max_det=300,
        nc=0,
        max_nms=30000,
        max_wh=7680,
        partial_agnostic_ids=None,
    ):
        if isinstance(prediction, (list, tuple)):  
            prediction = prediction[0]  

        bs = prediction.shape[0]  
        nc = nc or (prediction.shape[1] - 4)  
        nm = prediction.shape[1] - nc - 4 
        mi = 4 + nc 

        if isinstance(conf_thres, np.ndarray):
            m_per_class = prediction[:, 4:mi] > conf_thres.reshape(nc, 1)
            xc = np.any(m_per_class, axis=1) 
        else:
            xc = prediction[:, 4:mi].max(1) > conf_thres 

        prediction = prediction.transpose(0, 2, 1)  

        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction): 
            x = x[xc[xi]]  
  
            if not x.shape[0]:
                continue

            box, clas, mask_data = np.split(x, (4, 4 + nc), 1)
            conf, j = clas.max(1, keepdims=True), clas.argmax(1, keepdims=True)

            x = np.concatenate((box, conf, j.astype('float32'), mask_data), 1)
            if isinstance(conf_thres, np.ndarray):
                j_flat = j.astype(int).ravel() 
                thresholds_for_best_class = conf_thres[j_flat] 
                filter_mask = conf.ravel() > thresholds_for_best_class 
            else:
                filter_mask = conf.ravel() > conf_thres 

            x = x[filter_mask]
            n = x.shape[0]  
            
            if not n:  
                continue
            
            if n > max_nms:  
                x = x[np.argsort(-x[:, 4])[:max_nms]]  

            # --- 決定用來做「位移」的 class id（partial-agnostic 的核心）---
            group_ids = x[:, 5].copy()
            if partial_agnostic_ids is not None:
                mask = np.isin(group_ids, partial_agnostic_ids)
                if mask.any():
                    group_ids[mask] = partial_agnostic_ids[0]  # 統一成同一個代表 id

            c = group_ids[:, None] * (0 if agnostic else max_wh)
            scores = x[:, 4]
            boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)
            i = self._nms_rotated(boxes, scores, iou_thres)
            i = i[:max_det]
            output[xi] = x[i]

        return output

    def _nms_rotated(self, boxes, scores, threshold=0.45):
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int8)
        sorted_idx = np.argsort(-scores)
        boxes = boxes[sorted_idx]
        ious = np.triu(self._batch_probiou(boxes, boxes), 1)
        pick = np.stack(np.nonzero(ious.max(0) < threshold), 1).squeeze(-1)
        return sorted_idx[pick]

    def _batch_probiou(self, obb1, obb2, eps=1e-7):
        x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
        x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, -1))
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2))

        t1 = (((a1 + a2) * np.power(y1 - y2, 2) + (b1 + b2) * np.power(x1 - x2, 2)) / ((a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2) + eps)) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2) + eps)) * 0.5
        t3 = np.log(((a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2)) / (4 * np.sqrt((a1 * b1 - np.power(c1, 2)).clip(0) * (a2 * b2 - np.power(c2, 2)).clip(0)) + eps) + eps) * 0.5
        bd = (t1 + t2 + t3).clip(eps, 100.0)
        hd = np.sqrt(1.0 - np.exp(-bd) + eps)
        return 1 - hd

    def _get_covariance_matrix(self, boxes):
        gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), -1)
        a, b, c = np.split(gbbs, 3, axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = np.power(cos, 2)
        sin2 = np.power(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    def xywhr2xyxyxyxy(self, xywhr):
        cos, sin, cat, stack = (np.cos, np.sin, np.concatenate, np.stack)
        ctr = xywhr[..., :2]
        w, h, angle = (xywhr[..., i : i + 1] for i in range(2, 5))
        cos_value, sin_value = cos(angle), sin(angle)
        vec1 = [w / 2 * cos_value, w / 2 * sin_value]
        vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
        vec1 = cat(vec1, -1)
        vec2 = cat(vec2, -1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return stack([pt1, pt2, pt3, pt4], -2)

    # ---------------------------
    # utils: crop
    # ---------------------------
    def crop_by_angle(self, img, cx, cy, w, h, angle_rad):
        """
        裁切旋轉影像
        """
        angle_deg = np.degrees(angle_rad)
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        M[0, 2] += (w / 2) - cx
        M[1, 2] += (h / 2) - cy
        warped_img = cv2.warpAffine(img, M, (int(w), int(h)))
        M_inv = cv2.invertAffineTransform(M)
        return warped_img, M_inv

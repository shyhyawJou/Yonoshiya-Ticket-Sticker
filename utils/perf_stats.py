"""
perf_stats.py
=============
效能計時 log。三種輸出，全部固定開著（成本是每幀幾次 perf_counter，微秒級）：

  [OCR#n ▶ 類型]   OCR 線程開始處理一次讀取（ocr_engine.AsyncOCR._worker_loop）
  [OCR#n ■ 類型]   主迴圈套用完那次結果之後（app.py），附整段計時。
                   兩行之間就是這次讀取的所有 log（翻轉投票、OCR res、比對 / 解析…）
  [效能]           主迴圈每 interval_sec 秒彙總一行：每幀各段的平均、p95、最大

另外 LOOP_PROFILER 有一份從開機起累計的計數，LogicEngine 在訂單開始時取快照、
結束時相減，就是 [訂單結束] 那行的「期間效能」。

時間戳一律 time.monotonic()（跨執行緒比較用；跳電校時不影響，見 app.py 主迴圈的說明）。
"""

import threading
import time
from typing import Dict, List, Optional

from loguru import logger


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ------------------------------------------------------------------
# OCR 每次讀取
# ------------------------------------------------------------------
def ocr_begin_line(metadata: dict) -> str:
    tr = metadata.get("trace") or {}
    queued = ""
    if tr.get("req_mono") is not None and tr.get("start_mono") is not None:
        queued = f" 排隊 {1000 * (tr['start_mono'] - tr['req_mono']):.0f}ms"
    return (
        f"[OCR#{tr.get('job', '?')} ▶ {metadata.get('type')}] tray={metadata.get('tray_id')} "
        f"第 {tr.get('read_no', '?')} 次讀取 crop={tr.get('crop', '?')}{queued}"
    )


def ocr_done_line(metadata: dict, drained_mono: float, apply_ms: float,
                  qty_calls: int = 0, qty_ms: float = 0.0) -> str:
    """
    Args:
        drained_mono: 主迴圈從 queue 取到這份結果的時間
        apply_ms:     logic.apply_ocr_result 花的時間（傳票解析 / 貼紙比對，含數量欄 rec）
        qty_calls, qty_ms: 這次 apply 裡數量欄 rec（StandaloneRecDLA）跑了幾次、共幾 ms
    """
    tr = metadata.get("trace") or {}
    p = tr.get("prof") or {}
    path = "+".join(p.get("path") or []) or "一般"
    parts = [f"[OCR#{tr.get('job', '?')} ■ {metadata.get('type')}] 路徑={path}"]

    det_ms, cls_ms, rec_ms = p.get("det_ms", 0.0), p.get("cls_ms", 0.0), p.get("rec_ms", 0.0)
    seg = []
    if p.get("det_n"):
        pre, inf = p.get("det_pre_ms", 0.0), p.get("det_infer_ms", 0.0)
        seg.append(
            f"det {p['det_n']}次 {det_ms:.1f}ms"
            f"（前處理 {pre:.1f}／DLA {inf:.1f}／後處理 {max(0.0, det_ms - pre - inf):.1f}）"
        )
    if p.get("cls_n"):
        seg.append(f"cls {p['cls_n']}框 {cls_ms:.1f}ms")
    boxes = p.get("rec_boxes", 0)
    if boxes:
        seg.append(
            f"rec {boxes}框/{p.get('rec_rounds', 0)}輪 {rec_ms:.1f}ms"
            f"（每框 {rec_ms / boxes:.1f}ms，其中 DLA {p.get('rec_infer_ms', 0.0) / boxes:.1f}ms）"
        )
    elif p.get("rec_rounds"):
        seg.append("rec 0框")
    if seg:
        parts.append(" | ".join(seg))

    start, end = tr.get("start_mono"), tr.get("end_mono")
    if start is not None and end is not None:
        thread_ms = 1000 * (end - start)
        other_ms = max(0.0, thread_ms - det_ms - cls_ms - rec_ms)
        queued = ""
        if tr.get("req_mono") is not None:
            queued = f"，排隊 {1000 * (start - tr['req_mono']):.0f}ms"
        qty = f"（數量欄 rec {qty_calls}次 {qty_ms:.1f}ms）" if qty_calls else ""
        parts.append(
            f"OCR 線程 {thread_ms:.0f}ms（裁切/前後處理 {other_ms:.0f}ms{queued}）"
            f" + 等主迴圈 {1000 * (drained_mono - end):.0f}ms"
            f" + 解析/比對 {apply_ms:.1f}ms{qty}"
        )

    # 這一輪：起點（出現 / 傳票確認 / 上一輪結果，見 TrayStateMachine._task_trace）
    # -> 送 OCR（等穩定）-> 推論完、結果套用（得到結果）
    start = tr.get("cycle_start_mono")
    if start is not None and tr.get("req_mono") is not None:
        done = drained_mono + apply_ms / 1000.0
        resets = tr.get("still_resets", 0)
        moved = f"（期間移動重置 {resets} 次）" if resets else ""
        early = tr.get("waited_before_ready", 0.0)
        before = f"（傳票確認前已在桌上 {early:.1f}s）" if early else ""
        parts.append(
            f"本輪 {tr.get('cycle_origin', '出現')}→送OCR {tr['req_mono'] - start:.2f}s{moved}"
            f" → 得到結果 {done - start:.2f}s{before}"
        )

    return " | ".join(parts)


# ------------------------------------------------------------------
# 主迴圈每幀
# ------------------------------------------------------------------
class LoopProfiler:
    # (key, 顯示名稱)；順序就是 [效能] 那行的順序，也大致是主迴圈的執行順序
    STAGES = (
        ("camera", "等相機"), ("depth", "深度"), ("record", "錄影"), ("apply", "套用OCR"),
        ("mmr_pre", "前處理"), ("mmr_infer", "推論"), ("mmr_post", "後處理"), ("mmr_plot", "畫框"),
        ("logic", "logic"), ("dispatch", "送OCR"), ("overlay", "overlay/串流"),
    )
    MMR_KEYS = ("mmr_pre", "mmr_infer", "mmr_post", "mmr_plot")

    def __init__(self, interval_sec: float = 900.0):
        self.interval_sec = interval_sec
        self._lock = threading.Lock()
        self._reset_window(time.monotonic())
        # 從開機起累計，訂單開始 / 結束時取快照相減
        self._cum = {"frames": 0, "total_ms": 0.0, "mmr_ms": 0.0, "mmr_infer_ms": 0.0,
                     "ocr_jobs": 0, "ocr_ms": 0.0}

    def _reset_window(self, now: float) -> None:
        self._win_start = now
        self._win: Dict[str, List[float]] = {k: [] for k, _ in self.STAGES}
        self._win_total: List[float] = []
        self._win_ocr: List[float] = []

    def add_frame(self, stages: Dict[str, float], total_ms: float) -> None:
        with self._lock:
            for k, _ in self.STAGES:
                self._win[k].append(stages.get(k, 0.0))
            self._win_total.append(total_ms)
            self._cum["frames"] += 1
            self._cum["total_ms"] += total_ms
            self._cum["mmr_ms"] += sum(stages.get(k, 0.0) for k in self.MMR_KEYS)
            self._cum["mmr_infer_ms"] += stages.get("mmr_infer", 0.0)

    def add_ocr(self, thread_ms: float) -> None:
        with self._lock:
            self._win_ocr.append(thread_ms)
            self._cum["ocr_jobs"] += 1
            self._cum["ocr_ms"] += thread_ms

    def maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._win_start < self.interval_sec:
            return
        with self._lock:
            span = now - self._win_start
            n = len(self._win_total)
            if n:
                w, a = self._win, {k: _avg(v) for k, v in self._win.items()}
                mmr = sum(a[k] for k in self.MMR_KEYS)
                ocr = (f"OCR {len(self._win_ocr)} 次 平均 {_avg(self._win_ocr):.0f}ms"
                       if self._win_ocr else "OCR 0 次")
                logger.info(
                    f"[效能] 近 {span:.0f}s {n} 幀 {n / span:.1f}fps｜每幀 {_avg(self._win_total):.1f}ms"
                    f"（p95 {_pct(self._win_total, 0.95):.1f}／最大 {max(self._win_total):.1f}）"
                    f" = 等相機 {a['camera']:.1f} + 深度 {a['depth']:.1f} + 錄影 {a['record']:.1f}"
                    f" + 套用OCR {a['apply']:.1f}"
                    f" + mmr {mmr:.1f}（前處理 {a['mmr_pre']:.1f}／推論 {a['mmr_infer']:.1f}"
                    f" p95 {_pct(w['mmr_infer'], 0.95):.1f}／後處理 {a['mmr_post']:.1f}"
                    f"／畫框 {a['mmr_plot']:.1f}）"
                    f" + logic {a['logic']:.1f} + 送OCR {a['dispatch']:.1f}"
                    f" + overlay/串流 {a['overlay']:.1f}｜{ocr}"
                )
            self._reset_window(now)

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self._cum)
        snap["mono"] = time.monotonic()
        return snap

    def since(self, snap: Optional[dict]) -> str:
        """從快照到現在的效能摘要，給 [訂單結束] 那行用"""
        if not snap:
            return ""
        now = self.snapshot()
        frames = now["frames"] - snap["frames"]
        span = now["mono"] - snap["mono"]
        if frames <= 0 or span <= 0:
            return ""
        jobs = now["ocr_jobs"] - snap["ocr_jobs"]
        ocr = f"，OCR {jobs} 次 平均 {(now['ocr_ms'] - snap['ocr_ms']) / jobs:.0f}ms" if jobs else "，OCR 0 次"
        return (
            f"期間 {frames} 幀 {frames / span:.1f}fps、每幀 {(now['total_ms'] - snap['total_ms']) / frames:.1f}ms"
            f"、mmr {(now['mmr_ms'] - snap['mmr_ms']) / frames:.1f}ms"
            f"（推論 {(now['mmr_infer_ms'] - snap['mmr_infer_ms']) / frames:.1f}）{ocr}"
        )


# 整個程式一份：主迴圈寫入，LogicEngine 讀快照（切換模式重建 LogicEngine 也不受影響）
LOOP_PROFILER = LoopProfiler()

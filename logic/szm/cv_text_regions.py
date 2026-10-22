"""Model-free region extraction. Returns image, quads, original-to-image transform."""
from pathlib import Path
import importlib.util
import numpy as np


def _load(name):
    path = Path(__file__).resolve().parents[3] / "ONNX_on_nuc" / "suzumo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_cv_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quarter_matrix(cw, w, h):
    return np.array({
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, h-1], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, w-1], [0, -1, h-1], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, w-1], [0, 0, 1]],
    }[cw], dtype=float)


def extract_regions(image, backend):
    if backend == "cv_table":
        grid = _load("table_grid")
        cw, table = grid.detect_orientation(image)
        if table is None:
            return image, np.empty((0, 4, 2), np.float32), np.eye(3)
        warped = table["warped"]
        cells = grid.table_cells(table)
        rects = [(cells["order_no_band"], True), (cells["header_qty"], False)]
        for row in cells["rows"]:
            rects.extend([(row["name"], False), (row["qty"], False)])
        boxes = []
        for rect, lead in rects:
            box = grid.cell_ink_box(warped, rect, drop_short_lead=lead)
            if box:
                x0, y0, x1, y1 = box
                boxes.append([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])
        transform = np.vstack([table["matrix"], [0,0,1]]) @ quarter_matrix(cw, image.shape[1], image.shape[0])
        return warped, np.asarray(boxes, np.float32).reshape(-1,4,2), transform
    if backend == "cv_ink":
        ink = _load("sticker_box")
        mask = ink.word_mask(image)
        boxes = [] if mask is None else [ink.rect_corners(mask)[0]]
        return image, np.asarray(boxes, np.float32).reshape(-1,4,2), np.eye(3)
    raise ValueError(f"Not a CV backend: {backend}")

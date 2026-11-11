#!/usr/bin/env python3
"""
merge_config.py <old_config.yaml> <new_config.yaml>

把 old_config.yaml 裡的:
    camera_params.hik.ReverseX.value
    camera_params.hik.ReverseY.value
    preset_roi.ticket_roi.points
複製到 new_config.yaml 的相同路徑,其餘內容維持 new_config.yaml 原樣,
並盡量保留原有格式(縮排、註解、引號風格)。

優先使用 ruamel.yaml (round-trip 模式,保留格式);
如果環境沒裝 ruamel.yaml,則自動退回 PyYAML (仍能正確合併,但輸出格式
可能會重新排版,例如引號風格、行寬等,不影響程式讀取,但檔案看起來會不同)。
"""
import sys


def get_path(d, path):
    cur = d
    for key in path:
        cur = cur[key]
    return cur


def set_path(d, path, value):
    cur = d
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


PATHS = [
    ("camera_params", "hik", "ReverseX", "value"),
    ("camera_params", "hik", "ReverseY", "value"),
    ("preset_roi", "ticket_roi", "points"),
]


def merge_with_ruamel(old_path, new_path):
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # 避免長陣列被自動換行拆散

    with open(old_path, "r", encoding="utf-8") as f:
        old_cfg = yaml.load(f)
    with open(new_path, "r", encoding="utf-8") as f:
        new_cfg = yaml.load(f)

    for path in PATHS:
        value = get_path(old_cfg, path)
        set_path(new_cfg, path, value)

    with open(new_path, "w", encoding="utf-8") as f:
        yaml.dump(new_cfg, f)


def merge_with_pyyaml(old_path, new_path):
    import yaml

    with open(old_path, "r", encoding="utf-8") as f:
        old_cfg = yaml.safe_load(f)
    with open(new_path, "r", encoding="utf-8") as f:
        new_cfg = yaml.safe_load(f)

    for path in PATHS:
        value = get_path(old_cfg, path)
        set_path(new_cfg, path, value)

    with open(new_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    if len(sys.argv) != 3:
        print("用法: merge_config.py <old_config.yaml> <new_config.yaml>", file=sys.stderr)
        sys.exit(2)

    old_path, new_path = sys.argv[1], sys.argv[2]

    try:
        merge_with_ruamel(old_path, new_path)
        print("已用 ruamel.yaml 合併完成 (保留原始格式)。")
    except ImportError:
        try:
            merge_with_pyyaml(old_path, new_path)
            print("未偵測到 ruamel.yaml,已改用 PyYAML 合併完成 (輸出格式可能與原檔略有不同,但內容正確)。")
        except ImportError:
            print(
                "ERROR: 系統沒有安裝 ruamel.yaml 或 PyYAML,無法安全合併 YAML。"
                " 請先執行 `pip3 install ruamel.yaml` 或 `pip3 install pyyaml`。",
                file=sys.stderr,
            )
            sys.exit(1)
    except KeyError as e:
        print(f"ERROR: 舊版 config 裡缺少預期的欄位: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

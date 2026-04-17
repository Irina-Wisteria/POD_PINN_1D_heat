"""比较 POD-PINN 与普通 PINN 训练结果的汇总脚本。

这个脚本读取两个训练目录下的 `summary.json`，在终端打印一张紧凑的对比表，
并额外生成一份合并后的 `comparison_summary.json` 方便后续分析或绘图脚本直接复用。
"""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    """读取单个模型的 summary 文件。

    如果文件不存在，直接抛出异常，让调用者明确知道前置训练步骤还没有完成。
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_seconds(value: float) -> str:
    """将训练耗时格式化为保留两位小数的秒数字符串。"""
    return f"{value:.2f}s"


def format_float(value: float) -> str:
    """将误差类指标格式化为科学计数法，便于对齐比较。"""
    return f"{value:.6e}"


def main() -> None:
    """读取两个模型的汇总信息，打印表格并导出合并后的 JSON。"""
    root = Path(__file__).resolve().parent
    pod_summary = load_summary(root / "outputs" / "summary.json")
    plain_summary = load_summary(root / "outputs_plain_pinn" / "summary.json")

    # rows 的第一行是表头，后面两行分别对应 POD-PINN 与普通 PINN 的关键指标。
    rows = [
        (
            "Model",
            "Params",
            "Train Time",
            "Rel L2",
            "Final Err",
            "Max Abs",
        ),
        (
            "POD-PINN",
            str(pod_summary["parameter_count"]),
            format_seconds(pod_summary["training_seconds"]),
            format_float(pod_summary["metrics"]["relative_l2_error"]),
            format_float(pod_summary["metrics"]["final_time_relative_error"]),
            format_float(pod_summary["metrics"]["max_abs_error"]),
        ),
        (
            "Plain PINN",
            str(plain_summary["parameter_count"]),
            format_seconds(plain_summary["training_seconds"]),
            format_float(plain_summary["metrics"]["relative_l2_error"]),
            format_float(plain_summary["metrics"]["final_time_relative_error"]),
            format_float(plain_summary["metrics"]["max_abs_error"]),
        ),
    ]

    # 逐列计算最大宽度，保证终端中打印出来的表格能够整齐对齐。
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row_index, row in enumerate(rows):
        line = " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        print(line)
        if row_index == 0:
            # 在表头下方打印分隔线，增强可读性。
            print("-+-".join("-" * width for width in widths))

    # 同时保留一份机器可读的总表，便于后续脚本统一读取比较结果。
    comparison = {
        "pod_pinn": pod_summary,
        "plain_pinn": plain_summary,
    }
    with (root / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)


if __name__ == "__main__":
    main()

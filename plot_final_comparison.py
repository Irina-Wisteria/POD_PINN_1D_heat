"""绘制 POD-PINN 与普通 PINN 在最终时刻上的解和误差对比图。

脚本从两个训练目录中读取 `field_data.npz`，校验两者使用的是同一套网格与真解，
然后在同一张图中对比最终时刻的解曲线以及对应误差曲线。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数，允许用户自定义输入数据和输出图片路径。"""
    parser = argparse.ArgumentParser(
        description="Plot final solution and final error curves for POD-PINN and plain PINN."
    )
    parser.add_argument(
        "--pod-data",
        type=str,
        default="outputs/field_data.npz",
        help="Path to POD-PINN saved field data.",
    )
    parser.add_argument(
        "--plain-data",
        type=str,
        default="outputs_plain_pinn/field_data.npz",
        help="Path to plain PINN saved field data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_solution_error_comparison.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取 `.npz` 文件并转换成普通字典，便于后续按键访问。"""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}. Run the training scripts first to generate field_data.npz."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    """加载两组预测结果，做一致性校验，并绘制最终时刻对比图。"""
    args = parse_args()
    root = Path(__file__).resolve().parent
    pod_path = (root / args.pod_data).resolve()
    plain_path = (root / args.plain_data).resolve()
    output_path = (root / args.output).resolve()

    pod = load_npz(pod_path)
    plain = load_npz(plain_path)

    x = pod["x"]
    # 两个模型必须使用相同的空间网格，否则直接叠加比较是没有意义的。
    if not np.allclose(x, plain["x"]):
        raise ValueError("The x grids are not identical between POD-PINN and plain PINN outputs.")

    exact_final = pod["exact_final"]
    # 两份数据中的真解也必须一致，否则说明输入数据来源不同，比较结果不可靠。
    if not np.allclose(exact_final, plain["exact_final"]):
        raise ValueError("The exact final profiles do not match between the two outputs.")

    pod_final = pod["prediction_final"]
    plain_final = plain["prediction_final"]
    pod_error = pod["final_error"]
    plain_error = plain["final_error"]

    pod_final_l2 = np.linalg.norm(pod_error) / np.linalg.norm(exact_final)
    plain_final_l2 = np.linalg.norm(plain_error) / np.linalg.norm(exact_final)
    pod_max = np.max(np.abs(pod_error))
    plain_max = np.max(np.abs(plain_error))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # 上半部分展示最终时刻的解曲线，直接看模型是否贴近真解。
    axes[0].plot(x, exact_final, color="black", linewidth=2.0, label="Exact")
    axes[0].plot(x, pod_final, linestyle="--", linewidth=2.0, label="POD-PINN")
    axes[0].plot(x, plain_final, linestyle=":", linewidth=2.2, label="Plain PINN")
    axes[0].set_ylabel("u(x, t_max)")
    axes[0].set_title("Final-time solution comparison")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # 下半部分展示最终时刻误差，图例里额外给出相对 L2 和最大绝对误差，便于快速比较。
    axes[1].plot(
        x,
        pod_error,
        linestyle="--",
        linewidth=2.0,
        label=f"POD-PINN error | rel={pod_final_l2:.3e}, max={pod_max:.3e}",
    )
    axes[1].plot(
        x,
        plain_error,
        linestyle=":",
        linewidth=2.2,
        label=f"Plain PINN error | rel={plain_final_l2:.3e}, max={plain_max:.3e}",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Prediction - Exact")
    axes[1].set_title("Final-time error comparison")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    # 输出目录不存在时自动创建，保证脚本可以直接运行。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved comparison figure to: {output_path}")
    print(f"POD-PINN final relative error: {pod_final_l2:.6e}")
    print(f"Plain PINN final relative error: {plain_final_l2:.6e}")


if __name__ == "__main__":
    main()

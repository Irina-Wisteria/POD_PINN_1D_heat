"""绘制 POD-PINN 与普通 PINN 的二维 `(x, t)` 场对比图。

图像采用 2 行 2 列布局：
1. 第一行展示两个模型各自的预测场；
2. 第二行展示两个模型相对于真解的误差场。

为了让横向比较更直观，同一行中的两个子图使用完全一致的颜色范围：
1. 预测场共用一套 `vmin / vmax`；
2. 误差场共用一套关于 0 对称的颜色范围。

这样可以避免因为每张子图单独缩放色标而造成的“看起来都差不多”但实际误差不同的问题。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# 使用非交互式后端，保证脚本在命令行和服务器环境中都能正常保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数，允许自定义输入数据和输出图片路径。"""
    parser = argparse.ArgumentParser(
        description="Plot side-by-side (x, t) prediction and error fields for POD-PINN and plain PINN."
    )
    parser.add_argument(
        "--pod-data",
        type=str,
        default="outputs/field_data.npz",
        help="POD-PINN 保存的场数据文件路径。",
    )
    parser.add_argument(
        "--plain-data",
        type=str,
        default="outputs_plain_pinn/field_data.npz",
        help="普通 PINN 保存的场数据文件路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="space_time_model_comparison.png",
        help="输出图片路径。",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取 `.npz` 文件并转换成普通字典。"""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}. Run the training scripts first to generate field_data.npz."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def validate_grids(
    pod: dict[str, np.ndarray],
    plain: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """校验两个模型输出是否建立在同一套空间网格、时间网格和真解之上。"""
    x = pod["x"]
    t = pod["t"]
    exact = pod["exact"]

    if not np.allclose(x, plain["x"]):
        raise ValueError("The x grids are not identical between POD-PINN and plain PINN outputs.")
    if not np.allclose(t, plain["t"]):
        raise ValueError("The t grids are not identical between POD-PINN and plain PINN outputs.")
    if not np.allclose(exact, plain["exact"]):
        raise ValueError("The exact fields do not match between the two outputs.")

    return x, t, exact


def compute_shared_limits(
    pod_prediction: np.ndarray,
    plain_prediction: np.ndarray,
    pod_error: np.ndarray,
    plain_error: np.ndarray,
) -> tuple[float, float, float]:
    """计算两行子图共用的颜色范围。

    返回值依次为：
    1. 预测场统一色标下界
    2. 预测场统一色标上界
    3. 误差场统一色标的绝对值上界
    """
    prediction_min = float(min(np.min(pod_prediction), np.min(plain_prediction)))
    prediction_max = float(max(np.max(pod_prediction), np.max(plain_prediction)))
    error_abs_max = float(max(np.max(np.abs(pod_error)), np.max(np.abs(plain_error))))
    return prediction_min, prediction_max, error_abs_max


def main() -> None:
    """加载两个模型的二维场数据，并绘制统一色标的时空对比图。"""
    args = parse_args()
    root = Path(__file__).resolve().parent
    pod_path = (root / args.pod_data).resolve()
    plain_path = (root / args.plain_data).resolve()
    output_path = (root / args.output).resolve()

    pod = load_npz(pod_path)
    plain = load_npz(plain_path)
    x, t, exact = validate_grids(pod, plain)

    pod_prediction = pod["prediction"]
    plain_prediction = plain["prediction"]
    pod_error = pod_prediction - exact
    plain_error = plain_prediction - exact

    pred_vmin, pred_vmax, err_abs_max = compute_shared_limits(
        pod_prediction,
        plain_prediction,
        pod_error,
        plain_error,
    )

    # imshow 的图像数组按 `(纵轴, 横轴)` 解释，因此这里转置为 `(t, x)` 再绘图。
    extent = [x.min(), x.max(), t.max(), t.min()]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    prediction_images = [
        (pod_prediction.T, "POD-PINN prediction"),
        (plain_prediction.T, "Plain PINN prediction"),
    ]
    error_images = [
        (pod_error.T, "POD-PINN error"),
        (plain_error.T, "Plain PINN error"),
    ]

    top_mappables = []
    for axis, (image, title) in zip(axes[0], prediction_images):
        # 第一行固定使用同一组 `vmin / vmax`，确保两个预测场的颜色含义一致。
        im = axis.imshow(
            image,
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=pred_vmin,
            vmax=pred_vmax,
        )
        axis.set_title(title)
        axis.set_ylabel("t")
        top_mappables.append(im)

    bottom_mappables = []
    for axis, (image, title) in zip(axes[1], error_images):
        # 第二行误差图围绕 0 对称设置颜色范围，正误差和负误差的视觉权重一致。
        im = axis.imshow(
            image,
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-err_abs_max,
            vmax=err_abs_max,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        bottom_mappables.append(im)

    # 为每一行创建共享颜色条，进一步强调“同一行色标一致”的对比关系。
    pred_cbar = fig.colorbar(top_mappables[0], ax=axes[0, :], shrink=0.92, pad=0.03)
    pred_cbar.set_label("u(x, t)")

    err_cbar = fig.colorbar(bottom_mappables[0], ax=axes[1, :], shrink=0.92, pad=0.03)
    err_cbar.set_label("Prediction - Exact")

    fig.suptitle("Space-time field comparison", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved space-time comparison figure to: {output_path}")
    print(f"Prediction color scale: [{pred_vmin:.6e}, {pred_vmax:.6e}]")
    print(f"Error color scale: [{-err_abs_max:.6e}, {err_abs_max:.6e}]")


if __name__ == "__main__":
    main()

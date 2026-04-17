"""POD-PINN 求解一维热方程的完整示例。

整体流程分为两步：
1. 先利用解析解快照做 POD 分解，提取主要空间模态，构建低维表示；
2. 再让 PINN 只学习时间相关的模态系数，从而降低学习难度和参数规模。

相较于普通 PINN 直接学习 `(x, t) -> u(x, t)`，这里把主要空间结构先固定下来，
网络只负责学习随时间变化的系数，因此更适合展示降阶建模与物理约束结合的思路。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

# 使用非交互式后端，避免在无显示环境中绘图失败。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# 对 PINN 类问题使用双精度通常更稳，尤其是在计算导数和误差时。
torch.set_default_dtype(torch.float64)


@dataclass
class ProblemConfig:
    """热方程的物理参数和解析解初值配置。"""

    nu: float = 0.01
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    mode_numbers: tuple[int, ...] = (1, 2, 4)
    mode_amplitudes: tuple[float, ...] = (1.0, 0.7, -0.5)


@dataclass
class TrainingConfig:
    """POD 和 PINN 训练阶段共同使用的超参数配置。"""

    n_x: int = 256
    n_snapshots: int = 160
    n_collocation: int = 160
    hidden_width: int = 64
    hidden_depth: int = 3
    epochs: int = 2500
    lr: float = 1.0e-3
    energy_threshold: float = 1.0
    max_modes: int = 6
    seed: int = 42
    log_every: int = 250
    device: str = "cpu"


class CoefficientNet(nn.Module):
    """学习 POD 模态系数随时间变化规律的全连接网络。"""

    def __init__(self, out_dim: int, width: int, depth: int) -> None:
        super().__init__()
        # 输入只是一维时间，输出是每个保留 POD 模态对应的时间系数。
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t_scaled: torch.Tensor) -> torch.Tensor:
        """根据缩放后的时间输入预测所有模态系数。"""
        return self.net(t_scaled)


def count_parameters(model: nn.Module) -> int:
    """统计模型中所有可训练参数的总数。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parse_args() -> TrainingConfig:
    """解析命令行参数，并组装成训练配置对象。"""
    parser = argparse.ArgumentParser(
        description="Complete POD-PINN example on the 1D heat equation."
    )
    parser.add_argument("--n-x", type=int, default=256, help="Number of spatial points.")
    parser.add_argument(
        "--n-snapshots", type=int, default=160, help="Number of snapshots for POD."
    )
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=160,
        help="Number of collocation times per optimization step.",
    )
    parser.add_argument(
        "--hidden-width", type=int, default=64, help="Hidden layer width."
    )
    parser.add_argument(
        "--hidden-depth", type=int, default=3, help="Number of hidden layers."
    )
    parser.add_argument("--epochs", type=int, default=2500, help="Adam iterations.")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate.")
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=1.0,
        help="POD cumulative energy threshold.",
    )
    parser.add_argument(
        "--max-modes", type=int, default=6, help="Maximum retained POD modes."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--log-every", type=int, default=250, help="Print every N epochs."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        n_x=args.n_x,
        n_snapshots=args.n_snapshots,
        n_collocation=args.n_collocation,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        epochs=args.epochs,
        lr=args.lr,
        energy_threshold=args.energy_threshold,
        max_modes=args.max_modes,
        seed=args.seed,
        log_every=args.log_every,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    """固定随机种子，减少不同运行之间的随机波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_solution(x: np.ndarray, t: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """计算给定空间网格和时间网格上的解析解快照矩阵。"""
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    solution = np.zeros_like(x_grid)
    for mode, amplitude in zip(cfg.mode_numbers, cfg.mode_amplitudes):
        decay = np.exp(-cfg.nu * (mode * math.pi) ** 2 * t_grid)
        solution += amplitude * np.sin(mode * math.pi * x_grid) * decay
    return solution


def compute_pod_basis(
    snapshot_matrix: np.ndarray,
    energy_threshold: float,
    max_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """对快照矩阵做 SVD，并按能量阈值与最大模态数选取 POD 基。

    返回值依次为：
    1. 截断后的空间模态矩阵
    2. 全部奇异值
    3. 每个奇异值对应的能量占比
    4. 累积能量占比
    5. 最终选取的模态数
    """

    spatial_modes, singular_values, _ = np.linalg.svd(
        snapshot_matrix, full_matrices=False
    )
    energy = singular_values**2
    energy_ratio = energy / np.sum(energy)
    cumulative_energy = np.cumsum(energy_ratio)
    # 用机器精度相关的阈值过滤掉数值上接近 0 的奇异值，避免把噪声当成有效模态。
    tolerance = np.finfo(singular_values.dtype).eps * max(snapshot_matrix.shape) * singular_values[0]
    active_modes = int(np.sum(singular_values > tolerance))
    active_modes = max(active_modes, 1)
    cumulative_active = cumulative_energy[:active_modes]

    if energy_threshold >= cumulative_active[-1] - 1.0e-12:
        rank = active_modes
    else:
        rank = int(np.searchsorted(cumulative_active, energy_threshold) + 1)

    rank = min(rank, max_modes, active_modes)
    return (
        spatial_modes[:, :rank],
        singular_values,
        energy_ratio,
        cumulative_energy,
        rank,
    )


def second_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    """用二阶中心差分近似空间二阶导数。

    这里的输入通常是 POD 空间基矩阵，每一列表示一个模态。
    输出与输入形状相同，边界点保留为 0，内部点使用中心差分。
    """

    second = np.zeros_like(values)
    second[1:-1, :] = (
        values[2:, :] - 2.0 * values[1:-1, :] + values[:-2, :]
    ) / (dx**2)
    return second


def scale_time(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    """把时间变量线性缩放到 `[-1, 1]` 区间，改善网络输入尺度。"""
    return 2.0 * (t - t_min) / (t_max - t_min) - 1.0


def coefficient_time_derivative(
    coefficients: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """逐个模态系数对时间求导，得到系数的一阶时间导数。

    因为网络输出是一个多维向量，这里逐列调用自动微分，再拼接回完整矩阵。
    """

    grads = []
    for index in range(coefficients.shape[1]):
        grad = torch.autograd.grad(
            coefficients[:, index : index + 1],
            t,
            grad_outputs=torch.ones_like(coefficients[:, index : index + 1]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(grad)
    return torch.cat(grads, dim=1)


def build_training_tensors(
    x: np.ndarray,
    t_snapshots: np.ndarray,
    snapshot_matrix: np.ndarray,
    basis: np.ndarray,
    basis_xx: np.ndarray,
    training_cfg: TrainingConfig,
    problem_cfg: ProblemConfig,
) -> dict[str, torch.Tensor]:
    """把 NumPy 数据整理成训练时常用的 PyTorch 张量。"""
    device = torch.device(
        "cuda" if training_cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    # 真系数由 POD 基与快照矩阵投影得到，用于后续作图对比，不直接参与损失计算。
    coefficients_true = (basis.T @ snapshot_matrix).T
    tensors = {
        "basis": torch.from_numpy(basis).to(device),
        "basis_xx": torch.from_numpy(basis_xx).to(device),
        "x": torch.from_numpy(x).to(device),
        "t_snapshots": torch.from_numpy(t_snapshots[:, None]).to(device),
        "coefficients_true": torch.from_numpy(coefficients_true).to(device),
        # `u0` 与 `t0` 用于构造初值损失，约束模型在初始时刻恢复真实解。
        "u0": torch.from_numpy(snapshot_matrix[:, 0][None, :]).to(device),
        "t0": torch.tensor([[problem_cfg.t_min]], dtype=torch.get_default_dtype()).to(device),
    }
    return tensors


def train_model(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    training_cfg: TrainingConfig,
    problem_cfg: ProblemConfig,
) -> tuple[list[float], list[float], list[float]]:
    """训练 POD-PINN，并记录总损失、PDE 损失和初值损失。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.lr)
    loss_history: list[float] = []
    pde_history: list[float] = []
    ic_history: list[float] = []

    basis = tensors["basis"]
    basis_xx = tensors["basis_xx"]
    u0 = tensors["u0"]
    t0 = tensors["t0"]
    device = basis.device

    for epoch in range(1, training_cfg.epochs + 1):
        # 在时间区间内随机采样若干配点，网络只需要学习这些时刻上的模态系数。
        t_f = torch.rand(
            training_cfg.n_collocation,
            1,
            device=device,
            dtype=torch.get_default_dtype(),
        )
        t_f = problem_cfg.t_min + (problem_cfg.t_max - problem_cfg.t_min) * t_f
        t_f.requires_grad_(True)

        coefficients_f = model(scale_time(t_f, problem_cfg.t_min, problem_cfg.t_max))
        coefficients_t = coefficient_time_derivative(coefficients_f, t_f)

        # 由系数和空间基重建 `u_t` 与 `u_xx`，再构造降阶后的 PDE 残差。
        u_t = coefficients_t @ basis.T
        u_xx = coefficients_f @ basis_xx.T
        # 边界点采用齐次 Dirichlet 条件，POD 基在边界上本来就接近 0，
        # 因此这里主要在内部点上计算热方程残差。
        residual = u_t[:, 1:-1] - problem_cfg.nu * u_xx[:, 1:-1]
        loss_pde = torch.mean(residual**2)

        # 初值损失要求 t = 0 时的降阶重建结果与真实初值一致。
        coefficients_0 = model(scale_time(t0, problem_cfg.t_min, problem_cfg.t_max))
        u0_pred = coefficients_0 @ basis.T
        loss_ic = torch.mean((u0_pred - u0) ** 2)

        loss = loss_pde + 10.0 * loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))
        pde_history.append(float(loss_pde.detach().cpu()))
        ic_history.append(float(loss_ic.detach().cpu()))

        if epoch == 1 or epoch % training_cfg.log_every == 0:
            # 输出训练日志，帮助观察物理残差和初值约束是否都在收敛。
            print(
                f"Epoch {epoch:5d} | total={loss_history[-1]:.6e} "
                f"| pde={pde_history[-1]:.6e} | ic={ic_history[-1]:.6e}"
            )

    return loss_history, pde_history, ic_history


def evaluate_model(
    model: nn.Module,
    basis: np.ndarray,
    x: np.ndarray,
    t_eval: np.ndarray,
    exact_eval: np.ndarray,
    training_cfg: TrainingConfig,
    problem_cfg: ProblemConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """在评估时间网格上预测模态系数和完整场，并计算误差指标。"""
    device = torch.device(
        "cuda" if training_cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    basis_torch = torch.from_numpy(basis).to(device)
    t_tensor = torch.from_numpy(t_eval[:, None]).to(device)

    model.eval()
    with torch.no_grad():
        coefficients_pred = model(scale_time(t_tensor, problem_cfg.t_min, problem_cfg.t_max))
        u_pred = coefficients_pred @ basis_torch.T

    # `u_pred_np` 的形状是 `(n_t, n_x)`，因此把真解转置后再做逐元素比较。
    coefficients_pred_np = coefficients_pred.cpu().numpy()
    u_pred_np = u_pred.cpu().numpy()
    exact_eval_nt = exact_eval.T

    relative_l2_error = float(
        np.linalg.norm(u_pred_np - exact_eval_nt) / np.linalg.norm(exact_eval_nt)
    )
    final_time_error = float(
        np.linalg.norm(u_pred_np[-1] - exact_eval_nt[-1]) / np.linalg.norm(exact_eval_nt[-1])
    )
    max_abs_error = float(np.max(np.abs(u_pred_np - exact_eval_nt)))

    metrics = {
        "relative_l2_error": relative_l2_error,
        "final_time_relative_error": final_time_error,
        "max_abs_error": max_abs_error,
        "n_x": int(x.size),
        "n_t_eval": int(t_eval.size),
    }
    return coefficients_pred_np, u_pred_np, metrics


def save_plots(
    output_dir: Path,
    x: np.ndarray,
    t_eval: np.ndarray,
    exact_eval: np.ndarray,
    u_pred_nt: np.ndarray,
    coefficients_true_nt: np.ndarray,
    coefficients_pred_nt: np.ndarray,
    singular_values: np.ndarray,
    cumulative_energy: np.ndarray,
    loss_history: list[float],
) -> None:
    """保存 POD 光谱、系数拟合、时空场对比以及训练损失图。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    # 同一张图上展示奇异值和累积能量，方便判断保留模态数是否合理。
    plt.semilogy(singular_values, marker="o", label="Singular values")
    plt.plot(cumulative_energy, marker="s", label="Cumulative energy")
    plt.xlabel("Mode index")
    plt.ylabel("Magnitude")
    plt.title("POD spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pod_spectrum.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(coefficients_true_nt.shape[1], 1, figsize=(8, 2.5 * coefficients_true_nt.shape[1]), sharex=True)
    if coefficients_true_nt.shape[1] == 1:
        axes = [axes]
    for index, axis in enumerate(axes):
        # 对每个保留模态分别比较真实系数与网络学习到的系数随时间的演化。
        axis.plot(t_eval, coefficients_true_nt[:, index], label="True coefficient")
        axis.plot(t_eval, coefficients_pred_nt[:, index], "--", label="PINN coefficient")
        axis.set_ylabel(f"a{index + 1}(t)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("t")
    fig.suptitle("Reduced coefficients learned by PINN")
    fig.tight_layout()
    fig.savefig(output_dir / "coefficients.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    extent = [x.min(), x.max(), t_eval.max(), t_eval.min()]
    # 绘制真解、预测解和误差三张时空图，便于直观看出降阶模型的整体精度。
    images = [
        exact_eval.T,
        u_pred_nt,
        u_pred_nt - exact_eval.T,
    ]
    titles = ["Exact field", "POD-PINN prediction", "Prediction error"]
    for axis, image, title in zip(axes, images, titles):
        im = axis.imshow(image, aspect="auto", extent=extent, cmap="coolwarm")
        axis.set_ylabel("t")
        axis.set_title(title)
        fig.colorbar(im, ax=axis)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(output_dir / "space_time_comparison.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, exact_eval[:, -1], label="Exact")
    plt.plot(x, u_pred_nt[-1], "--", label="POD-PINN")
    plt.xlabel("x")
    plt.ylabel("u(x, t_max)")
    plt.title("Final time profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_time_profile.png", dpi=180)
    plt.close()


def save_prediction_data(
    output_dir: Path,
    x: np.ndarray,
    t_eval: np.ndarray,
    exact_eval: np.ndarray,
    u_pred_nt: np.ndarray,
    coefficients_true_nt: np.ndarray,
    coefficients_pred_nt: np.ndarray,
) -> None:
    """保存完整场预测、最终时刻误差和模态系数，供后续比较脚本使用。"""
    np.savez(
        output_dir / "field_data.npz",
        x=x,
        t=t_eval,
        exact=exact_eval,
        prediction=u_pred_nt.T,
        exact_final=exact_eval[:, -1],
        prediction_final=u_pred_nt[-1],
        final_error=u_pred_nt[-1] - exact_eval[:, -1],
        coefficients_true=coefficients_true_nt,
        coefficients_pred=coefficients_pred_nt,
    )


def main() -> None:
    """执行 POD 基构造、PINN 训练、评估与结果保存的完整流程。"""
    training_cfg = parse_args()
    problem_cfg = ProblemConfig()
    set_seed(training_cfg.seed)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(problem_cfg.x_min, problem_cfg.x_max, training_cfg.n_x)
    t_snapshots = np.linspace(
        problem_cfg.t_min, problem_cfg.t_max, training_cfg.n_snapshots
    )
    dx = x[1] - x[0]

    # 先用解析解生成快照矩阵，再通过 SVD 提取主导空间结构。
    snapshot_matrix = exact_solution(x, t_snapshots, problem_cfg)
    basis, singular_values, energy_ratio, cumulative_energy, rank = compute_pod_basis(
        snapshot_matrix, training_cfg.energy_threshold, training_cfg.max_modes
    )
    basis_xx = second_derivative(basis, dx)
    coefficients_true_nt = (basis.T @ snapshot_matrix).T

    # 打印降阶结果，便于确认当前保留了多少个主导模态。
    print("Selected POD rank:", rank)
    print("Energy ratios:", np.round(energy_ratio[:rank], 6))
    print("Cumulative energy:", np.round(cumulative_energy[:rank], 6))

    tensors = build_training_tensors(
        x,
        t_snapshots,
        snapshot_matrix,
        basis,
        basis_xx,
        training_cfg,
        problem_cfg,
    )

    # 网络输出维度等于最终保留的 POD 模态数。
    model = CoefficientNet(
        out_dim=rank,
        width=training_cfg.hidden_width,
        depth=training_cfg.hidden_depth,
    ).to(tensors["basis"].device)

    # 记录纯训练耗时，便于和普通 PINN 比较计算开销。
    start_time = time.time()
    loss_history, _, _ = train_model(model, tensors, training_cfg, problem_cfg)
    training_seconds = time.time() - start_time

    t_eval = t_snapshots
    exact_eval = snapshot_matrix
    coefficients_pred_nt, u_pred_nt, metrics = evaluate_model(
        model, basis, x, t_eval, exact_eval, training_cfg, problem_cfg
    )

    save_plots(
        output_dir,
        x,
        t_eval,
        exact_eval,
        u_pred_nt,
        coefficients_true_nt,
        coefficients_pred_nt,
        singular_values,
        cumulative_energy,
        loss_history,
    )
    save_prediction_data(
        output_dir,
        x,
        t_eval,
        exact_eval,
        u_pred_nt,
        coefficients_true_nt,
        coefficients_pred_nt,
    )

    # 汇总关键信息到 JSON，供比较脚本和 README 中的数据引用。
    summary = {
        "model_name": "pod_pinn",
        "problem": asdict(problem_cfg),
        "training": asdict(training_cfg),
        "selected_rank": rank,
        "parameter_count": count_parameters(model),
        "training_seconds": training_seconds,
        "metrics": metrics,
        "output_dir": str(output_dir),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Training finished in {training_seconds:.2f} seconds.")
    print(
        "Relative L2 error: "
        f"{metrics['relative_l2_error']:.6e} | "
        f"Final time error: {metrics['final_time_relative_error']:.6e} | "
        f"Max abs error: {metrics['max_abs_error']:.6e}"
    )
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""普通 PINN 求解一维热方程的基线脚本。

该脚本直接学习时空映射 `(x, t) -> u(x, t)`，并通过三个损失项约束模型：
1. PDE 残差损失：逼近热方程 `u_t - nu * u_xx = 0`
2. 边界条件损失：约束 `x = 0` 和 `x = 1` 处的解为 0
3. 初值条件损失：约束 `t = 0` 时的解匹配解析初值

训练完成后会输出预测场、误差指标和若干可视化图片，供后续与 POD-PINN 做比较。
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

# 使用非交互式后端，保证脚本在服务器或无界面环境中也能保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# 本项目统一使用 float64，提高 PDE 残差计算时的数值稳定性。
torch.set_default_dtype(torch.float64)


@dataclass
class ProblemConfig:
    """热方程物理问题配置。

    这里采用多个正弦模态叠加作为初值，因此解析解可以直接写出，
    便于评估 PINN 的预测误差。
    """

    nu: float = 0.01
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    mode_numbers: tuple[int, ...] = (1, 2, 4)
    mode_amplitudes: tuple[float, ...] = (1.0, 0.7, -0.5)


@dataclass
class TrainingConfig:
    """训练与评估阶段使用的超参数配置。"""

    n_x_eval: int = 256
    n_t_eval: int = 160
    n_collocation: int = 768
    n_boundary: int = 256
    n_initial: int = 256
    hidden_width: int = 64
    hidden_depth: int = 4
    epochs: int = 1200
    lr: float = 1.0e-3
    seed: int = 42
    log_every: int = 250
    device: str = "cpu"


class FieldNet(nn.Module):
    """标准全连接网络，输入为空间和时间，输出为温度场标量值。"""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        # 第一层接收二维输入 `(x, t)`，之后堆叠若干隐藏层，最后输出单个标量。
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """对线性层做 Xavier 初始化，帮助网络更稳定地开始训练。"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """执行前向传播，返回对 `u(x, t)` 的预测值。"""
        return self.net(features)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，用于和其他模型做规模对比。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parse_args() -> TrainingConfig:
    """解析命令行参数，并映射成统一的训练配置对象。"""
    parser = argparse.ArgumentParser(
        description="Plain PINN baseline on the 1D heat equation."
    )
    parser.add_argument("--n-x-eval", type=int, default=256, help="Spatial evaluation points.")
    parser.add_argument("--n-t-eval", type=int, default=160, help="Temporal evaluation points.")
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=768,
        help="Interior collocation points per epoch.",
    )
    parser.add_argument(
        "--n-boundary",
        type=int,
        default=256,
        help="Boundary condition points per epoch.",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=256,
        help="Initial condition points per epoch.",
    )
    parser.add_argument(
        "--hidden-width", type=int, default=64, help="Hidden layer width."
    )
    parser.add_argument(
        "--hidden-depth", type=int, default=4, help="Number of hidden layers."
    )
    parser.add_argument("--epochs", type=int, default=1200, help="Adam iterations.")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate.")
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
        n_x_eval=args.n_x_eval,
        n_t_eval=args.n_t_eval,
        n_collocation=args.n_collocation,
        n_boundary=args.n_boundary,
        n_initial=args.n_initial,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子，提升实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_solution(x: np.ndarray, t: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """计算一维热方程在给定网格上的解析解。"""
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    solution = np.zeros_like(x_grid)
    for mode, amplitude in zip(cfg.mode_numbers, cfg.mode_amplitudes):
        # 每个正弦模态都会随时间按指数规律衰减。
        decay = np.exp(-cfg.nu * (mode * math.pi) ** 2 * t_grid)
        solution += amplitude * np.sin(mode * math.pi * x_grid) * decay
    return solution


def exact_initial_condition(x: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    """根据模态叠加公式生成 `t = 0` 时刻的精确初值。"""
    initial = np.zeros_like(x)
    for mode, amplitude in zip(cfg.mode_numbers, cfg.mode_amplitudes):
        initial += amplitude * np.sin(mode * math.pi * x)
    return initial


def scale_inputs(
    x: torch.Tensor, t: torch.Tensor, problem_cfg: ProblemConfig
) -> torch.Tensor:
    """将物理域中的 `(x, t)` 线性缩放到 `[-1, 1]`，有助于网络训练。"""
    x_scaled = 2.0 * (x - problem_cfg.x_min) / (problem_cfg.x_max - problem_cfg.x_min) - 1.0
    t_scaled = 2.0 * (t - problem_cfg.t_min) / (problem_cfg.t_max - problem_cfg.t_min) - 1.0
    return torch.cat([x_scaled, t_scaled], dim=1)


def sample_uniform(
    count: int, low: float, high: float, device: torch.device
) -> torch.Tensor:
    """在给定区间内均匀采样一批列向量，用作训练点。"""
    values = torch.rand(count, 1, device=device, dtype=torch.get_default_dtype())
    return low + (high - low) * values


def train_model(
    model: nn.Module,
    training_cfg: TrainingConfig,
    problem_cfg: ProblemConfig,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """训练普通 PINN，并记录总损失、PDE、边界和初值损失的历史。

    返回值中的四个列表按时间顺序分别对应：
    总损失、PDE 残差损失、边界条件损失、初值条件损失。
    """

    device = torch.device(
        "cuda" if training_cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.lr)
    loss_history: list[float] = []
    pde_history: list[float] = []
    bc_history: list[float] = []
    ic_history: list[float] = []

    x0_np = np.linspace(problem_cfg.x_min, problem_cfg.x_max, training_cfg.n_initial)
    u0_np = exact_initial_condition(x0_np, problem_cfg)[:, None]
    x0 = torch.from_numpy(x0_np[:, None]).to(device)
    t0 = torch.full_like(x0, problem_cfg.t_min)
    u0 = torch.from_numpy(u0_np).to(device)

    for epoch in range(1, training_cfg.epochs + 1):
        # 1) 在时空内部采样配点，用于构造 PDE 残差损失。
        x_f = sample_uniform(
            training_cfg.n_collocation, problem_cfg.x_min, problem_cfg.x_max, device
        )
        t_f = sample_uniform(
            training_cfg.n_collocation, problem_cfg.t_min, problem_cfg.t_max, device
        )
        x_f.requires_grad_(True)
        t_f.requires_grad_(True)

        features_f = scale_inputs(x_f, t_f, problem_cfg)
        u_f = model(features_f)
        # 通过自动微分得到一阶和二阶导数，进而构造热方程残差。
        u_x = torch.autograd.grad(
            u_f, x_f, grad_outputs=torch.ones_like(u_f), create_graph=True
        )[0]
        u_t = torch.autograd.grad(
            u_f, t_f, grad_outputs=torch.ones_like(u_f), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        residual = u_t - problem_cfg.nu * u_xx
        loss_pde = torch.mean(residual**2)

        # 2) 在左右边界处采样时间点，约束齐次 Dirichlet 边界条件。
        tb = sample_uniform(
            training_cfg.n_boundary, problem_cfg.t_min, problem_cfg.t_max, device
        )
        xb_left = torch.full_like(tb, problem_cfg.x_min)
        xb_right = torch.full_like(tb, problem_cfg.x_max)
        u_left = model(scale_inputs(xb_left, tb, problem_cfg))
        u_right = model(scale_inputs(xb_right, tb, problem_cfg))
        loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

        # 3) 使用解析初值监督 `t = 0` 时刻的预测结果。
        u0_pred = model(scale_inputs(x0, t0, problem_cfg))
        loss_ic = torch.mean((u0_pred - u0) ** 2)

        # 初值和边界通常比 PDE 残差更容易影响解的整体形态，这里给予更大权重。
        loss = loss_pde + 5.0 * loss_bc + 10.0 * loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))
        pde_history.append(float(loss_pde.detach().cpu()))
        bc_history.append(float(loss_bc.detach().cpu()))
        ic_history.append(float(loss_ic.detach().cpu()))

        if epoch == 1 or epoch % training_cfg.log_every == 0:
            # 定期打印日志，方便观察不同损失项是否同步下降。
            print(
                f"Epoch {epoch:5d} | total={loss_history[-1]:.6e} "
                f"| pde={pde_history[-1]:.6e} | bc={bc_history[-1]:.6e} "
                f"| ic={ic_history[-1]:.6e}"
            )

    return loss_history, pde_history, bc_history, ic_history


def evaluate_model(
    model: nn.Module,
    x_eval: np.ndarray,
    t_eval: np.ndarray,
    exact_eval: np.ndarray,
    training_cfg: TrainingConfig,
    problem_cfg: ProblemConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """在规则网格上评估模型，并计算相对 L2 误差等指标。"""
    device = torch.device(
        "cuda" if training_cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    x_grid, t_grid = np.meshgrid(x_eval, t_eval, indexing="ij")
    # 将二维网格展平后送入网络，再在末尾恢复成二维场。
    features = np.stack([x_grid.ravel(), t_grid.ravel()], axis=1)
    features_tensor = torch.from_numpy(features).to(device)
    x_tensor = features_tensor[:, :1]
    t_tensor = features_tensor[:, 1:]

    model.eval()
    with torch.no_grad():
        u_pred = model(scale_inputs(x_tensor, t_tensor, problem_cfg))

    u_pred_np = u_pred.cpu().numpy().reshape(x_eval.size, t_eval.size)
    relative_l2_error = float(
        np.linalg.norm(u_pred_np - exact_eval) / np.linalg.norm(exact_eval)
    )
    final_time_error = float(
        np.linalg.norm(u_pred_np[:, -1] - exact_eval[:, -1]) / np.linalg.norm(exact_eval[:, -1])
    )
    max_abs_error = float(np.max(np.abs(u_pred_np - exact_eval)))

    metrics = {
        "relative_l2_error": relative_l2_error,
        "final_time_relative_error": final_time_error,
        "max_abs_error": max_abs_error,
        "n_x": int(x_eval.size),
        "n_t_eval": int(t_eval.size),
    }
    return u_pred_np, metrics


def save_plots(
    output_dir: Path,
    x_eval: np.ndarray,
    t_eval: np.ndarray,
    exact_eval: np.ndarray,
    u_pred: np.ndarray,
    loss_history: list[float],
) -> None:
    """保存时空场对比图、训练损失曲线和最终时刻剖面图。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    # 通过转置把数组从 `(x, t)` 形式改成 imshow 更适合展示的 `(t, x)` 形式。
    extent = [x_eval.min(), x_eval.max(), t_eval.max(), t_eval.min()]
    images = [exact_eval.T, u_pred.T, (u_pred - exact_eval).T]
    titles = ["Exact field", "Plain PINN prediction", "Prediction error"]
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
    plt.title("Plain PINN training loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x_eval, exact_eval[:, -1], label="Exact")
    plt.plot(x_eval, u_pred[:, -1], "--", label="Plain PINN")
    plt.xlabel("x")
    plt.ylabel("u(x, t_max)")
    plt.title("Final time profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_time_profile.png", dpi=180)
    plt.close()


def save_prediction_data(
    output_dir: Path,
    x_eval: np.ndarray,
    t_eval: np.ndarray,
    exact_eval: np.ndarray,
    u_pred: np.ndarray,
) -> None:
    """将预测场及最终时刻切片保存为 `.npz`，便于后续统一比较。"""
    np.savez(
        output_dir / "field_data.npz",
        x=x_eval,
        t=t_eval,
        exact=exact_eval,
        prediction=u_pred,
        exact_final=exact_eval[:, -1],
        prediction_final=u_pred[:, -1],
        final_error=u_pred[:, -1] - exact_eval[:, -1],
    )


def main() -> None:
    """执行完整流程：配置、训练、评估、绘图和结果落盘。"""
    training_cfg = parse_args()
    problem_cfg = ProblemConfig()
    set_seed(training_cfg.seed)

    output_dir = Path(__file__).resolve().parent / "outputs_plain_pinn"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_eval = np.linspace(problem_cfg.x_min, problem_cfg.x_max, training_cfg.n_x_eval)
    t_eval = np.linspace(problem_cfg.t_min, problem_cfg.t_max, training_cfg.n_t_eval)
    exact_eval = exact_solution(x_eval, t_eval, problem_cfg)

    # 模型直接在二维输入 `(x, t)` 上学习温度场。
    model = FieldNet(
        width=training_cfg.hidden_width,
        depth=training_cfg.hidden_depth,
    ).to(
        torch.device(
            "cuda"
            if training_cfg.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
    )

    # 统计纯训练阶段耗时，便于与 POD-PINN 的训练成本直接比较。
    start_time = time.time()
    loss_history, _, _, _ = train_model(model, training_cfg, problem_cfg)
    training_seconds = time.time() - start_time

    u_pred, metrics = evaluate_model(
        model, x_eval, t_eval, exact_eval, training_cfg, problem_cfg
    )
    save_plots(output_dir, x_eval, t_eval, exact_eval, u_pred, loss_history)
    save_prediction_data(output_dir, x_eval, t_eval, exact_eval, u_pred)

    # summary.json 保留所有关键配置和指标，供比较脚本统一读取。
    summary = {
        "model_name": "plain_pinn",
        "problem": asdict(problem_cfg),
        "training": asdict(training_cfg),
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

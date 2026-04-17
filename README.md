# POD-PINN: 1D Heat Equation Example

This folder now contains both:

- a POD-PINN example
- a plain PINN baseline without POD

Both scripts solve the same 1D heat equation, so you can compare accuracy, training time, and model size directly.

## What the example does

The script solves the 1D heat equation

`u_t = nu * u_xx, x in [0, 1], t in [0, 1]`

with homogeneous Dirichlet boundary conditions and an analytic initial field built from a few sine modes.

The workflow is:

1. Generate high-fidelity snapshots from the analytic solution.
2. Apply POD to extract a low-dimensional spatial basis.
3. Represent the field as `u(x, t) ~= sum_i a_i(t) * phi_i(x)`.
4. Train a PINN that takes `t` as input and predicts the reduced coefficients `a_i(t)`.
5. Reconstruct the full field and compare it with the exact solution.

## Files

- `pod_pinn_heat1d.py`: POD + PINN example
- `plain_pinn_heat1d.py`: ordinary PINN baseline
- `compare_pinn_models.py`: reads both `summary.json` files and prints a comparison table
- `plot_final_comparison.py`: draws POD-PINN and plain PINN final solution/error curves in one figure
- `plot_space_time_model_comparison.py`: draws a 2x2 `(x, t)` comparison figure with predictions on the first row and error fields on the second row, using a shared color scale within each row
- `TRAINING_DESIGN_CN.md`: Chinese write-up of the full mathematical and code design
- `outputs/`: POD-PINN outputs
- `outputs_plain_pinn/`: plain PINN outputs

## Requirements

The script uses:

- `numpy`
- `matplotlib`
- `torch`

## Run

From this folder:

Run POD-PINN:

```powershell
python .\pod_pinn_heat1d.py
```

Run plain PINN:

```powershell
python .\plain_pinn_heat1d.py
```

Print a side-by-side comparison after both runs finish:

```powershell
python .\compare_pinn_models.py
```

Draw the final-time solution and error curves in one figure:

```powershell
python .\plot_final_comparison.py
```

Draw the 2D `(x, t)` prediction/error comparison figure:

```powershell
python .\plot_space_time_model_comparison.py
```

Quick checks:

```powershell
python .\pod_pinn_heat1d.py --epochs 1200 --n-collocation 128
python .\plain_pinn_heat1d.py --epochs 1200 --n-collocation 512
```

The POD-PINN default settings now keep the full set of active modes in this toy problem. This matters here because the analytic solution is built from three true sine modes, and truncating to two modes introduces a visible final-time bias even though the third mode carries very little total energy.

## Outputs

After execution:

- `pod_pinn_heat1d.py` writes plots and `summary.json` into `outputs/`
- `plain_pinn_heat1d.py` writes plots and `summary.json` into `outputs_plain_pinn/`
- `compare_pinn_models.py` writes `comparison_summary.json`
- `plot_final_comparison.py` writes `final_solution_error_comparison.png`
- `plot_space_time_model_comparison.py` writes `space_time_model_comparison.png`
- both training scripts also save `field_data.npz` for later plotting without retraining

## Notes

- The POD-PINN example uses POD to reduce the spatial dimension and uses PINN only for the temporal dynamics in the reduced space.
- The plain PINN baseline learns the full field `u(x, t)` directly from PDE residuals, boundary conditions, and the initial condition.
- The space-time comparison script uses a 2x2 layout: the first row shows predicted fields and the second row shows error fields for POD-PINN and plain PINN side by side.
- The two prediction plots share one color scale, and the two error plots share another symmetric color scale, so horizontal comparisons stay visually meaningful.
- Because the snapshots satisfy zero boundary values, the POD basis also respects the boundary condition.
- If you want to extend this to Burgers, wave, or 2D diffusion problems, the same pattern still works: first build a POD basis, then let PINN learn the reduced coefficients from the governing physics.

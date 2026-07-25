from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import checkpoint_dir, load_config
from .data import load_dataset
from .deps import require_numpy, require_torch
from .models import TransformerTrajectoryModel
from .utils import set_seed, write_metrics_csv


def normalize_positions(np, positions):
    out = positions.astype(np.float32).copy()
    out[..., 0] /= 90.0
    out[..., 1] /= 180.0
    out[..., 2] /= 20.0
    return out


def build_samples(cfg, horizon: int, fast: bool = False):
    np = require_numpy()
    data = load_dataset(cfg)
    history = int(cfg["data"]["history_length"])
    plane_pos = normalize_positions(np, data["plane_pos"])
    xs = []
    ys = []
    stride = 4 if fast else 1
    max_t = plane_pos.shape[0] - horizon
    for k in range(plane_pos.shape[1]):
        for t in range(history, max_t, stride):
            xs.append(plane_pos[t - history : t, k, :])
            ys.append(plane_pos[t : t + horizon, k, :])
    return np.stack(xs), np.stack(ys)


def train_transformer(config: str | Path, horizon: int, fast: bool = False, epochs: int | None = None) -> Path:
    cfg = load_config(config)
    set_seed(int(cfg["project"]["seed"]) + horizon)
    np = require_numpy()
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device_label = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})"
    else:
        device_label = "cpu (torch.cuda.is_available() is False)"
    print(f"Using device: {device_label}", flush=True)
    x, y = build_samples(cfg, horizon, fast=fast)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    perm = rng.permutation(len(x))
    split = int(0.85 * len(x))
    train_idx, val_idx = perm[:split], perm[split:]
    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    tcfg = cfg["transformer"]
    model = TransformerTrajectoryModel(
        history_length=int(cfg["data"]["history_length"]),
        horizon=horizon,
        d_model=int(tcfg["d_model"]),
        nhead=int(tcfg["nhead"]),
        num_layers=int(tcfg["num_layers"]),
        dim_feedforward=int(tcfg["dim_feedforward"]),
        dropout=float(tcfg["dropout"]),
    ).module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    batch_size = int(tcfg["batch_size"])
    n_epochs = int(epochs or (tcfg["fast_epochs"] if fast else tcfg["epochs"]))
    rows = []
    started = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        print(f"transformer h={horizon} epoch={epoch} train={rows[-1]['train_loss']:.6f} val={val_loss:.6f}", flush=True)

    out = checkpoint_dir(cfg) / f"transformer_h{horizon}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "horizon": horizon,
            "history_length": int(cfg["data"]["history_length"]),
            "config": cfg,
            "device": str(device),
            "device_label": device_label,
            "elapsed_s": time.perf_counter() - started,
        },
        out,
    )
    metrics = checkpoint_dir(cfg).parent / "metrics" / f"transformer_h{horizon}.csv"
    write_metrics_csv(metrics, rows)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Transformer trajectory predictor.")
    parser.add_argument("--config", default="configs/paper_result.yaml")
    parser.add_argument("--horizon", type=int, choices=[5, 25], required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    out = train_transformer(args.config, args.horizon, fast=args.fast, epochs=args.epochs)
    print(f"Wrote checkpoint to {out}")


if __name__ == "__main__":
    main()

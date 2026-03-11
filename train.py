import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
python train.py --data_dir traindata --epochs 400 --w_uy 5 --batch_size 2 --lr 5e-05 --loss wmse --w_k 5 --w_eps 5 --width 64 --modes_x 32 --modes_y 16 --n_blocks 4 --lambda_div 1.0 --early_stop 100


python train.py --data_dir traindata --epochs 600 --batch_size 2 --lr 5e-05 --loss wmse --w_k 5 --w_eps 5 --width 64 --modes_x 32 --modes_y 16 --n_blocks 4 --lambda_div 1.0 --early_stop 120 --resume fno2d_openfoam.pth
"""

# =========================
# Utils
# =========================
def find_npz_files(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found under: {root_dir}")
    return files


def compute_global_scales(npz_files):
    u_max = 0.0
    k_max = 0.0
    eps_max = 0.0

    x_min = float("inf")
    x_max = -float("inf")
    y_min = float("inf")
    y_max = -float("inf")

    for f in npz_files:
        d = np.load(f, allow_pickle=True)

        if "stats" in d:
            stats = d["stats"][0]
            u_max = max(u_max, float(stats.get("u_max", 0.0)))
            k_max = max(k_max, float(stats.get("k_max", 0.0)))
            eps_max = max(eps_max, float(stats.get("eps_max", 0.0)))
        else:
            if "output" in d:
                out = d["output"]  # (4,Nx,Ny)
                u_max = max(u_max, float(np.max(np.abs(out[0:2]))))
                k_max = max(k_max, float(np.max(out[2])))
                eps_max = max(eps_max, float(np.max(out[3])))

        if "input" in d:
            inp = d["input"]  # (3,Nx,Ny)  [phi, X, Y]
            X = inp[1]
            Y = inp[2]
            x_min = min(x_min, float(np.min(X)))
            x_max = max(x_max, float(np.max(X)))
            y_min = min(y_min, float(np.min(Y)))
            y_max = max(y_max, float(np.max(Y)))

    scales = {
        "u_ref": max(u_max, 1e-12),
        "k_ref": max(k_max, 1e-12),
        "eps_ref": max(eps_max, 1e-12),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    print("✅ Global scales:", scales)
    return scales


def _move_optimizer_state_to_device(optimizer, device):
    # 有些情况下 load_state_dict 后 state tensor 在 CPU，这里强制搬到 GPU/CPU一致
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# =========================
# Dataset
# =========================
class OpenFOAMNPZDataset(Dataset):
    """
    expects each .npz to include:
      - input : (3, Nx, Ny)   channels [phi, X, Y]
      - output: (4, Nx, Ny)   channels [Ux, Uy, k, eps]
    """

    def __init__(self, files, scales, k_eps_floor=1e-6):
        self.files = list(files)
        self.u_ref = float(scales["u_ref"])
        self.k_ref = float(scales["k_ref"])
        self.eps_ref = float(scales["eps_ref"])
        self.k_eps_floor = float(k_eps_floor)

        self.x_min = float(scales["x_min"])
        self.x_max = float(scales["x_max"])
        self.y_min = float(scales["y_min"])
        self.y_max = float(scales["y_max"])
        self.xy_eps = 1e-12

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        if "input" not in data or "output" not in data:
            raise KeyError(f"{path} missing 'input' or 'output' keys.")

        x = torch.from_numpy(data["input"]).to(torch.float32).permute(1, 2, 0)  # (Nx,Ny,3)

        # normalize X,Y to [-1,1]
        X = x[..., 1]
        Y = x[..., 2]
        X = 2.0 * (X - self.x_min) / (self.x_max - self.x_min + self.xy_eps) - 1.0
        Y = 2.0 * (Y - self.y_min) / (self.y_max - self.y_min + self.xy_eps) - 1.0
        x[..., 1] = X
        x[..., 2] = Y

        y = torch.from_numpy(data["output"]).to(torch.float32)  # (4,Nx,Ny)

        # U scaling
        y[0:2] = y[0:2] / self.u_ref

        # k, eps log scaling (strictly positive)
        k_norm = y[2] / self.k_ref
        eps_norm = y[3] / self.eps_ref
        k_norm = torch.clamp(k_norm, min=self.k_eps_floor)
        eps_norm = torch.clamp(eps_norm, min=self.k_eps_floor)

        y[2] = torch.log(k_norm)
        y[3] = torch.log(eps_norm)

        y = y.permute(1, 2, 0)  # (Nx,Ny,4)
        return x, y


# =========================
# Losses (masked)
# =========================
def masked_mse_loss(y_pred, y_true, x_in, eps=1e-12):
    mask = (x_in[..., 0] > 0).float().unsqueeze(-1)  # [B,Nx,Ny,1]
    diff2 = ((y_pred - y_true) ** 2) * mask
    denom = mask.sum() * y_true.shape[-1] + eps
    return diff2.sum() / denom


def masked_weighted_mse_loss(y_pred, y_true, x_in, weights=(1.0, 1.0, 3.0, 3.0), eps=1e-12):
    mask = (x_in[..., 0] > 0).float().unsqueeze(-1)  # [B,Nx,Ny,1]
    diff2 = ((y_pred - y_true) ** 2) * mask

    denom = mask.sum(dim=(1, 2), keepdim=True) + eps
    mse_ch = diff2.sum(dim=(1, 2)) / denom.squeeze(-1).squeeze(-1)  # [B,4]

    w = torch.tensor(weights, device=y_pred.device, dtype=y_pred.dtype).view(1, 4)
    return (mse_ch * w).sum(dim=1).mean()


def masked_divergence_free_loss(y_pred, x_in, scales, eps=1e-12):
    """
    div(u)=0 in physical space using FD on structured grid (X_norm,Y_norm are uniform in [-1,1]).
    """
    u = y_pred[..., 0]  # [B,Nx,Ny]
    v = y_pred[..., 1]
    mask = (x_in[..., 0] > 0).float()  # [B,Nx,Ny]

    B, Nx, Ny = u.shape
    dxn = 2.0 / (Nx - 1 + eps)
    dyn = 2.0 / (Ny - 1 + eps)

    du_dXn = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * dxn)
    dv_dYn = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2.0 * dyn)

    sx = 2.0 / (float(scales["x_max"]) - float(scales["x_min"]) + eps)
    sy = 2.0 / (float(scales["y_max"]) - float(scales["y_min"]) + eps)

    div = du_dXn * sx + dv_dYn * sy  # [B,Nx-2,Ny-2]
    mask_int = mask[:, 1:-1, 1:-1]
    return (div**2 * mask_int).sum() / (mask_int.sum() + eps)


# =========================
# FNO components
# =========================
class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        # x: [B,C,Nx,Ny]
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            (B, self.out_channels, x_ft.size(-2), x_ft.size(-1)),
            dtype=torch.cfloat,
            device=x.device
        )

        out_ft[:, :, :self.modes_x, :self.modes_y] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes_x, :self.modes_y],
            self.weights
        )

        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x_out


class FNOBlock(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super().__init__()
        self.fourier = FourierLayer2D(width, width, modes_x, modes_y)
        self.conv = nn.Conv2d(width, width, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x):
        return torch.relu(self.alpha * self.fourier(x) + self.conv(x))


class FNO2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, width=64, modes_x=32, modes_y=16, n_blocks=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes_x, modes_y) for _ in range(n_blocks)])
        self.fc1 = nn.Linear(width, width)
        self.fc_out = nn.Linear(width, out_channels)

    def forward(self, x):
        # x: [B,Nx,Ny,C]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return x


def denormalize_output(y_pred, scales):
    """
    y_pred: [B,Nx,Ny,4] channels:
      [Ux/u_ref, Uy/u_ref, log(k/k_ref), log(eps/eps_ref)]
    """
    y = y_pred.clone()
    y[..., 0:2] = y[..., 0:2] * float(scales["u_ref"])
    y[..., 2] = torch.exp(y[..., 2]) * float(scales["k_ref"])
    y[..., 3] = torch.exp(y[..., 3]) * float(scales["eps_ref"])
    return y


# =========================
# Train script
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="traindata")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes_x", type=int, default=32)
    ap.add_argument("--modes_y", type=int, default=16)
    ap.add_argument("--n_blocks", type=int, default=4)

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--loss", type=str, default="wmse", choices=["mmse", "wmse"])
    ap.add_argument("--w_ux", type=float, default=1.0)
    ap.add_argument("--w_uy", type=float, default=1.0)
    ap.add_argument("--w_k",  type=float, default=3.0)
    ap.add_argument("--w_eps",type=float, default=3.0)

    ap.add_argument("--lambda_div", type=float, default=0.0)
    ap.add_argument("--early_stop", type=int, default=60)
    ap.add_argument("--save_path", type=str, default="fno2d_openfoam.pth")

    # resume
    ap.add_argument("--resume", type=str, default="", help="resume from checkpoint .pth")
    ap.add_argument("--start_epoch", type=int, default=0, help="manual start epoch if needed (only if ckpt has no epoch)")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------
    # load ckpt early (for scales)
    # -----------------
    ckpt = None
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        print(f"✅ Found resume checkpoint: {args.resume}")

    files = find_npz_files(args.data_dir)
    print(f"Found {len(files)} npz files under: {args.data_dir}")

    n = len(files)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_val = int(round(n * args.val_ratio))
    n_train = n - n_val

    train_files = [files[i] for i in idx[:n_train]]
    val_files   = [files[i] for i in idx[n_train:]] if n_val > 0 else []
    print(f"Train: {len(train_files)}  Val: {len(val_files)}")

    # scales: prefer checkpoint scales to keep normalization consistent
    if ckpt is not None and "scales" in ckpt:
        scales = ckpt["scales"]
        print("✅ Using scales from checkpoint (recommended).")
    else:
        scales = compute_global_scales(train_files)

    train_ds = OpenFOAMNPZDataset(train_files, scales)
    val_ds = OpenFOAMNPZDataset(val_files, scales) if len(val_files) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = FNO2D(
        width=args.width,
        modes_x=args.modes_x,
        modes_y=args.modes_y,
        n_blocks=args.n_blocks
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, threshold=1e-4, min_lr=1e-6
    )

    x0, y0 = train_ds[0]
    print("Input shape :", x0.shape)
    print("Output shape:", y0.shape)

    # data loss
    if args.loss == "mmse":
        data_loss_fn = masked_mse_loss
    else:
        def data_loss_fn(y_pred, y_true, x_in):
            return masked_weighted_mse_loss(
                y_pred, y_true, x_in,
                weights=(args.w_ux, args.w_uy, args.w_k, args.w_eps)
            )

    # -----------------
    # resume states (model/optim/scheduler/best/epoch)
    # -----------------
    best_val = float("inf")
    no_improve = 0
    start_epoch = 0 if args.start_epoch is None else int(args.start_epoch)

    if ckpt is not None:
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print("✅ Loaded model_state.")
        else:
            raise KeyError("Checkpoint missing 'model_state'.")

        # optimizer/scheduler (new checkpoints)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            _move_optimizer_state_to_device(optimizer, device)
            print("✅ Loaded optimizer_state.")

            # allow CLI lr override
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            print(f"✅ Override lr to args.lr = {args.lr:.2e}")

        else:
            print("⚠️ Checkpoint has no optimizer_state (old ckpt). Optimizer will start fresh.")

        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            print("✅ Loaded scheduler_state.")
        else:
            print("⚠️ Checkpoint has no scheduler_state (old ckpt). Scheduler will start fresh.")

        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])
            print(f"✅ Loaded best_val = {best_val:.6e}")

        if "no_improve" in ckpt:
            no_improve = int(ckpt["no_improve"])
            print(f"✅ Loaded no_improve = {no_improve}")

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"✅ start_epoch set to {start_epoch} (ckpt epoch + 1)")
        else:
            print(f"⚠️ Checkpoint has no epoch. Using start_epoch={start_epoch} from args.")

    eps = 1e-12

    # =========================
    # Train loop
    # =========================
    for epoch in range(start_epoch, args.epochs):
        # -----------------
        # train
        # -----------------
        model.train()
        train_loss_sum = 0.0
        train_data_sum = 0.0
        train_div_sum  = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)

            data_loss = data_loss_fn(y_pred, y, x)

            div_loss = torch.zeros((), device=device)
            if args.lambda_div > 0:
                div_loss = masked_divergence_free_loss(y_pred, x, scales)

            loss = data_loss + args.lambda_div * div_loss
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_data_sum += float(data_loss.item())
            train_div_sum  += float(div_loss.item())

        n_batches = max(len(train_loader), 1)
        train_loss = train_loss_sum / n_batches
        train_data = train_data_sum / n_batches
        train_div  = train_div_sum  / n_batches

        # -----------------
        # val
        # -----------------
        if val_loader is not None:
            model.eval()
            val_data_sum = 0.0
            val_div_sum  = 0.0

            # diagnostics
            val_mask_frac_sum = 0.0
            val_true_ux_norm_sum = 0.0
            val_pred_ux_norm_sum = 0.0
            val_pred_ux_maxabs = 0.0

            # stable phys rel L2: accumulate norms over entire val set
            diff_norm_sum = torch.zeros(4, device=device)
            true_norm_sum = torch.zeros(4, device=device)

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)

                    # data/div
                    val_data_sum += float(data_loss_fn(y_pred, y, x).item())
                    if args.lambda_div > 0:
                        val_div_sum += float(masked_divergence_free_loss(y_pred, x, scales).item())

                    # mask
                    mask = (x[..., 0] > 0).float()  # [B,Nx,Ny]
                    val_mask_frac_sum += float(mask.mean().item())

                    # phys outputs
                    y_true_phys = denormalize_output(y, scales)
                    y_pred_phys = denormalize_output(y_pred, scales)

                    # Ux norms (masked)
                    ux_true = y_true_phys[..., 0] * mask
                    ux_pred = y_pred_phys[..., 0] * mask
                    val_true_ux_norm_sum += float(torch.linalg.vector_norm(ux_true.reshape(ux_true.shape[0], -1), dim=1).mean().item())
                    val_pred_ux_norm_sum += float(torch.linalg.vector_norm(ux_pred.reshape(ux_pred.shape[0], -1), dim=1).mean().item())
                    val_pred_ux_maxabs = max(val_pred_ux_maxabs, float(torch.max(torch.abs(ux_pred)).item()))

                    # phys rel L2 accumulate (masked, all channels)
                    m = mask.unsqueeze(-1)  # [B,Nx,Ny,1]
                    diff = (y_pred_phys - y_true_phys) * m
                    tru  = y_true_phys * m

                    B0 = diff.shape[0]
                    diff_flat = diff.reshape(B0, -1, 4)
                    true_flat = tru.reshape(B0, -1, 4)

                    diff_n = torch.linalg.vector_norm(diff_flat, dim=1).sum(dim=0)  # [4]
                    true_n = torch.linalg.vector_norm(true_flat, dim=1).sum(dim=0)  # [4]
                    diff_norm_sum += diff_n
                    true_norm_sum += true_n

            nvb = max(len(val_loader), 1)
            val_data = val_data_sum / nvb
            val_div  = val_div_sum  / nvb

            val_mask_frac = val_mask_frac_sum / nvb
            val_true_ux_norm = val_true_ux_norm_sum / nvb
            val_pred_ux_norm = val_pred_ux_norm_sum / nvb

            phys_rel = (diff_norm_sum / (true_norm_sum + eps)).detach().cpu().numpy()

            scheduler.step(val_data)
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:04d} | lr {lr_now:.2e} | "
                f"Train {train_loss:.3e} (data {train_data:.3e}, div {train_div:.3e}) | "
                f"ValData {val_data:.3e} | ValDiv {val_div:.3e} | "
                f"ValMaskFrac {val_mask_frac:.3e} | ValTrueUxNorm {val_true_ux_norm:.3e} | "
                f"ValPredUxNorm {val_pred_ux_norm:.3e} | ValPredUxMaxAbs {val_pred_ux_maxabs:.3e} | "
                f"PhysRelL2: Ux {phys_rel[0]:.3e}, Uy {phys_rel[1]:.3e}, k {phys_rel[2]:.3e}, eps {phys_rel[3]:.3e}"
            )

            # early stop based on val_data
            improved = val_data < best_val - 1e-6
            if improved:
                best_val = val_data
                no_improve = 0

                torch.save({
                    "epoch": epoch,
                    "best_val": best_val,
                    "no_improve": no_improve,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scales": scales,
                    "args": vars(args),
                }, args.save_path)

            else:
                no_improve += 1

                # save "last"
                torch.save({
                    "epoch": epoch,
                    "best_val": best_val,
                    "no_improve": no_improve,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scales": scales,
                    "args": vars(args),
                }, Path(args.save_path).with_suffix(".last.pth"))

                if no_improve >= args.early_stop:
                    print(f"🛑 Early stopping at epoch {epoch:04d}, best val_data = {best_val:.3e}")
                    break

        else:
            print(
                f"Epoch {epoch:04d} | Train {train_loss:.3e} (data {train_data:.3e}, div {train_div:.3e}) | Val nan"
            )

            torch.save({
                "epoch": epoch,
                "best_val": best_val,
                "no_improve": no_improve,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scales": scales,
                "args": vars(args),
            }, args.save_path)

    print(f"✅ Saved checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()

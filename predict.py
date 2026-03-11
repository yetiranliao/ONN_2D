
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# python .\predict.py --npz "traindata\xxx.npz" --ckpt "fno2d_openfoam.pth" --show
# python .\predict.py --npz "traindata\xxx.npz" --ckpt "checkpoints\best.pth" --savefig "results\pred.png"

# ============================================================
# FNO model (auto-match checkpoint hyperparams)
# ============================================================


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
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B,C,Nx,Ny/2+1]

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
        x = self.fc0(x)                  # [B,Nx,Ny,width]
        x = x.permute(0, 3, 1, 2)         # [B,width,Nx,Ny]
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1)         # [B,Nx,Ny,width]
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)                # [B,Nx,Ny,4]
        return x


# ============================================================
# denormalize model outputs back to physical
# channels: [Ux/u_ref, Uy/u_ref, log(k/k_ref), log(eps/eps_ref)]
# ============================================================
def denormalize_output(y_pred_norm, scales):
    y = y_pred_norm.clone()
    y[..., 0:2] = y[..., 0:2] * float(scales["u_ref"])
    y[..., 2] = torch.exp(y[..., 2]) * float(scales["k_ref"])
    y[..., 3] = torch.exp(y[..., 3]) * float(scales["eps_ref"])
    return y


def _pick_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def normalize_xy_like_training(x_in, scales):
    """
    training里对 input 的 X,Y 做了 [-1,1] 归一化：
      Xn = 2*(X-xmin)/(xmax-xmin) - 1
      Yn = 2*(Y-ymin)/(ymax-ymin) - 1
    预测时必须一致。
    x_in: torch [1,Nx,Ny,3] with channels [phi, X, Y] in physical
    """
    x = x_in.clone()
    eps = 1e-12
    x_min = float(scales["x_min"]); x_max = float(scales["x_max"])
    y_min = float(scales["y_min"]); y_max = float(scales["y_max"])

    X = x[..., 1]
    Y = x[..., 2]
    Xn = 2.0 * (X - x_min) / (x_max - x_min + eps) - 1.0
    Yn = 2.0 * (Y - y_min) / (y_max - y_min + eps) - 1.0
    x[..., 1] = Xn
    x[..., 2] = Yn
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="某个样本的 npz 路径")
    ap.add_argument("--ckpt", type=str, default="fno2d_openfoam.pth", help="训练好的模型 checkpoint")
    ap.add_argument("--stride", type=int, default=16, help="quiver 抽样步长")
    ap.add_argument("--savefig", type=str, default="", help="如果给路径就保存图，比如 results/pred.png")
    ap.add_argument("--show", action="store_true", help="弹窗显示图")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    ckpt_path = Path(args.ckpt)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    print("✅ Using checkpoint:", ckpt_path.resolve())
    print("✅ Using npz       :", npz_path.resolve())

    # ----- load checkpoint -----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    scales = ckpt["scales"]
    ckpt_args = ckpt.get("args", {})

    width = int(ckpt_args.get("width", 64))
    modes_x = int(ckpt_args.get("modes_x", 32))
    modes_y = int(ckpt_args.get("modes_y", 16))
    n_blocks = int(ckpt_args.get("n_blocks", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model hyperparams:", {"width": width, "modes_x": modes_x, "modes_y": modes_y, "n_blocks": n_blocks})

    model = FNO2D(in_channels=3, out_channels=4, width=width, modes_x=modes_x, modes_y=modes_y, n_blocks=n_blocks).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ----- load npz -----
    data = np.load(npz_path, allow_pickle=True)

    kX = _pick_key(data, ["X"])
    kY = _pick_key(data, ["Y", "Z"])
    kphi = _pick_key(data, ["sdf", "phi", "Phi"])
    if kX is None or kY is None or kphi is None:
        raise KeyError(f"npz keys missing. Found keys={list(data.keys())}")

    X = data[kX].astype(np.float64)
    Y = data[kY].astype(np.float64)
    phi = data[kphi].astype(np.float64)

    if "input" not in data:
        raise KeyError("npz has no 'input' (expected channels [phi,X,Y])")

    # input: (3,Nx,Ny) -> [1,Nx,Ny,3]
    x_in_phys = torch.from_numpy(data["input"]).float().permute(1, 2, 0).unsqueeze(0)  # [1,Nx,Ny,3]
    x_in = normalize_xy_like_training(x_in_phys, scales)

    # TRUE output (optional)
    has_true = "output" in data
    if has_true:
        y_true_phys = torch.from_numpy(data["output"]).float().permute(1, 2, 0).unsqueeze(0)  # [1,Nx,Ny,4]
    else:
        y_true_phys = None

    # ----- predict -----
    with torch.no_grad():
        y_pred_norm = model(x_in.to(device)).cpu()      # [1,Nx,Ny,4] normalized
        y_pred_phys = denormalize_output(y_pred_norm, scales)

    # ----- fluid mask (important!) -----
    fluid_mask = (phi > 0)  # same convention as training (phi>0 is fluid)
    fluid_mask_f = fluid_mask.astype(np.float64)

    # predicted U
    Ux_pred = y_pred_phys[0, ..., 0].numpy()
    Uy_pred = y_pred_phys[0, ..., 1].numpy()
    Umag_pred = np.sqrt(Ux_pred**2 + Uy_pred**2)

    # true U
    if has_true:
        Ux_true = y_true_phys[0, ..., 0].numpy()
        Uy_true = y_true_phys[0, ..., 1].numpy()
        Umag_true = np.sqrt(Ux_true**2 + Uy_true**2)
        Umag_err = Umag_pred - Umag_true
    else:
        Ux_true = Uy_true = Umag_true = Umag_err = None

    # ----- apply mask to visualization (solid -> NaN) -----
    def apply_vis_mask(A):
        A = A.copy()
        A[~fluid_mask] = np.nan
        return A

    Ux_pred_v = apply_vis_mask(Ux_pred)
    Uy_pred_v = apply_vis_mask(Uy_pred)
    Umag_pred_v = apply_vis_mask(Umag_pred)

    if has_true:
        Ux_true_v = apply_vis_mask(Ux_true)
        Uy_true_v = apply_vis_mask(Uy_true)
        Umag_true_v = apply_vis_mask(Umag_true)
        Umag_err_v  = apply_vis_mask(Umag_err)
    else:
        Ux_true_v = Uy_true_v = Umag_true_v = Umag_err_v = None

    # ----- plotting helpers -----
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    def imshow(ax, A, title, vmin=None, vmax=None):
        im = ax.imshow(A.T, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.9)
        ax.contour(X, Y, phi, levels=[0.0], colors="k", linewidths=1.2)

    # consistent color ranges if TRUE exists
    if has_true:
        umin, umax = np.nanmin(Umag_true_v), np.nanmax(Umag_true_v)
        uxmin, uxmax = np.nanmin(Ux_true_v), np.nanmax(Ux_true_v)
        uymin, uymax = np.nanmin(Uy_true_v), np.nanmax(Uy_true_v)
    else:
        umin = umax = uxmin = uxmax = uymin = uymax = None

    # ----- figure 1: phi -----
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4))
    imshow(ax1, phi, "phi(x,y)  (contour=0 is boundary)")
    fig1.tight_layout()

    # ----- figure 2: speed / components -----
    if has_true:
        fig2, axs = plt.subplots(2, 3, figsize=(15, 8))
        imshow(axs[0, 0], Umag_true_v, "|U| TRUE (fluid only)", vmin=umin, vmax=umax)
        imshow(axs[0, 1], Ux_true_v,   "Ux TRUE (fluid only)",  vmin=uxmin, vmax=uxmax)
        imshow(axs[0, 2], Uy_true_v,   "Uy TRUE (fluid only)",  vmin=uymin, vmax=uymax)

        imshow(axs[1, 0], Umag_pred_v, "|U| PRED (same scale)", vmin=umin,  vmax=umax)
        imshow(axs[1, 1], Ux_pred_v,   "Ux PRED (same scale)",  vmin=uxmin, vmax=uxmax)
        imshow(axs[1, 2], Uy_pred_v,   "Uy PRED (same scale)",  vmin=uymin, vmax=uymax)
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(1, 1, figsize=(7, 4))
        imshow(ax3, Umag_err_v, "Speed error: |U|pred - |U|true (fluid only)")
        fig3.tight_layout()
    else:
        fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
        imshow(axs[0], Umag_pred_v, "|U| PRED (fluid only)")
        imshow(axs[1], Ux_pred_v,   "Ux PRED (fluid only)")
        imshow(axs[2], Uy_pred_v,   "Uy PRED (fluid only)")
        fig2.tight_layout()
        fig3 = None

    # ----- figure 4: quiver on predicted speed -----
    s = max(1, int(args.stride))
    fig4, ax4 = plt.subplots(1, 1, figsize=(7, 4))
    imshow(ax4, Umag_pred_v, f"|U| PRED + quiver (stride={s}) (fluid only)", vmin=umin, vmax=umax)

    # quiver: solid -> 0 vectors
    Ux_q = Ux_pred * fluid_mask_f
    Uy_q = Uy_pred * fluid_mask_f

    ax4.quiver(
        X[::s, ::s], Y[::s, ::s],
        Ux_q[::s, ::s], Uy_q[::s, ::s],
        angles="xy", scale_units="xy", scale=None, width=0.0025
    )
    fig4.tight_layout()

    # ----- save/show -----
    if args.savefig:
        out = Path(args.savefig)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out.with_name(out.stem + "_phi.png"), dpi=200)
        fig2.savefig(out.with_name(out.stem + "_U.png"), dpi=200)
        fig4.savefig(out.with_name(out.stem + "_quiver.png"), dpi=200)
        if has_true and fig3 is not None:
            fig3.savefig(out.with_name(out.stem + "_err.png"), dpi=200)
        print("Saved figures to:", out.parent.resolve())

    if args.show or (not args.savefig):
        plt.show()


if __name__ == "__main__":
    main()

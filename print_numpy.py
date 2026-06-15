import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Path setting
# =========================
base_dir = "./result_denseball/10DoF_analytic_nominalP/no_initP_k_un_vn_SOD_SDD_export"
save_dir = os.path.join(base_dir, "motion_plots_intrinsic_extrinsic")
os.makedirs(save_dir, exist_ok=True)

# =========================
# Plot style
# =========================
figsize = (8, 5)
label_fontsize = 20
tick_fontsize = 16
legend_fontsize = 16
linewidth = 2.0
dpi = 300

# =========================
# Load parameters
# =========================
ts = np.load(os.path.join(base_dir, "ts_mm.npy"))        # (V, 3)
tp = np.load(os.path.join(base_dir, "tp_mm.npy"))        # (V, 3)
rot = np.load(os.path.join(base_dir, "rot_deg.npy"))     # (V, 3)
skew = np.load(os.path.join(base_dir, "skew.npy"))       # (V, 3) or (V, 1)

V = ts.shape[0]
x = np.arange(V)

# =========================
# Intrinsic parameters: cu, cv, f
# =========================
delta_cu = ts[:, 0]
delta_cv = ts[:, 1]
delta_f = ts[:, 2]

plt.figure(figsize=figsize)
plt.plot(x, delta_cu, linewidth=linewidth, label=r"$\Delta c_u$")
plt.plot(x, delta_cv, linewidth=linewidth, label=r"$\Delta c_v$")
plt.plot(x, delta_f, linewidth=linewidth, label=r"$\Delta f$")
plt.xlabel("view index", fontsize=label_fontsize)
plt.ylabel("intrinsic parameters (mm)", fontsize=label_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "intrinsic_parameters_cu_cv_f.png"), dpi=dpi)
plt.close()

# =========================
# Intrinsic parameter: skew
# =========================
if skew.ndim == 2:
    delta_gamma = skew[:, 0]
else:
    delta_gamma = skew

plt.figure(figsize=figsize)
plt.plot(x, delta_gamma, linewidth=linewidth, label=r"$\Delta \gamma$")
plt.xlabel("view index", fontsize=label_fontsize)
plt.ylabel(r"intrinsic parameter $\Delta \gamma$ (deg)", fontsize=label_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "intrinsic_parameter_skew_gamma.png"), dpi=dpi)
plt.close()

# =========================
# Extrinsic parameters: translation
# =========================
delta_tx = tp[:, 0]
delta_ty = tp[:, 1]
delta_tz = tp[:, 2]

plt.figure(figsize=figsize)
plt.plot(x, delta_tx, linewidth=linewidth, label=r"$\Delta t_x$")
plt.plot(x, delta_ty, linewidth=linewidth, label=r"$\Delta t_y$")
plt.plot(x, delta_tz, linewidth=linewidth, label=r"$\Delta t_z$")
plt.xlabel("view index", fontsize=label_fontsize)
plt.ylabel("extrinsic parameters (mm)", fontsize=label_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "extrinsic_parameters_translation.png"), dpi=dpi)
plt.close()

# =========================
# Extrinsic parameters: rotation
# =========================
delta_rx = rot[:, 0]
delta_ry = rot[:, 1]
delta_rz = rot[:, 2]

plt.figure(figsize=figsize)
plt.plot(x, delta_rx, linewidth=linewidth, label=r"$\Delta r_x$")
plt.plot(x, delta_ry, linewidth=linewidth, label=r"$\Delta r_y$")
plt.plot(x, delta_rz, linewidth=linewidth, label=r"$\Delta r_z$")
plt.xlabel("view index", fontsize=label_fontsize)
plt.ylabel("extrinsic parameters (deg)", fontsize=label_fontsize)
plt.tick_params(axis="both", labelsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "extrinsic_parameters_rotation.png"), dpi=dpi)
plt.close()

print(f"Saved plots to: {save_dir}")
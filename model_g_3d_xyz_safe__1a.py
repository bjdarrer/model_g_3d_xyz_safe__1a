"""
Model G Particle 3D (proper x–y–z) — SAFE + RESUMABLE

- Written by Brendan Darrer aided by ChatGPT5.1 date: 21st November 2025 16:25 GMT - updated ----
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb
- with ChatGPT5.1 writing it and Brendan guiding it to produce a clean code.

- 3D upgrade of: model_g_2d_xy_safe__2a.py
- Full 3D Cartesian grid (x, y, z); no symmetry assumptions.
- No vortical motion (advection velocity v = 0).
- Segmented integration with on-disk checkpoints; safe to interrupt/resume.
- Incremental frame rendering; MP4 assembled at the end.
- Defaults intended for a mid-range desktop; adjust nx, ny, nz as needed.

Install:
    pip install numpy scipy matplotlib imageio imageio[ffmpeg]

Run:
    python3 model_g_3d_xyz_safe__1a.py

Outputs (under out_model_g_3d_xyz_safe__1a/):
    - frames/               PNG frames (mid-z slices)
    - model_g_3d_xyz_safe__1a.mp4
    - final_snapshot.png    (mid-z pY heatmap at final time)
    - checkpoint_3d.npz     (auto-resume)

Notes:
- Equations follow eqs13 with eqs17 parameters.
- Boundary conditions: Dirichlet pG=pX=pY=0 on all sides of the 3D box.
- Forcing chi(x,y,z,t): sum of Gaussian seeds in 3D * temporal bell around Tseed.
"""

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio

# ---------------- Configuration ----------------
# Domain (Cartesian)
Lx = 20.0   # width  in x
Ly = 20.0   # height in y
Lz = 20.0   # depth  in z

# Time
Tfinal = 40.0          # total simulated time
segment_dt = 0.5       # integrate and checkpoint every 0.5 time unit

# Grid / solver
# Choose smaller grid than 2D version to keep 3D manageable
nx = 96
ny = 96
nz = 96

max_step = 0.01        # soft cap for RK23 internal step
atol = 1e-6
rtol = 1e-6

# Animation
nt_anim = 480          # total frames across [0, Tfinal]

# Output paths
run_name   = "model_g_3d_xyz_safe__1a"
out_dir    = f"out_{run_name}"
frames_dir = os.path.join(out_dir, "frames")
ckpt_path  = os.path.join(out_dir, "checkpoint_3d.npz")
mp4_path   = os.path.join(out_dir, f"{run_name}.mp4")
final_png  = os.path.join(out_dir, "final_snapshot.png")

# Ensure directories exist
os.makedirs(frames_dir, exist_ok=True)

# ---------------- Parameters (eqs17) ----------------
params = {
    'a': 14.0,
    'b': 29.0,
    'dx': 1.0,
    'dy': 12.0,
    'p': 1.0,
    'q': 1.0,
    'g': 0.1,
    's': 0.0,
    'u': 0.0,
    # 3D advection vector (vx, vy, vz); eqs17 -> zero (no vortical motion)
    'v': (0.0, 0.0, 0.0),
    'w': 0.0,
}

# ---------------- Forcing chi(x,y,z,t) (Gaussian seeds) ----------------
def bell(s, x):
    return np.exp(- (x/s)**2 / 2.0)

Tseed = 10.0
# 3D seed centers: (xc, yc, zc)
seed_centers = [ (0.0, 0.0, 0.0) ]
seed_sigma_space = 2.0          # spatial width of each seed
seed_sigma_time  = 3.0          # temporal width

# ---------------- Grid setup ----------------
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
z = np.linspace(-Lz/2, Lz/2, nz)

dx_space = x[1] - x[0]
dy_space = y[1] - y[0]
dz_space = z[1] - z[0]

# Coordinate arrays via broadcasting for forcing
X3 = x[None, None, :]  # shape (1, 1, nx)
Y3 = y[None, :, None]  # shape (1, ny, 1)
Z3 = z[:, None, None]  # shape (nz, 1, 1)

# --- Auto color (z) limit setup ---
auto_zlim = True
zlim_global = [-1.0, 1.0]  # fallback if auto_zlim=False

# ---------------- Homogeneous steady state ----------------
a = params['a']; b = params['b']
p_par = params['p']; q_par = params['q']
g_par = params['g']; s_par = params['s']
u_par = params['u']; w_par = params['w']

G0 = (a + g_par*w_par) / (q_par - g_par*p_par)
X0 = (p_par*a + q_par*w_par) / (q_par - g_par*p_par)
Y0 = ((s_par*X0**2 + b) * X0 / (X0**2 + u_par)) if (X0**2 + u_par) != 0 else 0.0
print(f"Homogeneous state: G0={G0:.6g}, X0={X0:.6g}, Y0={Y0:.6g}")

# ---------------- Operators (7-point Laplacian, central gradients) ------------
def laplacian_3d(u):
    """
    Seven-point Laplacian with Dirichlet boundary (zeros at edges).
    u shape: (nz, ny, nx)  -> axes: z, y, x
    """
    d2x = (np.roll(u, -1, axis=2) - 2*u + np.roll(u, 1, axis=2)) / (dx_space**2)
    d2y = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / (dy_space**2)
    d2z = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / (dz_space**2)
    out = d2x + d2y + d2z

    # Dirichlet boundaries -> enforce zeros at edges
    out[:, :,  0] = 0.0
    out[:, :, -1] = 0.0
    out[:,  0, :] = 0.0
    out[:, -1, :] = 0.0
    out[ 0, :, :] = 0.0
    out[-1, :, :] = 0.0
    return out

def gradx(u):
    gx = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2*dx_space)
    gx[:, :,  0] = 0.0
    gx[:, :, -1] = 0.0
    return gx

def grady(u):
    gy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy_space)
    gy[:,  0, :] = 0.0
    gy[:, -1, :] = 0.0
    return gy

def gradz(u):
    gz = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dz_space)
    gz[ 0, :, :] = 0.0
    gz[-1, :, :] = 0.0
    return gz

# ---------------- Forcing field ----------------
def chi_xyz_t(t):
    if not seed_centers:
        return np.zeros((nz, ny, nx))

    spatial = np.zeros((nz, ny, nx))
    for center in seed_centers:
        if len(center) == 3:
            xc, yc, zc = center
        else:
            # fallback: 2D-style center, zc=0
            xc, yc = center
            zc = 0.0
        spatial += np.exp(
            -((X3 - xc)**2 + (Y3 - yc)**2 + (Z3 - zc)**2)
            / (2 * seed_sigma_space**2)
        )
    return -spatial * bell(seed_sigma_time, t - Tseed)

# ---------------- Packing helpers ----------------
N = nx * ny * nz

def pack(pG, pX, pY):
    return np.concatenate([pG.ravel(), pX.ravel(), pY.ravel()])

def unpack(y_flat):
    pG = y_flat[0:N].reshape(nz, ny, nx)
    pX = y_flat[N:2*N].reshape(nz, ny, nx)
    pY = y_flat[2*N:3*N].reshape(nz, ny, nx)
    return pG, pX, pY

# ---------------- RHS for solve_ivp (3D) -------------------------------------
VX, VY, VZ = params['v']

def rhs(t, y_flat):
    pG, pX, pY = unpack(y_flat)

    lapG = laplacian_3d(pG)
    lapX = laplacian_3d(pX)
    lapY = laplacian_3d(pY)

    # Advection terms (currently zero, but kept for completeness)
    advG = VX*gradx(pG) + VY*grady(pG) + VZ*gradz(pG)
    advX = VX*gradx(pX) + VY*grady(pX) + VZ*gradz(pX)
    advY = VX*gradx(pY) + VY*grady(pY) + VZ*gradz(pY)

    forcing = chi_xyz_t(t)

    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s  = s_par * (Xtot**3 - X0**3)
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)

    dpGdt = lapG - advG - q_par * pG + g_par * pX
    dpXdt = params['dx']*lapX - advX + p_par * pG - (1.0 + b) * pX + u_par * pY - nonlinear_s + nonlinear_xy + forcing
    dpYdt = params['dy']*lapY - advY + b * pX - u_par * pY + (-nonlinear_xy + nonlinear_s)

    # Dirichlet boundaries: keep edges at zero by clamping time derivative
    for arr in (dpGdt, dpXdt, dpYdt):
        arr[:, :,  0] = 0.0
        arr[:, :, -1] = 0.0
        arr[:,  0, :] = 0.0
        arr[:, -1, :] = 0.0
        arr[ 0, :, :] = 0.0
        arr[-1, :, :] = 0.0

    return pack(dpGdt, dpXdt, dpYdt)

# ---------------- Plot / frame rendering -------------------------------------
def render_frame(Yvec, t, fpath):
    pG, pX, pY = unpack(Yvec)
    # Take mid-z slice (closest to z=0)
    k_mid = np.argmin(np.abs(z - 0.0))

    pG_slice = pG[k_mid, :, :]
    pX_slice = pX[k_mid, :, :]
    pY_slice = pY[k_mid, :, :]

    plt.figure(figsize=(8, 7))
    plt.suptitle(f"Model G 3D (mid-z slice) — t={t:.3f}, z≈{z[k_mid]:.2f}")

    ax = plt.subplot(2, 2, 1)
    im = ax.imshow(
        pY_slice,
        origin='lower',
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=zlim_global[0],
        vmax=zlim_global[1],
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('pY (Y) mid-z')

    ax = plt.subplot(2, 2, 2)
    im = ax.imshow(
        pG_slice,
        origin='lower',
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=zlim_global[0],
        vmax=zlim_global[1],
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('pG (G) mid-z')

    ax = plt.subplot(2, 2, 3)
    im = ax.imshow(
        pX_slice / 10.0,
        origin='lower',
        extent=[x[0], x[-1], y[0], y[-1]],
        vmin=zlim_global[0],
        vmax=zlim_global[1],
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('pX/10 (X scaled) mid-z')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fpath, dpi=120)
    plt.close()

# ---------------- Checkpoint logic -------------------------------------------
def save_ckpt(t_curr, y_curr, next_frame_idx, frames_done):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp = ckpt_path + ".tmp"
    try:
        np.savez_compressed(
            tmp,
            t_curr=t_curr,
            y_curr=y_curr,
            next_frame_idx=next_frame_idx,
            frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
        )
    except Exception as e:
        print(f"[ERROR] Could not write checkpoint tmp file: {e}")
        return
    if os.path.exists(tmp):
        os.replace(tmp, ckpt_path)
    else:
        print(f"[WARN] Temporary checkpoint {tmp} missing, skipping rename.")

def load_ckpt():
    if not os.path.exists(ckpt_path):
        return None
    data = np.load(ckpt_path, allow_pickle=True)
    frames_done = set(int(v) for v in np.array(data['frames_done']).ravel().tolist())
    return {
        't_curr': float(data['t_curr']),
        'y_curr': data['y_curr'],
        'next_frame_idx': int(data['next_frame_idx']),
        'frames_done': frames_done,
    }

# ---------------- Segmented integration (resumable) --------------------------
def estimate_z_limits(y_init):
    """Estimate global z-limits (color scale) for consistent heatmaps."""
    pG, pX, pY = unpack(y_init)
    vals = np.concatenate([pG.ravel(), (pX/10.0).ravel(), pY.ravel()])
    zmin, zmax = np.percentile(vals, [1, 99])  # ignore outliers
    margin = 0.1 * (zmax - zmin)
    # Handle pathological case (all zeros)
    if zmax - zmin < 1e-14:
        return [-1.0, 1.0]
    return [zmin - margin, zmax + margin]

def main():
    frame_times = np.linspace(0.0, Tfinal, nt_anim)

    # initial condition
    y0 = np.zeros(3 * N)

    # resume or start
    ck = load_ckpt()
    if ck is None:
        t_curr = 0.0
        y_curr = y0
        next_frame_idx = 0
        frames_done = set()
        print("[Start] Fresh run")
    else:
        t_curr = ck['t_curr']
        y_curr = ck['y_curr']
        next_frame_idx = ck['next_frame_idx']
        frames_done = ck['frames_done']
        print(f"[Resume] t={t_curr:.3f}, next_frame={next_frame_idx}/{nt_anim}, frames_done={len(frames_done)}")

    # --- Auto z-limit estimation ---
    global zlim_global
    if auto_zlim:
        zlim_global = estimate_z_limits(y_curr)
        print(f"[Auto] Color limits fixed to {zlim_global}")

    # Render frames already due at t=0 or current time
    while next_frame_idx < nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
        tframe = frame_times[next_frame_idx]
        fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
        if next_frame_idx not in frames_done:
            render_frame(y_curr, tframe, fpath)
            frames_done.add(next_frame_idx)
        next_frame_idx += 1
        save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

    t_start_wall = time.time()

    # segmented solve
    while t_curr < Tfinal - 1e-12:
        t_seg_end = min(Tfinal, t_curr + segment_dt)
        print(f"[Integrate] {t_curr:.3f} -> {t_seg_end:.3f}  (segment_dt={segment_dt})")

        # t_eval includes any frame times within this segment
        t_eval = frame_times[(frame_times > t_curr + 1e-12) & (frame_times <= t_seg_end + 1e-12)]
        seg_sol = solve_ivp(
            rhs,
            (t_curr, t_seg_end),
            y_curr,
            method='RK23',
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            t_eval=t_eval if t_eval.size > 0 else None,
        )
        if seg_sol.status < 0:
            print("[WARN] Segment failure:", seg_sol.message)

        # render any requested frames for this segment
        if seg_sol.t.size > 0:
            for k, tframe in enumerate(seg_sol.t):
                fidx = np.searchsorted(frame_times, tframe)
                if fidx < nt_anim and abs(frame_times[fidx] - tframe) < 1e-9:
                    if fidx not in frames_done:
                        render_frame(seg_sol.y[:, k], tframe,
                                     os.path.join(frames_dir, f"frame_{fidx:04d}.png"))
                        frames_done.add(fidx)
                        save_ckpt(tframe, seg_sol.y[:, k], fidx+1, frames_done)

        # advance to end of segment
        y_curr = seg_sol.y[:, -1] if seg_sol.y.ndim == 2 else seg_sol.y
        t_curr = seg_sol.t[-1] if seg_sol.t.size > 0 else t_seg_end

        # catch-up rendering for any frames whose times are now <= t_curr
        while next_frame_idx < nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
            tframe = frame_times[next_frame_idx]
            fpath = os.path.join(frames_dir, f"frame_{next_frame_idx:04d}.png")
            if next_frame_idx not in frames_done:
                render_frame(y_curr, tframe, fpath)
                frames_done.add(next_frame_idx)
            next_frame_idx += 1
            save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

        # progress
        elapsed = time.time() - t_start_wall
        print(f"  -> Reached t={t_curr:.3f}/{Tfinal}, frames={len(frames_done)}/{nt_anim}, wall={elapsed:.1f}s")

        # rolling snapshot & checkpoint
        render_frame(y_curr, t_curr, final_png)
        save_ckpt(t_curr, y_curr, next_frame_idx, frames_done)

    # assemble MP4 at end
    print("[Video] Writing MP4:", mp4_path)
    with imageio.get_writer(mp4_path, fps=max(8, int(nt_anim / max(1, Tfinal/2)))) as writer:
        for i in range(nt_anim):
            f = os.path.join(frames_dir, f"frame_{i:04d}.png")
            img = imageio.imread(f)
            writer.append_data(img)
    print("[Done] MP4 saved.")
    print("Final snapshot:", final_png)

if __name__ == '__main__':
    main()


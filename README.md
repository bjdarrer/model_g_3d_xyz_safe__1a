# model_g_3d_xyz_safe__1a

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

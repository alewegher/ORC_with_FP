# ORC_with_FP

**Learning a neural-network terminal cost for MPC**, final project for *Optimization-based Robot
Control* (ORC) @ UniTN. The `01_actuators/`, `reactive_control/`, `optimal_control/`, `RL/`,
`lab_disi/` and `utils/` folders are the course's weekly lab material, originally authored by
**Andrea Del Prete** (course instructor; see the `andreadelprete/orc24` Docker image this repo's
image extends). Everything under [`Final_Project/`](Final_Project/) is this project's own work,
built on top of that base — **start there**.

## Final Project

Goal: train a neural network to approximate the MPC terminal cost (the cost-to-go / value
function) of a pendulum, then use it so a **short-horizon** MPC behaves like a much more
expensive **long-horizon** one — cutting per-step solve time without giving up trajectory
quality.

Studied on two systems, mirrored under [`Final_Project/Single_Pend/`](Final_Project/Single_Pend/)
and [`Final_Project/Double_Pend/`](Final_Project/Double_Pend/), each with the same four-stage
pipeline:

| Stage | Single pendulum | Double pendulum | What it does |
|---|---|---|---|
| 1. Train | `SinPendTrain.py` | `Double_pend_training.py` | Solves many OCPs from random initial states (`OCP_solve.py`), trains a feedforward NN (`neural_network.py`) to predict the optimal cost-to-go, saves `nn_cost_pred.pt` |
| 2. Test | `SinPendTest.py` | `Double_pend_Test.py` | Validates the trained NN against true OCP costs (scatter/residual plots) |
| 3. Open-loop comparison | `SinPendComp.py` / `SinPendComp_multiprocess.py` | `Double_pend_comp.py` / `Double_pend_comp_multi.py` | Single-shot OCP cost comparison of 3 formulations over K random initial states: short horizon **M** (no terminal cost), long horizon **M+N** (no terminal cost, reference), short horizon **M + NN terminal cost** |
| 4. Closed-loop MPC | `SinPend_MPC.py` | `Double_pend_MPC.py` | Actually drives the pendulum in receding-horizon fashion (`MPC_solve.py`) for the same 3 formulations from one fixed initial state, plots q/dq/torque trajectories |

Headline result: the trained network predicts the terminal cost accurately within its training
domain for both systems, letting formulation 3 (short horizon + NN terminal cost) match
formulation 2's (long horizon) trajectory quality at formulation 1's computational cost — in
aggregate, open-loop terms. The double pendulum's higher-dimensional, more nonlinear cost
landscape makes the learning task harder (visible in the residuals plot, worse near the training
domain's boundaries); the closed-loop experiments also surface a case where matching *aggregate*
open-loop cost does not guarantee matching a *specific* trajectory — see the report's Discussion
section for the full analysis and possible improvements.

Other files: `double_pend_dynamics.mw` (Maple worksheet, double-pendulum equations of motion
derived symbolically via Euler–Lagrange), `neural_network.py` (shared `NeuralNetwork` class).

**Reports**, in [`Final_Project/report/`](Final_Project/report/):
- [`Final_Report_ORC.pdf`](Final_Project/report/Final_Report_ORC.pdf) — full technical report.
- [`Project_outline.pdf`](Final_Project/report/Project_outline.pdf) — short, non-technical summary for sharing outside academia (e.g. LinkedIn).

Rebuild both from LaTeX source (refreshes figures from `Final_Project/saved_images/` first):
```bash
cd Final_Project/report && ./build.sh
```

## Running it — Docker (recommended)

The final project depends on `l4casadi` (compiles a trained PyTorch network into a CasADi
`Function`) and PyTorch/CUDA, which are easiest to get via the provided Docker image.

**Build the image** (from the repo root, so the `l4casadi` git submodule is in build context):
```bash
git submodule update --init          # fetches l4casadi/
docker build -f docker/Dockerfile -t orc24-gpu:latest .
```
This extends the course image `andreadelprete/orc24:v1` with a CUDA 12.1 toolkit, GPU PyTorch,
and `l4casadi` built from source. Requires an NVIDIA GPU, driver ≥ 530, and
`nvidia-container-toolkit` on the host — **Linux only** in practice: `--gpus all` needs the
NVIDIA Container Toolkit, which doesn't exist for Docker Desktop on macOS, and X11 forwarding
(below) would additionally need XQuartz + socat set up by hand. On macOS, run without `--gpus
all` (CPU-only, slower) and expect to adapt the X11 lines yourself.

**Run it** (Linux, with an NVIDIA GPU):
```bash
docker/run.sh          # ephemeral container, removed on exit
docker/run.sh keep     # persistent named container, reused (docker start -ai) on later calls
```
`run.sh` grants the container full GPU + X11 access (`--gpus all`, `--privileged`, X11 socket +
Xauthority mount) so plots and GUI windows open normally, and mounts this repo twice: once at
`/home/student/shared/ORC_with_FP` (the working directory) and once aliased at
`/home/student/shared/orc`, with `PYTHONPATH=/home/student/shared` — every script in this repo
imports itself as `orc.Final_Project...`, so the alias makes that resolve regardless of what the
checkout folder is actually named. It also maps port `7000` for the meshcat viewer.

Once inside the container, run any script module-style, e.g.:
```bash
python3 -m orc.Final_Project.Single_Pend.SinPend_MPC
```
Stage 1 (training) must run before stages 2–4, since it produces `nn_cost_pred.pt`. Figures land
in `Final_Project/saved_images/`.

## Native install (course lab material, no Docker)

Only needed for the `01_actuators/` / `reactive_control/` / `optimal_control/` / `RL/` lab
scripts — the final project's PyTorch/CUDA/`l4casadi` stack above is not covered by this path.
Follow these on Ubuntu 20.04 or 22.04 (native or VM):

```bash
sudo apt install terminator python3-numpy python3-scipy python3-matplotlib spyder3 curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
```

On Ubuntu 20.04 install these packages (on 22.04, swap `py38` for `py310`):
```bash
sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

Then add to `~/.bashrc` (or `~/.zshrc`):
```bash
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
```
where `<folder_containing_orc>` is the folder *containing* this checkout (mind the Python
version, e.g. `python3.8`, matching what robotpkg installed for your Ubuntu version).

For the meshcat viewer: `pip install meshcat`.

## License

GPLv3 (see [`LICENSE`](LICENSE)), inherited from the original course repository.

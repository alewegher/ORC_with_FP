# ORC_with_FP

Course repository for **"Learning and Optimization for Robot Control"** (Advanced Optimization-based Robot Control) @ UniTN. Contains the weekly lab material (`01_actuators/`, `reactive_control/`, `optimal_control/`, `RL/`, `utils/`) plus a final project, [`Final_Project/`](Final_Project/), on learning a neural-network terminal cost for MPC on a single and a double pendulum.

## Installation instructions for native Ubuntu machine

Follow these instructions if you have a computer with an Ubuntu Operating System, or you already have an Ubuntu virtual machine that you would rather use (e.g., to save space). Acceptable versions of Ubuntu are 20.04 or 22.04.

Open a terminal and execute the following commands:
```bash
sudo apt install terminator python3-numpy python3-scipy python3-matplotlib spyder3 curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
```

On Ubuntu 20.04 install these packages:
```bash
sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

For other versions of the Ubuntu OS you might need to use a different version of the python packages (e.g., on Ubuntu 22.04 you need to use py310 instead of py38). Configure the environment variables by adding the following lines to your `~/.bashrc` (or `~/.zshrc` if you use Zsh) file:
```bash
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
```
where `<folder_containing_orc>` is the folder containing the "orc" folder, which in turn contains all the python code of this class. Pay attention to the python version (e.g. `python3.8`) in the name of the python folder, which may be different from the one you have on your machine, depending on which OS version you have.

For using the meshcat viewer you need to install it with:
```bash
pip install meshcat
```

## Docker (recommended)

The final project additionally depends on `l4casadi` (compiles a trained PyTorch network into a CasADi `Function`) and PyTorch/CUDA, which are easiest to get via the provided Docker image rather than the native install above.

**Build the image** (from the repo root, so the `l4casadi` git submodule is in build context):
```bash
git submodule update --init          # fetches l4casadi/
docker build -f docker/Dockerfile -t orc24-gpu:latest .
```
This extends the course image `andreadelprete/orc24:v1` with a CUDA 12.1 toolkit, GPU PyTorch, and `l4casadi` built from source. Requires an NVIDIA GPU, driver ≥ 530, and `nvidia-container-toolkit` on the host.

**Run it**:
```bash
docker/run.sh          # ephemeral container, removed on exit
docker/run.sh keep      # persistent named container, reused on later calls
```
`run.sh` mounts this repo twice: once at `/home/student/shared/ORC_with_FP` (the working directory) and once aliased at `/home/student/shared/orc`, with `PYTHONPATH=/home/student/shared` — every script in this repo imports itself as `orc.Final_Project...`, so the alias makes that resolve regardless of what the checkout folder is actually named. It also forwards X11/XWayland so GUI windows (matplotlib, meshcat) work, and maps port `7000` for the meshcat viewer.

Once inside the container, run any script module-style, e.g.:
```bash
python3 -m orc.Final_Project.Single_Pend.SinPend_MPC
```

## Final Project

Goal: train a neural network to approximate the MPC terminal cost (cost-to-go / value function) of a pendulum, then use it so a **short-horizon** MPC behaves like a much more expensive **long-horizon** one. Studied on two systems, mirrored under [`Final_Project/Single_Pend/`](Final_Project/Single_Pend/) and [`Final_Project/Double_Pend/`](Final_Project/Double_Pend/):

| Stage | Single pendulum | Double pendulum | What it does |
|---|---|---|---|
| 1. Train | `SinPendTrain.py` | `Double_pend_training.py` | Solves many OCPs from random initial states (`OCP_solve.py`), trains a feedforward NN (`neural_network.py`) to predict the optimal cost-to-go, saves `nn_cost_pred.pt` |
| 2. Test | `SinPendTest.py` | `Double_pend_Test.py` | Validates the trained NN against true OCP costs (scatter/residual plots) |
| 3. Open-loop comparison | `SinPendComp.py` / `SinPendComp_multiprocess.py` | `Double_pend_comp.py` / `Double_pend_comp_multi.py` | Single-shot OCP cost comparison of 3 formulations over K random initial states: short horizon **M** (no terminal cost), long horizon **M+N** (no terminal cost, reference), short horizon **M + NN terminal cost** |
| 4. Closed-loop MPC | `SinPend_MPC.py` | `Double_pend_MPC.py` | Actually drives the pendulum in receding-horizon fashion (`MPC_solve.py`) for the same 3 formulations from one fixed initial state, plots q/dq/torque trajectories |

Run any stage from inside the Docker container as `python3 -m orc.Final_Project.<Single_Pend|Double_Pend>.<script>` (stage 1 must run before 2-4, since it produces `nn_cost_pred.pt`). Figures land in `Final_Project/saved_images/`.

**Report**: [`Final_Project/ORC_Final_project_report.pdf`](Final_Project/ORC_Final_project_report.pdf), built from LaTeX source in [`Final_Project/report/`](Final_Project/report/). Rebuild it with:
```bash
cd Final_Project/report && ./build.sh
```
A shorter, non-technical version for sharing outside academia is also built there as `linkedin.pdf`.

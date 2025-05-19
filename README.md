# GraNT

[Chen Zhang*](https://chen2hang.github.io), [Weixin Bu*](https://brysonwx.github.io), [Zeyi Ren](https://zeyi0827.github.io/zeyi.github.io), [Zhengwu Liu](https://www.eee.hku.hk/people/zhengwu-liu), [Yik-Chung Wu](https://www.eee.hku.hk/~ycwu), [Ngai Wong](https://www.eee.hku.hk/~nwong)

[[`Paper`](https://chen2hang.github.io/_publications/nonparametric_teaching_for_graph_proerty_learners/ICML_2025_Paper.pdf)] | [[`Project Page`](https://chen2hang.github.io/_publications/nonparametric_teaching_for_graph_proerty_learners/grant.html)]

This is the official PyTorch implementation of the paper: **[Nonparametric Teaching for Graph Property Learners](https://chen2hang.github.io/_publications/nonparametric_teaching_for_graph_proerty_learners/ICML_2025_Paper.pdf)** (ICML'2025).

# Guide for the start-up

This guide outlines the steps to start up GraNT, e.g.,
- Create and activate the Python virtual environment:
  ```shell
  conda create --name grant python==3.8
  conda activate grant
  ```
- Install PyTorch and PyTorch Geometric for both **NVIDIA GPUs** and **AMD GPUs**;
- Common setup, install some useful Python packages;
- Run the code.

## Table of Contents
1. [For NVIDIA GPUs](#for-nvidia-gpus)
2. [For AMD GPUs](#for-amd-gpus)
3. [Common Setup](#common-setup)
4. [Run the Code](#run-the-code)

---

## For NVIDIA GPUs

**NVIDIA Geforce RTX 3090 (24G)**

### 1. Install PyTorch

- First, check your CUDA version:
  ```shell
    nvcc --version
    nvidia-smi
  ```
- Based on the CUDA version (replace ${CUDA_VERSION} with the appropriate version), install PyTorch:
  ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
  ```
  NOTE: replace ${CUDA_VERSION} with the actual version (e.g., cu124, cu113, cu102, etc.).

### 2. Install PyTorch Geometric

- Follow the official installation guide for PyTorch Geometric:

  [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

- You can also install PyTorch Geometric using the following commands:
  ```shell
     pip install torch_geometric
  ```

- Then install the necessary dependencies:
  ```shell
     pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${PyTorch_VERSION}+${CUDA_VERSION}.html
  ```
  Replace ${PyTorch_VERSION} and ${CUDA_VERSION} with the appropriate versions. If encountering an error while installing `pyg_lib`, you can skip it.

---

## For AMD GPUs

**AMD Instinct MI210 (64GB)**

### 1. Install PyTorch

- Follow the official instructions to install PyTorch on AMD GPUs:

  [PyTorch Installation on ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html)

- Check your ROCm version:
  ```shell
     rocminfo
  ```

- To enable Docker for the current user (assuming the username is `tony`), execute the following commands as an administrator:
  ```shell
     usermod -aG docker tony  # Add user to docker group
     snap start docker        # Start Docker if installed via Snap
  ```

- For the user `tony`, run the following to pull and run the Docker image:
  ```shell
     docker pull rocm/pytorch:latest
     docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     --device=/dev/kfd --device=/dev/dri --group-add video \
     --ipc=host --shm-size 8G rocm/pytorch:latest
  ```
  NOTE: Adjust the `--shm-size 8G` according to your system's available shared memory (check using `df -h /dev/shm`)

- Once inside the Docker container, check the installed PyTorch version:
  ```shell
     pip list | grep torch
  ```
  
- If the installed PyTorch version is not compatible with your ROCm version, uninstall the existing PyTorch and reinstall the appropriate version:
  ```shell
     pip uninstall torch torchvision torchaudio
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
  ```
  
### 2. Install PyTorch Geometric

- Follow the official instructions and visit the external repository for PyTorch Geometric installation on ROCm:

  [PyG Installation on ROCm](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

- Download the appropriate version from the release page and install it inside the Docker container.

- Check your Docker container's GLIBC version to ensure compatibility:
  ```shell
     ldd --version
  ```
  
- To copy datasets to the Docker container, use the `docker cp` command:
  ```shell
     docker ps -a  # Find the container ID
     docker cp datasets/ ${container_id}:/root
  ```

---

## Common Setup

```shell
   pip install -r requirements.txt
```

---

## Run the Code

- modify the parameters for the dataset in `config.yaml` / `config_amd.yaml`;
- modify the `evaluator = Evaluator('ogbg-molhiv')` in `utils.py` if needed;
- [config your wandb](https://docs.wandb.ai/tutorials);
- then you can run the code like:
  ```shell
     python train.py --dataset QM9 --config_file /home/ubuntu/codes/GraNT/config.yaml
  ```

---

## Contact Us
Please feel free to contact us: [Weixin Bu](bwxhhjy@gmail.com) or [Chen Zhang](u3567831@connect.hku.hk) if you have any questions while starting up GraNT.

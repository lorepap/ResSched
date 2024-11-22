# Distributed Training Scheduler: Testbed Experiments

This repository contains the implementation and evaluation code for a distributed training scheduler based on Muri. The experiments demonstrate how efficient resource interleaving and scheduling policies can optimize deep learning workloads across a multi-node cluster.

---

## Contents

### `cluster_exp/`
This folder contains the core implementation and supporting scripts for running testbed experiments:

- **`cluster_spec/`**: Configuration files defining cluster details such as the number of nodes and GPUs per node.
- **`runtime/`**: gRPC runtime implementations for components like scheduler, trainer, master, and worker.
- **`trace-data/`**: Traces used for testbed evaluations.
- **`workloads/`**: Deep learning models and workloads evaluated in the experiments.
- **`calc.py`**: Computes metrics such as average Job Completion Time (JCT), makespan, and 99th percentile JCT.
- **`cluster.py`, `switch.py`, `node.py`**: Cluster and network simulation implementations.
- **`jobs.py`, `model.py`**: Define job parameters and deep learning models.
- **`flags.py`**: Argument configuration utility.
- **`log.py`, `utils.py`**: Auxiliary functions for logging and utility operations.
- **`matching.py`**: Implements the matching algorithm for multi-resource interleaving.
- **`run.py`**: Execution of scheduling policies.
- **`controller.py`, `scheduler.py`, `trainer.py`, `worker.py`, `task.py`**: Scheduler logic and component implementation.
- **`Makefile`**: Automates the preparation of the gRPC runtime.

---

## Setting Up the Environment

### Step 1: Configure Cluster Interconnect
Ensure all cluster nodes are properly connected and reachable.

### Step 2: Create and Activate Conda Environment
```bash
conda create -n scheduler_env python=3.8
conda activate scheduler_env


### Step 3: install Open MPI
[Install Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build) or other MPI implementation.

# Install gRPC
python -m pip install grpcio grpcio-tools

# Prepare gRPC runtime
cd <repo>/cluster_exp
make rpc

# Install other dependencies
conda install numpy
conda install -c conda-forge cvxpy
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
HOROVOD_GPU_OPERATIONS=NCCL python -m pip install horovod

# NLP dependencies
conda install -c huggingface transformers

# RL-specific dependencies
python -m pip install -r <repo>/cluster_exp/workloads/requirements.txt
```

### Step 5: prepare datasets (for testbed experiment)
- [Imagenet-1k](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) for CV models.
- [Wikitext](https://huggingface.co/datasets/wikitext) for NLP models.
Store these datsets in ```<repo>/cluster_exp/datasets/```

<!-- # 2. Reproduce testbed results (for SIGCOMM'22 artifact evaluation)
- ```cd <repo>/cluster_exp```
- Table 3&4, Figure 8: ```bash run.sh <scheduler>```, ```<scheduler>``` can be set to
  - ```shortest```: SRTF
  - ```shortest-gpu```: SRSF
  - ```multi-resource-blossom-same-gpu```: Muri-S
  - ```dlas-gpu```: Tiresias
  - ```themis```: Themis
  - ```multi-resource-blossom-same-gpu-unaware```: Muri-L
- Each test takes about 1 day. -->

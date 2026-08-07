# Project Transfer Document — Slurm Bridge on OCP

## Overview

This is a proof-of-concept that deploys [Slurm with Slurm Bridge](https://slinky.schedmd.com/) on Red Hat OpenShift, enabling Kubernetes-native job submission routed through Slurm's HPC scheduler. Workloads are submitted as ordinary Kubernetes Jobs — no Slurm CLI knowledge required.

**Repository:** [RHEcosystemAppEng/slurm-bridge-on-ocp](https://github.com/RHEcosystemAppEng/slurm-bridge-on-ocp)  
**Target cluster:** OpenShift 4.x (validated on OCP 4.22.6 / K8s v1.35.5)  
**Status:** Functional POC — validated end-to-end with CPU training workloads

---

## What's Been Done

| Milestone | Status | Evidence |
|-----------|--------|----------|
| Automated deployment (operator + Slurm + Bridge) | Done | `./scripts/deploy.sh` — idempotent, handles RBAC workarounds |
| Smoke test (Bridge-routed pod) | Done | Busybox pod confirmed via `squeue` on slurmctld |
| Training demo — AG News subset (8k rows) | Done | 23% → 91% accuracy, ~36 min on CPU |
| Training demo — Full AG News (120k rows) | Done | 23% → 94% accuracy, ~7h 53min on CPU |
| Detached training with PVC-backed results | Done | Results persist after pod exits; fetch any time |
| Demo script (`demos/text-classifier-demo.sh`) | Done | Supports `--detach`, `--fetch-results`, `--dataset`, `--gpu` |
| Documentation (architecture, deploy guide, demo) | Done | `docs/` directory |

---

## Possible Goals / Next Steps

### 1. GPU Validation

**Priority: High**

The training image auto-detects CUDA and the demo script supports `--gpu N`, but GPU training has not been tested on a real cluster. The code paths exist:

- `train.py` uses `nccl` backend when CUDA is available
- `Dockerfile` uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- GPU nodes just need the Bridge external-node label

**What's needed:** An OCP cluster with the NVIDIA GPU Operator installed and at least one GPU node. Label the nodes, run `./demos/text-classifier-demo.sh --image <img> --gpu 1`, and confirm the training completes and uses the GPU.

### 2. Multi-Node Distributed Training

**Priority: Medium**

Currently validated: single-node, multi-process DDP (2 processes sharing one pod). The code (`train.py`) is DDP-ready and would work across multiple nodes with only a launch-command change — but this hasn't been tested through Bridge.

**Open question:** Can Bridge route a multi-pod Job (e.g., PyTorchJob or multiple replicas) and have Slurm coordinate them? This is at the boundary of what Bridge's pod-level interception model supports.

### 3. Autoscaling Integration

**Priority: Medium-Low (exploratory)**

Bridge uses a fixed pool of labeled worker nodes. This conflicts with Kubernetes-native autoscalers that operate without Slurm queue awareness.

**Possible approaches:**
- KEDA with a Slurm REST API scaler (reads `squeue` job count, scales node group)
- Custom controller bridging Slurm queue depth to a K8s HPA/ScaledObject metric
- Slurm's own `ResumeProgram`/`SuspendProgram` hooks to provision/deprovision OCP nodes

All require significant engineering. Left as future work.

### 4. Integration with OpenShift AI / Kubeflow

**Priority: Medium**

The whole point of Bridge is transparent integration. A natural next step is submitting training jobs from OpenShift AI (RHOAI) notebooks or Kubeflow Pipelines and confirming they route through Slurm without modification.

### 5. 20 Newsgroups Dataset Testing

**Priority: Low**

The `--dataset news20` path exists (11k train, 7k test, 20 classes) but hasn't been live-tested. Good for validating the model on a harder classification task.

---

## Known Shortcomings

### Architectural

| Issue | Impact | Workaround |
|-------|--------|------------|
| **Fixed compute pool** — no autoscaling | Manual node labeling; can't react to queue depth | None yet; future work |
| **Namespace isolation too broad** — Bridge intercepts ALL pods in labeled namespaces | Build pods, sidecars, infra pods all get stuck | Keep workload namespaces separate from build/infra |
| **Single-namespace constraint** — Bridge's Token CR must be in same namespace as Slurm cluster | Can't split Bridge into its own namespace | Always install Bridge into `slurm` namespace |
| **NodeSet replicas=0** — Slurm "compute" is actually OCP workers | No traditional Slurm node lifecycle management | Acceptable for POC |

### Operational

| Issue | Impact | Status |
|-------|--------|--------|
| **Bridge `schedulerConfig.partition` defaults to Helm release name** | Jobs fail with "Invalid partition name specified" if not overridden | Fixed in `configs/slurm-bridge-values.yaml` |
| **RBAC patches reset on Helm upgrade** | Bridge scheduler crashes with permission errors after upgrade | Documented; `deploy-bridge.sh` reapplies automatically |
| **`oc cp` times out on files >200MB** | Can't copy large model checkpoints directly | Script uses gzip streaming workaround |
| **Helper pods need full OpenShift securityContext** | Pods stay Pending without it | Fixed in demo script |
| **Slurm chart pinned to v1.2.0** | May miss upstream fixes/features | Deliberate: stability over latest |
| **Bridge chart version unpinned** | Could break on breaking upstream changes | Should pin after validation |

### Training-Specific

| Issue | Impact | Notes |
|-------|--------|-------|
| **CPU-only validation** | GPU path exists but unconfirmed | Image supports both; needs GPU cluster |
| **Memory sizing** | 3Gi/6Gi OOMKills with 2 DDP processes | 6Gi/12Gi is stable |
| **OMP_NUM_THREADS default** | torchrun sets it to 1, doubling training time | Explicitly set to 2 in Job spec |
| **Full dataset overfitting** | Accuracy peaks at epoch 1 (94.3%) then drops to 93.6% | 1-2 epochs optimal for 120k dataset |
| **AG News license** | Research/non-commercial only | Fine for internal use; don't redistribute externally |

---

## Dependencies

| Component | Source | Version |
|-----------|--------|---------|
| OpenShift | — | 4.x (tested on 4.22.6) |
| Helm | — | 3.x (tested on 3.19.0) |
| cert-manager | jetstack/cert-manager | v1.13.0 |
| Slinky operator | `oci://ghcr.io/slinkyproject/charts/slurm-operator` | 1.2.1 |
| Slurm chart | `oci://ghcr.io/slinkyproject/charts/slurm` | **1.2.0** (pinned) |
| Slurm Bridge | `oci://ghcr.io/slinkyproject/charts/slurm-bridge` | 1.2.1 (unpinned) |
| PyTorch | pip / Docker base | 2.5.1 |
| transformers | pip | 4.46.3 |
| Model | Hugging Face (baked into image) | distilbert-base-uncased |

---

## Quick Start for New Owners

```bash
# 1. Log into your OCP cluster
oc login --token=<token> --server=<server>

# 2. Deploy the full stack (~1 min)
./scripts/deploy.sh

# 3. Verify
oc get pods -n slurm

# 4. Build the training image
oc new-project text-classifier-build
oc new-build --name=text-classifier-trainer --binary --strategy=docker
oc start-build text-classifier-trainer --from-dir=training/ --follow
oc policy add-role-to-group system:image-puller \
  system:serviceaccounts:text-classifier-demo -n text-classifier-build

# 5. Run training (detached — results persist on PVC)
./demos/text-classifier-demo.sh \
  --image image-registry.openshift-image-registry.svc:5000/text-classifier-build/text-classifier-trainer:latest \
  --dataset full --detach

# 6. Fetch results whenever ready
./demos/text-classifier-demo.sh --fetch-results

# 7. Cleanup
./scripts/cleanup.sh
```

---

## File Map

```
├── configs/
│   ├── slurm-values.yaml           # Partition config, NodeSet replicas
│   ├── slurm-bridge-values.yaml    # Namespace selector + partition override
│   └── token.yaml                  # JWT token CR for Bridge auth
├── scripts/
│   ├── deploy.sh                   # Master orchestrator
│   ├── deploy-operator.sh          # Slinky CRDs + operator
│   ├── deploy-slurm.sh            # Slurm cluster (controller + REST API)
│   ├── deploy-bridge.sh           # Bridge + RBAC patches + node labels
│   └── cleanup.sh                 # Teardown
├── training/
│   ├── train.py                   # DDP-ready DistilBERT fine-tuning
│   ├── predict.py                 # Inference on checkpoint
│   ├── Dockerfile                 # GPU image (CUDA 12.4)
│   ├── Dockerfile.cpu             # CPU-only image
│   ├── data/                      # AG News subset (8k/2k)
│   ├── data-full/                 # Full AG News (120k/7.6k)
│   └── data-news/                 # 20 Newsgroups (11k/7k)
├── demos/
│   └── text-classifier-demo.sh   # End-to-end demo (submit, monitor, fetch)
├── docs/
│   ├── ARCHITECTURE.md           # System design
│   ├── DEPLOYMENT_GUIDE.md       # Step-by-step deployment
│   ├── DEMO.md                   # Training demo walkthrough
│   ├── DEMO_RECORDING_SCRIPT.md  # Video recording runbook
│   └── TRANSFER.md               # This document
└── results/                      # Local training outputs (gitignored)
```

---

## Validated Results

### Full AG News (120k train / 7.6k test, CPU, 3 epochs)

```json
{
  "baseline_accuracy": 0.235,
  "epochs": [
    {"epoch": 1, "train_loss": 0.2153, "eval_accuracy": 0.9434},
    {"epoch": 2, "train_loss": 0.1331, "eval_accuracy": 0.9411},
    {"epoch": 3, "train_loss": 0.0941, "eval_accuracy": 0.9364}
  ],
  "final_accuracy": 0.9364
}
```

Training time: ~7h 53min on 4 CPU cores (2 DDP processes, `OMP_NUM_THREADS=2`).

---

## Contacts and References

- **Upstream Slinky:** https://slinky.schedmd.com/
- **Base project:** https://github.com/RHEcosystemAppEng/slurm-on-ocp
- **This repo:** https://github.com/RHEcosystemAppEng/slurm-bridge-on-ocp

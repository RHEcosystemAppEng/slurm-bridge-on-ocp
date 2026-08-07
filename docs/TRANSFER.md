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

**Priority: Medium-High**

The current deployment uses a **fixed pool** of labeled worker nodes — you label 3 (or N) nodes at deploy time and that's the compute Bridge can use. This means:

- Jobs queue if all labeled nodes are busy, even if the cluster has idle capacity elsewhere
- Idle labeled nodes waste resources when no Slurm jobs are running
- No way to burst for large training runs and scale back down afterward

This is the single biggest gap between "POC" and "production-ready."

**Why it's hard:** Slurm and Kubernetes have fundamentally different scaling models. Slurm manages a static node inventory and decides what runs where. Kubernetes autoscalers (Cluster Autoscaler, KEDA, MachineSet scaling) add/remove nodes based on pod-level demand. Bridge sits in the middle — it intercepts K8s pods but submits them as Slurm jobs, so neither system has full visibility into the other's state.

**Approach A: KEDA + Slurm REST API scaler**

Write a custom KEDA scaler that queries `slurmrestd` (`GET /slurm/v0.0.44/jobs/` with state filter for pending jobs). When pending job count exceeds a threshold, KEDA scales a MachineSet (or a node group in the cloud provider) to add OCP worker nodes. A post-scale hook labels the new nodes as Bridge external nodes. On scale-down, unlabel and drain.

- Pros: Clean separation; KEDA is well-supported on OCP
- Cons: Latency (new nodes take minutes to provision); label lifecycle management; race conditions between Slurm queue and KEDA cooldown

**Approach B: Slurm elastic compute (`ResumeProgram` / `SuspendProgram`)**

Slurm natively supports elastic scaling through configurable programs called when nodes transition between idle→powered-down and powered-down→resuming. These could be wired to OpenShift MachineSet scaling or cloud provider APIs:

```
ResumeProgram = /usr/local/bin/ocp-resume-node.sh   # scales MachineSet up + labels node
SuspendProgram = /usr/local/bin/ocp-suspend-node.sh # unlabels + scales MachineSet down
SuspendTime = 300  # idle for 5min → power down
```

- Pros: Slurm-native; scheduling decisions stay with Slurm (which has queue depth awareness); proven pattern in traditional HPC clouds
- Cons: Scripts run inside slurmctld pod (security implications); need RBAC to modify MachineSets from inside the cluster; mismatch between Slurm's node-centric model and OCP's machine-centric model

**Approach C: Custom Kubernetes controller**

A dedicated controller that watches both the Slurm REST API (pending jobs, idle nodes) and OCP MachineSets/MachineAutoscalers. It bridges the gap by:

1. Polling Slurm queue depth every N seconds
2. If pending jobs > threshold for > M seconds, scaling up MachineSets
3. Labeling new Ready nodes as Bridge external nodes
4. On scale-down: draining Slurm jobs from a node (`scontrol update NodeName=X State=DRAIN`), waiting for completion, then unlabeling and deleting the Machine

- Pros: Most control; can encode complex policies (e.g., "never scale below 3 nodes", "max burst to 10 for GPU jobs")
- Cons: Most engineering effort; custom CRD + reconciliation loop; failure modes need careful handling

**Recommended starting point:** Approach A (KEDA) for a quick win on CPU workloads, with Approach B as a longer-term target for tighter Slurm integration. Approach C is the "full production" answer but requires the most investment.

**Open questions:**
- How long does it take for a new OCP worker node to become schedulable by Bridge after MachineSet scale-up? (Includes: machine provisioning, node bootstrap, kubelet ready, Bridge controller detecting the label and adding it to Slurm's node inventory)
- Can Bridge handle nodes appearing/disappearing dynamically, or does it require a restart?
- What's the minimum idle time before scale-down is safe (must account for Slurm job draining)?

### 4. Integration with OpenShift AI / Kubeflow

**Priority: Medium**

The whole point of Bridge is transparent integration. A natural next step is submitting training jobs from OpenShift AI (RHOAI) notebooks or Kubeflow Pipelines and confirming they route through Slurm without modification.

### 5. More Practical / Production-Relevant Workloads

**Priority: Medium**

The current demo (AG News text classification with DistilBERT) is deliberately small and fast for validation purposes. To make the POC more compelling for production conversations, consider replacing or supplementing it with workloads that better represent real enterprise use cases:

**Larger language models:**
- Fine-tune a **Llama 2 7B** or **Mistral 7B** LoRA adapter on a domain-specific dataset (e.g., customer support tickets, internal documentation). This exercises GPU memory management, gradient checkpointing, and actual multi-hour training times that benefit from Slurm's job scheduling.
- Fine-tune **CodeLlama** on internal code repositories — directly relevant to developer tooling teams.

**Computer vision:**
- Train a **YOLOv8** or **ResNet** model on a defect detection dataset (manufacturing QC). This is a common Red Hat customer use case and exercises different resource profiles (high GPU utilization, large image batches).
- Medical imaging classification (e.g., chest X-ray) — demonstrates compliance-sensitive workloads where Slurm's accounting and resource isolation add value.

**Tabular / structured data:**
- Train an **XGBoost** or **LightGBM** ensemble on a fraud detection dataset. Demonstrates that Bridge isn't limited to deep learning — any batch workload benefits from Slurm scheduling.
- Time-series forecasting (demand planning, anomaly detection) with PyTorch Forecasting.

**Multi-job pipelines:**
- A realistic ML pipeline: data preprocessing Job → training Job → evaluation Job → model registration. Submit all through Bridge and demonstrate Slurm managing the dependency chain (or use Kubeflow Pipelines with Bridge-managed execution).

**Why this matters more than 20 Newsgroups:** The 20 Newsgroups dataset (`--dataset news20`) tests a harder classification task (20 classes vs 4) but doesn't fundamentally change the story. It's still a small text classification problem on a small model. The value of Slurm scheduling becomes most obvious when:
- Training takes hours/days (queue management matters)
- Multiple users compete for GPUs (fair-share scheduling matters)
- Jobs have heterogeneous resource requirements (Slurm's partition/QOS/priority system matters)
- Compliance requires job accounting and audit trails

A single LoRA fine-tuning run on a 7B model with real data would demonstrate all of these better than any number of DistilBERT experiments.

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

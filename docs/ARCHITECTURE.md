# Architecture — Slurm Bridge on OpenShift

## Overview

This project runs a Slurm cluster inside OpenShift using the Slinky operator and exposes it to Kubernetes-native workloads via Slurm Bridge. The POC demonstrates the full path with a PyTorch fine-tuning job submitted as a standard Kubernetes Job.

## Components

| Component | Namespace | Role |
|-----------|-----------|------|
| Slinky Operator | `slinky` | Watches Controller/NodeSet CRs, reconciles Slurm pods |
| slurmctld | `slurm` | Slurm scheduler and job queue |
| slurmrestd | `slurm` | Slurm REST API (required by Bridge) |
| Bridge admission | `slurm` | Intercepts pod creation in labeled namespaces |
| Bridge scheduler | `slurm` | Schedules intercepted pods via Slurm |
| Bridge controllers | `slurm` | Manages Bridge lifecycle |
| OCP worker nodes | cluster | External compute pool (labeled during deploy; GPU nodes require NVIDIA GPU Operator) |

> **Note:** Bridge is installed into the **Slurm cluster namespace** (`slurm`), not the Slinky operator's namespace (`slinky`). Bridge's `Token` CR reads the `slurm-auth-jwt` secret (created by the Slurm Helm chart) and writes the `slurm-bridge-token` secret that Bridge's pods consume — both lookups are same-namespace, so Token, secret, and Bridge pods must all live alongside `slurmrestd` in `slurm`.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  OpenShift Cluster                                                   │
│                                                                      │
│  ┌────────────────────────┐                                         │
│  │  slinky namespace       │                                        │
│  │  ┌──────────────────┐  │                                         │
│  │  │  Slinky Operator │  │                                         │
│  │  │  (watches CRs)   │  │                                         │
│  │  └────────┬─────────┘  │                                         │
│  └───────────┼────────────┘                                         │
│              │ reconciles                                            │
│              ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  slurm namespace                                             │    │
│  │                                                               │    │
│  │  ┌────────────────────────────────┐                          │    │
│  │  │  Slurm Bridge                  │                          │    │
│  │  │  - admission controller        │                          │    │
│  │  │  - scheduler                   │                          │    │
│  │  │  - controllers                  │                          │    │
│  │  └──────────────┬───────────────────┘                        │    │
│  │                 │ REST API calls                              │    │
│  │  ┌──────────────▼──────────────┐  ┌─────────────┐            │    │
│  │  │ slurmctld                    │  │ slurmrestd  │            │    │
│  │  │ (scheduler)                  │◄─┤ (REST API)  │            │    │
│  │  └──────┬───────────────────────┘  └─────────────┘            │    │
│  │         │ schedule jobs                                       │    │
│  │         ▼                                                     │    │
│  │  ┌──────────────────────────────────────────┐                │    │
│  │  │  OCP worker nodes (external-node pool)   │                │    │
│  │  │  labeled during deploy-bridge.sh         │                │    │
│  │  └──────────────────────────────────────────┘                │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  text-classifier-demo namespace (managed-by-slurm=true)       │    │
│  │  ┌────────────────────────────────┐                          │    │
│  │  │  K8s Job (torchrun + train.py) │ ──► intercepted by Bridge│    │
│  │  └────────────────────────────────┘                          │    │
│  └───────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Job Flow via Slurm Bridge

1. A Kubernetes workload (Job, Pod, pipeline) is created in a namespace labeled `managed-by-slurm: "true"`
2. Bridge's admission controller intercepts the pod creation
3. Bridge translates the pod spec into an sbatch job via the Slurm REST API
4. slurmctld schedules the job onto an external OCP worker node
5. The pod runs on the node; Slurm tracks the job via `squeue`/`sacct`

> **Important (confirmed via live testing):** Bridge's admission controller intercepts **every** pod created in a `managed-by-slurm: "true"` namespace — not just the workload pods you intend to route through Slurm. This includes infrastructure pods you don't control the spec of, like OpenShift's own `BuildConfig` build pods. Those typically need a privileged/`hostPath` security context that Slurm has no way to satisfy, so Bridge repeatedly tries and fails to bind them, and they sit in `Pending` forever with **zero scheduling events**. Keep build/infra namespaces separate from any namespace labeled `managed-by-slurm`.

## Validated Workloads

The following use cases have been run end-to-end on a real OpenShift cluster through the Bridge path described above.

### Text Classification — DistilBERT Fine-Tuning (AG News)

**What was tested:** Fine-tuning `distilbert-base-uncased` on a 4-class news topic classifier (World / Sports / Business / Sci-Tech) submitted as a standard Kubernetes `batch/v1` Job routed through Bridge.

**Dataset:** AG News subset — 8,000 train / 2,000 test rows (~2 MB). This is a small, fast stand-in for a real HPC training job; the Bridge path, job spec, and DDP setup are identical for the full AG News dataset (120k train / 7.6k test) or any other drop-in replacement.

**Training setup:**
- Single-node, multi-process DDP via `torchrun` (`--nproc_per_node=2`, CPU)
- `OMP_NUM_THREADS=2`, 4 CPU cores, 6Gi–12Gi memory
- 3 epochs

**Confirmed results (CPU, 3 epochs):**
```
baseline accuracy (pre-training): 23.45%   (~chance for 4 classes)
epoch 1/3  train_loss=0.3831  eval_accuracy=90.35%
epoch 2/3  train_loss=0.1956  eval_accuracy=90.80%
epoch 3/3  train_loss=0.1132  eval_accuracy=90.85%
total wall-clock: ~36 minutes
```

**Confirmed via `squeue` on `slurmctld`** that the job was scheduled and run by Slurm, not Kubernetes' default scheduler — Bridge's admission controller intercepted the Job pod and routed it through the Slurm REST API as intended.

**GPU path:** The training image auto-detects CUDA and switches to the `nccl` DDP backend when GPUs are available. GPU nodes must be labeled as Bridge external nodes (see Deployment Guide) and have the NVIDIA GPU Operator installed. GPU execution has not yet been validated at scale — that is the planned next step (see Limitations below).

**What changes for a larger model or more data:** Only the dataset path or model name in the Job spec. The Bridge routing, DDP launch, and checkpoint/metrics output paths are the same regardless of model or dataset size.

## Key Differences from slurm-on-ocp

- **Helm-based deployment** — the Slurm chart creates Controller, NodeSet, and RestApi CRs together. The RestApi is required by Bridge.
- **slurmrestd** — exposes Slurm's job submission and monitoring as a REST API. Bridge calls this instead of using `oc exec ... sbatch`.
- **Token CR** — Bridge authenticates to slurmrestd with a JWT token managed by the `Token` CR.
- **Node labeling** — worker OCP nodes need `scheduler.slinky.slurm.net/external-node: "true"` for Bridge to consider them eligible.
- **Kubernetes-native submission** — workloads submit as standard Pods/Jobs; no Slurm CLI knowledge required in the workload definition.

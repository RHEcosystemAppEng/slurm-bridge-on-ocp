#!/bin/bash
set -euo pipefail

###############################################################################
# Text Classifier Demo — Submit a fine-tuning job via Slurm Bridge
#
# Submits a Kubernetes Job that fine-tunes DistilBERT on AG News (or 20
# Newsgroups) via Slurm Bridge. The job is an ordinary batch/v1 Job in a
# namespace labeled managed-by-slurm=true — Bridge intercepts it and routes
# through Slurm.
#
# Usage:
#   ./demos/text-classifier-demo.sh --image <image>
#   ./demos/text-classifier-demo.sh --image <image> --gpu 1
#   ./demos/text-classifier-demo.sh --image <image> --dataset full
#   ./demos/text-classifier-demo.sh --image <image> --dataset full --detach
#   ./demos/text-classifier-demo.sh --fetch-results
#   ./demos/text-classifier-demo.sh --cleanup
###############################################################################

IMAGE=""
GPU_COUNT=0
DATASET="default"
NAMESPACE="text-classifier-demo"
JOB_NAME="text-classifier-training"
PVC_NAME="training-results"
CLEANUP=false
DETACH=false
FETCH_RESULTS=false
EPOCHS=3
RESULTS_DIR="results/text-classifier-demo"

while [[ $# -gt 0 ]]; do
  case $1 in
    --image)         IMAGE="$2"; shift 2 ;;
    --gpu)           GPU_COUNT="$2"; shift 2 ;;
    --dataset)       DATASET="$2"; shift 2 ;;
    --namespace)     NAMESPACE="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --cleanup)       CLEANUP=true; shift ;;
    --detach)        DETACH=true; shift ;;
    --fetch-results) FETCH_RESULTS=true; shift ;;
    --help|-h)       sed -n '3,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
log()     { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn()    { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"; }
error()   { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $1"; }
section() { echo ""; echo -e "${BLUE}━━━ $1 ━━━${NC}"; echo ""; }

# ---------------------------------------------------------------------------
# Fetch results from PVC (can be run any time after training completes)
# ---------------------------------------------------------------------------
if [ "$FETCH_RESULTS" = true ]; then
  section "Fetching results from PVC"

  if ! oc get pvc "$PVC_NAME" -n "$NAMESPACE" &>/dev/null; then
    error "PVC '$PVC_NAME' not found in namespace '$NAMESPACE'"
    error "Has a training job been submitted? Try: oc get pvc -n $NAMESPACE"
    exit 1
  fi

  HELPER_POD="results-fetcher"
  oc delete pod "$HELPER_POD" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
  sleep 2

  log "Starting helper pod to access results PVC..."
  cat <<EOF | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER_POD}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: fetcher
    image: registry.access.redhat.com/ubi9/ubi:latest
    command: ["sh", "-c", "echo ready && sleep 600"]
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
    volumeMounts:
    - name: results
      mountPath: /results
      readOnly: true
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF

  log "Waiting for helper pod to be ready..."
  oc wait --for=condition=Ready pod/"$HELPER_POD" -n "$NAMESPACE" --timeout=120s

  if ! oc exec -n "$NAMESPACE" "$HELPER_POD" -- test -f /results/metrics.json 2>/dev/null; then
    warn "metrics.json not found — training may still be running"
    oc exec -n "$NAMESPACE" "$HELPER_POD" -- ls -la /results/ 2>/dev/null || true
    oc delete pod "$HELPER_POD" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    exit 1
  fi

  mkdir -p "$RESULTS_DIR/checkpoint"
  log "Copying metrics.json..."
  oc exec -n "$NAMESPACE" "$HELPER_POD" -- cat /results/metrics.json > "$RESULTS_DIR/metrics.json"

  log "Copying checkpoint files (streaming via gzip to handle large model files)..."
  for f in $(oc exec -n "$NAMESPACE" "$HELPER_POD" -- ls /results/checkpoint/); do
    log "  $f"
    oc exec -n "$NAMESPACE" "$HELPER_POD" -- sh -c "gzip -c /results/checkpoint/$f" \
      > "$RESULTS_DIR/checkpoint/${f}.gz" 2>/dev/null \
      && gunzip "$RESULTS_DIR/checkpoint/${f}.gz" \
      || oc exec -n "$NAMESPACE" "$HELPER_POD" -- cat "/results/checkpoint/$f" \
        > "$RESULTS_DIR/checkpoint/$f" 2>/dev/null
  done

  oc delete pod "$HELPER_POD" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true

  section "Results retrieved"
  log "Saved to $RESULTS_DIR/"
  echo ""
  cat "$RESULTS_DIR/metrics.json"
  echo ""
  log "Try the model:"
  log "  python3 training/predict.py --checkpoint-dir $RESULTS_DIR/checkpoint --interactive"
  exit 0
fi

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
if [ "$CLEANUP" = true ]; then
  section "Cleaning up text-classifier-demo"
  oc delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found
  oc delete pvc "$PVC_NAME" -n "$NAMESPACE" --ignore-not-found
  oc delete pod results-fetcher -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
  oc delete namespace "$NAMESPACE" --ignore-not-found
  log "Cleanup complete"
  exit 0
fi

# ---------------------------------------------------------------------------
# Validate args
# ---------------------------------------------------------------------------
if [ -z "$IMAGE" ]; then
  error "Missing required --image argument"
  echo "Usage: $0 --image <training-image> [--gpu N] [--dataset full|news20] [--detach]"
  exit 1
fi

# ---------------------------------------------------------------------------
# Resolve dataset path inside the container
# ---------------------------------------------------------------------------
case "$DATASET" in
  default|subset) DATA_DIR="/app/data" ;;
  full)           DATA_DIR="/app/data-full" ;;
  news20)         DATA_DIR="/app/data-news" ;;
  *) error "Unknown dataset: $DATASET (use: default, full, news20)"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Compute resource requests
# ---------------------------------------------------------------------------
if [ "$GPU_COUNT" -gt 0 ]; then
  NPROC="$GPU_COUNT"
  CPU_REQUEST="4"
  CPU_LIMIT="8"
  MEM_REQUEST="8Gi"
  MEM_LIMIT="16Gi"
  GPU_RESOURCE="\"nvidia.com/gpu\": \"$GPU_COUNT\""
else
  NPROC=2
  CPU_REQUEST="4"
  CPU_LIMIT="4"
  MEM_REQUEST="6Gi"
  MEM_LIMIT="12Gi"
  GPU_RESOURCE=""
fi

# ---------------------------------------------------------------------------
# Create namespace and label for Bridge
# ---------------------------------------------------------------------------
section "Setting up namespace"
oc create namespace "$NAMESPACE" 2>/dev/null || true
oc label namespace "$NAMESPACE" managed-by-slurm=true --overwrite
log "Namespace '$NAMESPACE' labeled for Slurm Bridge"

# ---------------------------------------------------------------------------
# Create PVC for results (persists after pod exits — no grace period needed)
# ---------------------------------------------------------------------------
if ! oc get pvc "$PVC_NAME" -n "$NAMESPACE" &>/dev/null; then
  log "Creating PVC for training results..."
  cat <<EOF | oc apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  namespace: ${NAMESPACE}
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 2Gi
EOF
else
  log "PVC '$PVC_NAME' already exists — reusing"
fi

# ---------------------------------------------------------------------------
# Delete previous job if it exists
# ---------------------------------------------------------------------------
oc delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
sleep 2

# ---------------------------------------------------------------------------
# Build Job manifest
# ---------------------------------------------------------------------------
section "Submitting training job"

GPU_RESOURCES_BLOCK=""
if [ "$GPU_COUNT" -gt 0 ]; then
  GPU_RESOURCES_BLOCK="\"nvidia.com/gpu\": \"$GPU_COUNT\","
fi

cat <<EOF | oc apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  template:
    metadata:
      annotations:
        slurmjob.slinky.slurm.net/account: "slurm"
        slurmjob.slinky.slurm.net/partition: "all"
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: ${IMAGE}
        command: ["sh", "-c"]
        args:
        - |
          torchrun --nnodes=1 --nproc_per_node=${NPROC} /app/train.py \
            --data-dir ${DATA_DIR} \
            --output-dir /results \
            --epochs ${EPOCHS}
          echo "=== Training complete ==="
        env:
        - name: OMP_NUM_THREADS
          value: "2"
        resources:
          requests:
            cpu: "${CPU_REQUEST}"
            memory: "${MEM_REQUEST}"
            ${GPU_RESOURCES_BLOCK}
          limits:
            cpu: "${CPU_LIMIT}"
            memory: "${MEM_LIMIT}"
            ${GPU_RESOURCES_BLOCK}
        volumeMounts:
        - name: results
          mountPath: /results
      volumes:
      - name: results
        persistentVolumeClaim:
          claimName: ${PVC_NAME}
EOF

log "Job '$JOB_NAME' submitted in namespace '$NAMESPACE'"
log "Dataset: $DATASET ($DATA_DIR)"
log "Processes: $NPROC | CPU: ${CPU_REQUEST}/${CPU_LIMIT} | Memory: ${MEM_REQUEST}/${MEM_LIMIT}"
if [ "$GPU_COUNT" -gt 0 ]; then
  log "GPUs: $GPU_COUNT"
fi
log "Results volume: PVC '$PVC_NAME' (persists after pod exits)"

# ---------------------------------------------------------------------------
# If --detach, exit here
# ---------------------------------------------------------------------------
if [ "$DETACH" = true ]; then
  section "Detached mode — results auto-persist on PVC"
  log "Results are written to PVC '$PVC_NAME' and persist after the pod exits."
  log "No grace period needed — fetch results any time after training completes."
  log ""
  log "Monitor training:"
  log "  oc get pods -n $NAMESPACE -w"
  log "  oc logs -n $NAMESPACE -l job-name=$JOB_NAME -f"
  log ""
  log "Check Slurm tracking:"
  log "  CTRL=\$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')"
  log "  oc exec -n slurm \$CTRL -c slurmctld -- squeue"
  log ""
  log "Fetch results (run any time after training finishes):"
  log "  ./demos/text-classifier-demo.sh --fetch-results"
  exit 0
fi

# ---------------------------------------------------------------------------
# Wait for pod to start, then tail logs
# ---------------------------------------------------------------------------
section "Waiting for training pod"
log "Waiting for pod to be scheduled and start running..."

TIMEOUT=300
ELAPSED=0
while true; do
  POD_PHASE=$(oc get pods -n "$NAMESPACE" -l job-name="$JOB_NAME" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
  if [ "$POD_PHASE" = "Running" ]; then
    break
  elif [ "$POD_PHASE" = "Failed" ]; then
    error "Pod failed before starting training"
    oc describe pod -n "$NAMESPACE" -l job-name="$JOB_NAME" | tail -20
    exit 1
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    error "Timed out waiting for pod to start (${TIMEOUT}s)"
    oc get pods -n "$NAMESPACE" -o wide
    oc describe pod -n "$NAMESPACE" -l job-name="$JOB_NAME" | tail -20
    exit 1
  fi
done

POD_NAME=$(oc get pods -n "$NAMESPACE" -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}')
log "Pod running: $POD_NAME"

section "Training logs"
oc logs -n "$NAMESPACE" "$POD_NAME" -f 2>/dev/null || true

# ---------------------------------------------------------------------------
# Retrieve results from PVC
# ---------------------------------------------------------------------------
section "Retrieving results"

mkdir -p "$RESULTS_DIR"

HELPER_POD="results-fetcher"
oc delete pod "$HELPER_POD" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
sleep 2

cat <<EOF | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER_POD}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: fetcher
    image: registry.access.redhat.com/ubi9/ubi:latest
    command: ["sh", "-c", "echo ready && sleep 300"]
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
    volumeMounts:
    - name: results
      mountPath: /results
      readOnly: true
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF

oc wait --for=condition=Ready pod/"$HELPER_POD" -n "$NAMESPACE" --timeout=120s

mkdir -p "$RESULTS_DIR/checkpoint"
oc exec -n "$NAMESPACE" "$HELPER_POD" -- cat /results/metrics.json > "$RESULTS_DIR/metrics.json" && \
  log "Saved: $RESULTS_DIR/metrics.json" || \
  warn "Failed to copy metrics.json"

log "Copying checkpoint (streaming via gzip)..."
for f in $(oc exec -n "$NAMESPACE" "$HELPER_POD" -- ls /results/checkpoint/); do
  log "  $f"
  oc exec -n "$NAMESPACE" "$HELPER_POD" -- sh -c "gzip -c /results/checkpoint/$f" \
    > "$RESULTS_DIR/checkpoint/${f}.gz" 2>/dev/null \
    && gunzip "$RESULTS_DIR/checkpoint/${f}.gz" \
    || oc exec -n "$NAMESPACE" "$HELPER_POD" -- cat "/results/checkpoint/$f" \
      > "$RESULTS_DIR/checkpoint/$f" 2>/dev/null
done
log "Saved: $RESULTS_DIR/checkpoint/"

oc delete pod "$HELPER_POD" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section "Done"
if [ -f "$RESULTS_DIR/metrics.json" ]; then
  log "Results saved to $RESULTS_DIR/"
  echo ""
  cat "$RESULTS_DIR/metrics.json"
  echo ""
  log "Try the model:"
  log "  python3 training/predict.py --checkpoint-dir $RESULTS_DIR/checkpoint --interactive"
else
  warn "Results not retrieved — check pod status:"
  warn "  oc get pods -n $NAMESPACE"
fi

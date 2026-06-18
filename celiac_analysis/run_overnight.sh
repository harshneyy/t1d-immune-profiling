#!/bin/bash
set -e
source /home/harshney/ml_env/bin/activate
LOGFILE="/home/harshney/celiac-immune-profiling/logs/overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /home/harshney/celiac-immune-profiling/logs

echo "=== CELIAC OVERNIGHT PIPELINE STARTED $(date) ===" | tee -a "$LOGFILE"

# Wait for download to complete
RAW_TAR="/home/harshney/celiac-immune-profiling/data/raw/GSE315138_RAW.tar"
echo "Waiting for download: $RAW_TAR" | tee -a "$LOGFILE"
while true; do
  if [ -f "$RAW_TAR" ]; then
    SIZE=$(stat -c%s "$RAW_TAR" 2>/dev/null || echo 0)
    if [ "$SIZE" -ge 380000000 ]; then
      echo "Download complete: ${SIZE} bytes at $(date)" | tee -a "$LOGFILE"
      break
    fi
    echo "  Downloading... $(du -h $RAW_TAR | cut -f1) / 365MB — $(date +%H:%M:%S)" | tee -a "$LOGFILE"
  fi
  sleep 60
done

echo "=== STEP 1: Preprocessing + scVI GPU Training — $(date) ===" | tee -a "$LOGFILE"
python /home/harshney/celiac-immune-profiling/scripts/step1_preprocess_scvi.py 2>&1 | tee -a "$LOGFILE"
echo "=== STEP 1 DONE $(date) ===" | tee -a "$LOGFILE"

echo "=== STEP 2: Geneformer Embeddings — $(date) ===" | tee -a "$LOGFILE"
python /home/harshney/celiac-immune-profiling/scripts/step2_geneformer.py 2>&1 | tee -a "$LOGFILE"
echo "=== STEP 2 DONE $(date) ===" | tee -a "$LOGFILE"

echo "=== STEP 3: Classification + Cross-Disease — $(date) ===" | tee -a "$LOGFILE"
python /home/harshney/celiac-immune-profiling/scripts/step3_classify_compare.py 2>&1 | tee -a "$LOGFILE"
echo "=== ALL DONE $(date) ===" | tee -a "$LOGFILE"

echo "Results: /home/harshney/celiac-immune-profiling/results/" | tee -a "$LOGFILE"
echo "Figures: /home/harshney/celiac-immune-profiling/figures/" | tee -a "$LOGFILE"

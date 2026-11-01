#!/bin/bash
# Pack the run directories of one model, with a checksum and a provenance note, to copy back to the machine that analyses them.
#   experiments/paper1/cluster/package_runs.sh Qwen/Qwen3-1.7B                       # runs/sweep (the study)
#   experiments/paper1/cluster/package_runs.sh Qwen/Qwen2.5-1.5B-Instruct platform_check_cuda   # a calibration run in its own directory
# On the receiving machine:  shasum -a 256 -c <package>.sha256 ; tar -xzf <package> -C experiments/paper1/runs/<directory>/ ; python experiments/paper1/verify_runs.py (study runs only)
set -euo pipefail
MODEL="${1:?model id}"
OUTDIR="${2:-sweep}"
cd "$(dirname "$0")/../../.."
slug="${MODEL//\//__}"
dir="experiments/paper1/runs/$OUTDIR/$slug"
[ -d "$dir" ] || { echo "no run directory for $MODEL in runs/$OUTDIR"; exit 1; }
prefix="p1_runs"; [ "$OUTDIR" = "sweep" ] || prefix="p1_calibration_${OUTDIR}"
out="experiments/paper1/cluster/packages/${prefix}_${slug}_$(date +%Y%m%d-%H%M).tar.gz"
mkdir -p "$(dirname "$out")"
{
  echo "model: $MODEL"; echo "runs directory: $OUTDIR"; echo "packed: $(date '+%F %T')"; echo "host: $(hostname)"; echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "finished datasets (DONE marker present):"
  for d in "$dir"/*/; do [ -e "${d}DONE" ] && echo "  $(basename "$d")"; done
} > "$dir/PACKAGED.txt"
tar -czf "$out" -C "experiments/paper1/runs/$OUTDIR" "$slug"
( cd "$(dirname "$out")" && { command -v sha256sum >/dev/null && sha256sum "$(basename "$out")" || shasum -a 256 "$(basename "$out")"; } > "$(basename "$out").sha256" )
echo "$out"; cat "$out.sha256"

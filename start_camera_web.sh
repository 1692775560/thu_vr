#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -u

export PYTHONUNBUFFERED=1
# Production OCR learns the arbitrary 0000..9999 serial range from each row.
# The ablation page still stores its separately entered ground truth.
unset THU_VR_LABEL_SEQUENCE_START THU_VR_LABEL_SEQUENCE_END
export THU_VR_LABEL_ROW_PREFIX_STEP="${THU_VR_LABEL_ROW_PREFIX_STEP:--1}"
export THU_VR_LABEL_ROW_PREFIX_ALLOW_SINGLE_ANCHOR="${THU_VR_LABEL_ROW_PREFIX_ALLOW_SINGLE_ANCHOR:-0}"
exec /usr/bin/python3 "${project_dir}/camera_web.py" \
  --host "${THU_VR_WEB_HOST:-0.0.0.0}" \
  --port "${THU_VR_WEB_PORT:-5090}"

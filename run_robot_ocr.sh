#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
camera_topic="${CAMERA_TOPIC:-/head_rgbd/color/image_raw}"
run_root="${THU_VR_RUN_ROOT:-${project_dir}/runs}"
stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${run_root}/${stamp}"
image_path="${run_dir}/camera_color.jpg"

if [[ ! -x "${project_dir}/.venv/bin/python" ]]; then
  echo "缺少 ${project_dir}/.venv；请先执行部署安装。" >&2
  exit 2
fi

mkdir -p "${run_dir}"

set -a
if [[ -f /home/unix_ai/config/ros_domain_id.env ]]; then
  source /home/unix_ai/config/ros_domain_id.env
fi
set +a
source /opt/ros/humble/setup.bash
if [[ -f /home/unix_ai/work/controller/install/setup.bash ]]; then
  source /home/unix_ai/work/controller/install/setup.bash
fi
set -u

echo "[1/2] 从 ${camera_topic} 抓取一帧..."
/usr/bin/python3 "${project_dir}/capture_ros_image.py" \
  --topic "${camera_topic}" \
  --output "${image_path}" \
  --timeout 10

echo "[2/2] 运行书脊标签 OCR..."
"${project_dir}/.venv/bin/python" "${project_dir}/real_scene_ocr.py" \
  "${image_path}" \
  --no-easy \
  --output-dir "${run_dir}" \
  --output-json result.json

echo "完成：${run_dir}"

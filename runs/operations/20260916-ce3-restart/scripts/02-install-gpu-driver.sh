#!/bin/sh
set -eu
export DEBIAN_FRONTEND=noninteractive
mkdir -p /var/log/wda
log=/var/log/wda/gpu-driver-install-20260916.log
if [ -e "$log" ]; then
    echo "REFUSED: driver-install log already exists; inspect before retrying."
    exit 20
fi
exec 3>&1
trap 'status=$?; printf "driver_install_exit_code=%s\n" "$status" >&3; tail -n 35 "$log" >&3; exit "$status"' EXIT
exec >"$log" 2>&1
date -u
echo "ENGINEERING SETUP ONLY: no model or scientific trial"
timeout --kill-after=30s 300s apt-get -o Acquire::Retries=0 update
candidate=$(apt-cache policy nvidia-driver-550-server | awk '/Candidate:/ {print $2}')
case "$candidate" in
    ""|"(none)") echo "ERROR: nvidia-driver-550-server has no repository candidate"; exit 21 ;;
esac
printf "resolved_nvidia_driver_package=%s\n" "$candidate"
timeout --kill-after=30s 900s apt-get install -y --no-install-recommends "linux-headers-$(uname -r)" "nvidia-driver-550-server=$candidate"
dpkg-query -W nvidia-driver-550-server
if timeout --kill-after=5s 30s nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; then
    echo "GPU_DRIVER_READY_NOT_SCIENTIFIC_QUALIFICATION"
else
    echo "DRIVER_PACKAGE_INSTALLED_BUT_GPU_NOT_READY: inspect module/reboot requirement"
    exit 22
fi
date -u

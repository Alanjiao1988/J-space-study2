#!/bin/sh
set -eu
python3 - <<'PY'
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.error
import urllib.request

result = {
    "schema": "wda/remote-environment-inventory/1",
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "read-only environment inventory; no installation, model download or model execution",
    "hostname": platform.node(),
    "kernel": platform.release(),
    "python": platform.python_version(),
    "logical_cpus": os.cpu_count(),
}
devices = []
for device in sorted(Path("/sys/bus/pci/devices").iterdir()):
    vendor = device / "vendor"
    if vendor.is_file() and vendor.read_text().strip() == "0x10de":
        devices.append({
            "pci_address": device.name,
            "vendor": vendor.read_text().strip(),
            "device": (device / "device").read_text().strip(),
            "class": (device / "class").read_text().strip(),
        })
result["nvidia_pci_devices"] = devices
binary = shutil.which("nvidia-smi")
if binary is None:
    result["gpu_driver"] = {"status": "nvidia-smi-not-installed"}
else:
    try:
        probe = subprocess.run(
            [binary, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=30, check=False,
        )
        result["gpu_driver"] = {
            "exit_code": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr,
            "status": "available" if probe.returncode == 0 else "driver-probe-failed",
        }
    except subprocess.TimeoutExpired:
        result["gpu_driver"] = {"status": "driver-probe-timeout"}
request = urllib.request.Request(
    "http://169.254.169.254/metadata/instance/compute/storageProfile/imageReference"
    "?api-version=2021-02-01&format=json",
    headers={"Metadata": "true"},
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        result["image_reference"] = json.load(response)
except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
    result["image_reference"] = {"status": "not-verified", "error": str(error)}
result["scientific_qualification"] = "not_run"
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
PY

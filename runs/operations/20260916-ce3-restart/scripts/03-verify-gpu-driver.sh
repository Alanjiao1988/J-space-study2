#!/bin/sh
set -eu
python3 - <<'PY'
import datetime
import json
import subprocess

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=90, check=False,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    passed = (
        result.returncode == 0 and len(rows) == 4
        and all("NVIDIA A100 80GB PCIe" in line and "81920 MiB" in line for line in rows)
    )
    receipt = {
        "returncode": result.returncode, "stdout": result.stdout,
        "stderr": result.stderr, "four_a100_80gb_driver_check_passed": passed,
    }
except subprocess.TimeoutExpired:
    receipt = {"status": "timeout", "four_a100_80gb_driver_check_passed": False}
receipt.update({
    "schema": "wda/gpu-driver-readiness/1",
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "hardware/driver verification only; no scientific qualification or model execution",
})
print(json.dumps(receipt, sort_keys=True))
PY

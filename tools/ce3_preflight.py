"""Read-only Azure CLI preflight. Never logs in, changes configuration, or deploys.

Run only after the operator allows Azure CLI's normal configuration directory
through the local sandbox. A failed CLI command is not retried or bypassed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

SUBSCRIPTION = "d124de35-7837-4ffe-ba2b-2ec1d31477d0"
LOCATION = "chinaeast3"
SIZE = "Standard_NC96ads_A100_v4"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "operations" / "20260916-ce3-restart"
STEPS = (
    ("cloud", ("cloud", "show", "--query", "{name:name,resourceManager:endpoints.resourceManager}")),
    ("account", ("account", "show", "--subscription", SUBSCRIPTION,
                 "--query", "{id:id,state:state,environmentName:environmentName}")),
    ("sku", ("vm", "list-skus", "--subscription", SUBSCRIPTION, "--location", LOCATION,
             "--resource-type", "virtualMachines", "--size", SIZE, "--all",
             "--query", "[].{name:name,family:family,locations:locations,capabilities:capabilities,restrictions:restrictions}")),
    ("usage", ("vm", "list-usage", "--subscription", SUBSCRIPTION, "--location", LOCATION,
               "--query", "[].{name:name.value,currentValue:currentValue,limit:limit}")),
)


class PreflightError(Exception):
    pass


def check_identity(cloud, account) -> None:
    if not isinstance(cloud, dict) or cloud.get("name") != "AzureChinaCloud":
        raise PreflightError("AzureChinaCloud must already be selected; no cloud switch is performed")
    if cloud.get("resourceManager", "").rstrip("/") != "https://management.chinacloudapi.cn":
        raise PreflightError("unexpected Azure China Resource Manager endpoint")
    if (
        not isinstance(account, dict) or account.get("id") != SUBSCRIPTION
        or account.get("state") != "Enabled" or account.get("environmentName") != "AzureChinaCloud"
    ):
        raise PreflightError("the selected enabled Azure China subscription was not verified")


def assess(skus, usage) -> dict:
    if not isinstance(skus, list) or not isinstance(usage, list):
        raise PreflightError("Azure SKU/usage response is not a list")
    matches = [s for s in skus if s.get("name") == SIZE and LOCATION in s.get("locations", [])]
    if len(matches) != 1:
        raise PreflightError("exact CE3 four-A100 SKU not uniquely found")
    sku = matches[0]
    if sku.get("restrictions") != []:
        raise PreflightError("SKU restrictions are present or unavailable; manual review required")
    caps = {c["name"]: c["value"] for c in sku.get("capabilities", [])}
    if caps.get("GPUs") != "4" or caps.get("vCPUs") != "96":
        raise PreflightError("live SKU does not confirm four GPUs and 96 vCPUs")
    family = sku.get("family")
    if not isinstance(family, str) or not family:
        raise PreflightError("SKU family is absent")
    result = {}
    for name in ("cores", family):
        rows = [u for u in usage if str(u.get("name", "")).casefold() == name.casefold()]
        if len(rows) != 1:
            raise PreflightError(f"quota not uniquely available for {name}")
        current, limit = rows[0].get("currentValue"), rows[0].get("limit")
        if type(current) is not int or type(limit) is not int or not 0 <= current <= limit:
            raise PreflightError(f"quota values malformed for {name}")
        available = limit - current
        if available < 96:
            raise PreflightError(f"{name} has only {available} free vCPUs; 96 required")
        result[name] = {"used": current, "limit": limit, "available": available}
    return {"vm_size": SIZE, "family": family, "gpus": 4, "vcpus": 96, "quota": result}


def inspect(output: Path, *, timeout: int = 120, runner=subprocess.run) -> Path:
    az = shutil.which("az")
    if az is None:
        raise PreflightError("Azure CLI is not installed")
    observed = {}
    for name, args in STEPS:
        command = [az, *args, "--output", "json", "--only-show-errors"]
        try:
            response = runner(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PreflightError(f"Azure CLI {name} timed out; not retried") from exc
        if response.returncode != 0:
            # Raw stderr may contain identity/session data. Keep it out of the repo.
            raise PreflightError(
                f"Azure CLI {name} failed (exit {response.returncode}); stopped without retry. "
                "If this is the known .azure sandbox denial, update that policy first."
            )
        try:
            observed[name] = json.loads(response.stdout)
        except json.JSONDecodeError as exc:
            raise PreflightError(f"Azure CLI {name} did not return JSON") from exc
        if name == "cloud" and (
            not isinstance(observed[name], dict) or observed[name].get("name") != "AzureChinaCloud"
        ):
            raise PreflightError("wrong or unverified cloud; no other requests were sent")
        if name == "account":
            check_identity(observed["cloud"], observed["account"])
    result = assess(observed["sku"], observed["usage"])
    receipt = {
        "schema": "wda/azure-cli-preflight/1", "at": datetime.now(timezone.utc).isoformat(),
        "subscription_id": SUBSCRIPTION, "location": LOCATION, "check": result,
        "mode": "read_only", "deployment_submitted": False,
        "live_capacity_guaranteed": False, "price_verified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-azure", action="store_true",
                        help="Execute read-only CLI requests using existing identity; no login or deployment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.inspect_azure:
        print(json.dumps({
            "subscription_id": SUBSCRIPTION, "region": LOCATION, "vm_size": SIZE,
            "commands": [["az", *step] for _, step in STEPS],
            "executed": False,
        }, indent=2))
        return 0
    try:
        path = inspect(args.output_dir)
    except (PreflightError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Read-only preflight recorded: {path}; no deployment submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

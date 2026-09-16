import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("ce3_preflight", Path(__file__).resolve().parents[2] / "tools/ce3_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def inventory():
    return ([{
        "name": preflight.SIZE, "locations": ["chinaeast3"],
        "family": "standardNCADSA100v4Family", "restrictions": [],
        "capabilities": [{"name": "GPUs", "value": "4"}, {"name": "vCPUs", "value": "96"}],
    }], [
        {"name": "cores", "currentValue": 0, "limit": 192},
        {"name": "standardNCADSA100v4Family", "currentValue": 0, "limit": 96},
    ])


def test_fixed_subscription_and_region():
    assert preflight.SUBSCRIPTION == "d124de35-7837-4ffe-ba2b-2ec1d31477d0"
    skus, usage = inventory()
    assert preflight.assess(skus, usage)["gpus"] == 4


@pytest.mark.parametrize("field", ["cores", "standardNCADSA100v4Family"])
def test_both_quota_limits_must_cover_96(field):
    skus, usage = inventory()
    next(row for row in usage if row["name"] == field)["currentValue"] = 97 if field == "cores" else 1
    with pytest.raises(preflight.PreflightError, match="free vCPUs"):
        preflight.assess(skus, usage)


def test_restricted_sku_is_not_treated_as_available():
    skus, usage = inventory()
    skus[0]["restrictions"] = [{"reasonCode": "NotAvailableForSubscription"}]
    with pytest.raises(preflight.PreflightError, match="restrictions"):
        preflight.assess(skus, usage)


def test_no_other_subscription_or_cloud():
    cloud = {"name": "AzureChinaCloud", "resourceManager": "https://management.chinacloudapi.cn/"}
    account = {"id": "wrong", "state": "Enabled", "environmentName": "AzureChinaCloud"}
    with pytest.raises(preflight.PreflightError, match="subscription"):
        preflight.check_identity(cloud, account)


def test_cli_error_is_not_retried_or_logged_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "az")
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="sensitive-example-output")
    with pytest.raises(preflight.PreflightError) as error:
        preflight.inspect(tmp_path / "output", runner=runner)
    assert len(calls) == 1
    assert "sensitive-example-output" not in str(error.value)
    assert not (tmp_path / "output").exists()


def test_default_cli_mode_does_not_execute(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("planning must not call Azure CLI")
    monkeypatch.setattr(preflight, "inspect", fail)
    assert preflight.main([]) == 0
    assert json.loads(capsys.readouterr().out)["executed"] is False


def test_template_has_one_vm_and_no_inbound_allow_rule():
    path = Path(__file__).resolve().parents[2] / "infra/azure/ce3-a100x4.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert len([r for r in value["resources"] if r["type"] == "Microsoft.Compute/virtualMachines"]) == 1
    nsg = next(r for r in value["resources"] if r["type"] == "Microsoft.Network/networkSecurityGroups")
    assert all(rule["properties"]["access"] == "Deny" for rule in nsg["properties"]["securityRules"])
    for key in ("sshPublicKey", "shutdownTimeUtc", "imageVersion"):
        assert "defaultValue" not in value["parameters"][key]

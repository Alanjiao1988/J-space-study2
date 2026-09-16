# CE3 four-A100 host

**Prepared, not deployed.** Only subscription
`d124de35-7837-4ffe-ba2b-2ec1d31477d0` is authorized. The resource-group-scoped
template cannot itself select the subscription; the deployment caller must
verify the subscription before submitting it.

`ce3-a100x4.json` describes one `Standard_NC96ads_A100_v4` in `chinaeast3`,
its VNet, NSG, NIC, outbound public IP, managed OS/data disks, and an Azure
shutdown schedule. These associated resources are not free merely because
the VM is deallocated. No resource-group creation or deployment has been
submitted by preparing these files.

The template uses `auditVmSku` and `guestPublicKey` rather than the portal's
automatically recognized `vmSize`/`sshPublicKey` form fields. This keeps the
explicit SKU whitelist and supplied public key instead of invoking the
unresponsive size picker or proposing a new persisted Azure SSH-key resource.
The ARM hardware profile and public-key properties are unchanged.

## Mandatory preflight

- The operator prefers Azure CLI. `python tools/ce3_preflight.py` prints
  the exact read-only commands without invoking them. Only after normal
  `.azure` directory access is allowed, use `--inspect-azure` to execute
  the subscription/SKU/quota checks. The script does not log in, switch
  subscriptions, redirect logs, or create resources.
- Confirm the selected subscription is enabled and the operator's current
  role permits deployment in the intended resource group.
- Check live CE3 SKU availability and any restrictions for the exact size;
  require at least 96 free regional vCPUs and 96 free family vCPUs.
  Documentation about a VM family is not evidence of quota or capacity.
- Verify API versions and an exact Jammy Gen2 image version in Azure China.
  The template deliberately has no `latest` image default.
- Obtain the subscription's EA price and confirm the chargeable components,
  proposed runtime, stop time and disk-retention policy.
- Supply a guest SSH **public** key. Never reuse the portal password as a
  guest password or place private keys, tokens or login credentials here.
- Validate the template in the selected subscription before deployment.
  Local JSON checks are not Azure deployment validation.

## Access and billing

All inbound traffic, including SSH, is denied by the NSG. The dedicated
Standard public IP supplies outbound connectivity; it does not make SSH
publicly reachable. The intended administration path is Azure VM Run Command.
Do not add public inbound rules merely to work around a control-plane failure.

`shutdownTimeUtc` is mandatory and must be chosen so its next occurrence is
inside the operator-approved runtime. A daily schedule is **not** an exact
cumulative GPU-hour limiter or proof of deallocation. Verify the schedule
resource after deployment, record actual VM power state, and deallocate when
blocked or idle. The earlier proposed 16 GPU-hour window would correspond to
four running hours for this VM; it must not be represented as a confirmed
budget without the operator's agreement.

The data disk detaches rather than being deleted with the VM, preserving
experiment evidence. It continues to incur storage charges. Resource cleanup
and retention must be explicitly recorded.

## Experiment startup

The template does not run a model, install an unpinned driver, fetch private
source, or execute a cloud-init script. Before model work, record:

1. Azure deployment ID, VM ID, actual region/size, driver and four physical
   GPU devices from `nvidia-smi`, plus driver/runtime compatibility.
2. The exact source commit, dependency lock, artifact digests and approved
   engineering or scientific run configuration.
3. A working runtime limit, actual shutdown state and a durable result path.

An engineering canary does not qualify Phase 0. Formal scientific stages
must pass their instrument and seal gates independently of VM creation.
See [the operations report](../../reports/ce3_restart_20260916.md).

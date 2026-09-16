"""Verified adapter to anthropics/jacobian-lens at the registered source revision.

Source verification is offline and imports no model libraries. The exact fitting
adapter is separate from NumPy's synthetic stochastic JVP helper. It is not called
by the Phase-0 engineering canary and does not qualify any scientific intervention.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

from wda.errors import ProtocolViolation

UPSTREAM_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
UPSTREAM_REPOSITORY = "anthropics/jacobian-lens"
REFERENCE_REPOSITORY = "Kameshr/jspace-lens-Qwen7B"
REFERENCE_REVISION = "c03bc9d1157c7cdfaa1940c5a60e6c987d89ad1a"
REFERENCE_FILENAME = "jlens.pt"

# Git blob IDs read from the pinned public tree; verified before importing any jlens code.
_PYTHON_BLOBS = {
    "__init__.py": "564252cb940cebf2109bba4edbda13ab1d2c3f83",
    "_logging.py": "12aaeb5da13bd021ff70d43146b7e06dad3c922e",
    "examples.py": "42b2ecd69a07a59cbfdfff64e6a190aefbf44926",
    "fitting.py": "02deb43a74c4c51436655690bfed55b8c37cc6c8",
    "hf.py": "dc1542cfd3313932040d7490e665dbdc936d9080",
    "hooks.py": "a0c382c58949828da45fb8586aa4496e86aeda81",
    "lens.py": "15ee648edd42816e0c70af15f03a1f20bca0e08b",
    "protocol.py": "053701062fdbb28716ac77d13377f1d7ae114816",
    "vis.py": "1ca8feb64c35306fe62b5ad865a17590f0089175",
}


class QualificationPending(ProtocolViolation):
    """An engineering adapter cannot substitute for missing scientific validation."""


def verify_source(source_root: Path) -> dict:
    package = Path(source_root).resolve() / "jlens"
    names = {p.relative_to(package).as_posix() for p in package.rglob("*.py")}
    if names != set(_PYTHON_BLOBS):
        raise QualificationPending("pinned upstream Python file inventory differs or is absent")
    hashes = {}
    for name, expected in _PYTHON_BLOBS.items():
        content = (package / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != expected:
            raise QualificationPending(f"pinned upstream source mismatch: jlens/{name}")
        hashes[f"jlens/{name}"] = hashlib.sha256(content).hexdigest()
    return {
        "repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION,
        "files_sha256": hashes, "estimator": "exact_row_batched_vjp",
        "dim_batch_default": 8, "save_dtype_required": "float32",
        "source_layers_default": "all blocks strictly below final block",
        "scientific_interventions": "not_qualified",
    }


def load_verified_upstream(source_root: Path):
    verify_source(source_root)
    from packaging.version import Version

    if Version(importlib.metadata.version("transformers")) < Version("5.5"):
        raise QualificationPending("pinned upstream requires transformers>=5.5")
    package = Path(source_root).resolve() / "jlens"
    loaded = [name for name in sys.modules if name == "jlens" or name.startswith("jlens.")]
    if loaded:
        for name in loaded:
            filename = getattr(sys.modules[name], "__file__", None)
            if filename is None or Path(filename).resolve().parent != package:
                raise QualificationPending("an unverified jlens module is already imported")
        return sys.modules["jlens"]
    spec = importlib.util.spec_from_file_location(
        "jlens", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["jlens"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in list(sys.modules):
            if name == "jlens" or name.startswith("jlens."):
                del sys.modules[name]
        raise
    return module


@dataclass(frozen=True)
class ExactFitSpec:
    """Explicit exact-fit parameters; stochastic n_probes is intentionally absent."""

    n_prompts: int
    skip_first: int
    max_seq_len: int
    dim_batch: int = 8

    def __post_init__(self):
        if (
            any(type(v) is not int for v in (
                self.n_prompts, self.skip_first, self.max_seq_len, self.dim_batch
            ))
            or self.n_prompts < 200 or self.skip_first < 16 or self.dim_batch < 1
            or self.max_seq_len <= self.skip_first + 1
        ):
            raise QualificationPending("exact fitting requires n>=200, skip>=16 and valid dimensions")


def fit_exact(upstream, adapter, prompts, spec: ExactFitSpec, *, run):
    """Faithful exact API wrapper; only an independently authorised Phase-A Run may call.

    No resuming, hidden filtering, fp16 saving, or stochastic-estimator substitution.
    The engineering canary never invokes this path.
    """
    from wda.phases import Phase
    from wda.runtime.run import Run
    from wda.governance import seal
    from wda.models import registry

    if not isinstance(run, Run) or run.phase is not Phase.PHASE_A:
        raise QualificationPending("exact fitting requires a separately authorised Phase-A Run")
    seal.require(run.seal_stage)
    registry.resolve("calib_qwen25_7b_instruct", Phase.PHASE_A, require_revision=True)
    if len(prompts) != spec.n_prompts or len(set(prompts)) != len(prompts):
        raise QualificationPending("exact fit needs the declared distinct prompt sample")
    for prompt in prompts:
        encoded = adapter.encode(prompt, max_length=spec.max_seq_len)
        if encoded.shape[-1] <= spec.skip_first + 1:
            raise QualificationPending("exact fit refuses prompts upstream would silently skip")
    for parameter in adapter._hf_model.parameters():
        if parameter.requires_grad or str(parameter.dtype) != "torch.float32":
            raise QualificationPending("exact fitting requires frozen fp32 model weights")
    lens = upstream.fit(
        adapter, list(prompts), source_layers=list(range(adapter.n_layers - 1)),
        target_layer=adapter.n_layers - 1, dim_batch=spec.dim_batch,
        max_seq_len=spec.max_seq_len, skip_first=spec.skip_first,
        checkpoint_path=None, checkpoint_every=None, resume=False,
    )
    if lens.n_prompts != spec.n_prompts:
        raise QualificationPending("upstream effective sample count differs from the declared sample")
    return lens


def save_fp32(lens, path: Path) -> None:
    """Use explicit upstream fp32 serialization, never its fp16 default."""
    import torch

    if Path(path).exists():
        raise FileExistsError("lens output is create-only")
    lens.save(str(path), dtype=torch.float32)


def reference_readout_fixture(*_args, **_kwargs):
    raise QualificationPending(
        "not_qualified: reference readout needs verified tensor schema, estimator and "
        f"normalization compatibility for {REFERENCE_REPOSITORY}@{REFERENCE_REVISION}/"
        f"{REFERENCE_FILENAME}. Its fused-Hutchinson provenance is not the exact "
        "upstream fit estimator; no reference file is downloaded or deserialized here."
    )


def known_intermediate_fixture(*_args, **_kwargs):
    raise QualificationPending(
        "not_qualified: no verified upstream probe-swap interface/donor corpus mapping "
        "is registered. A synthetic donor solved from the unembedding is not evidence "
        "for this positive control. Direct-substitution and matched-norm controls "
        "also require a verified scientific intervention definition."
    )

"""Acquire only the pinned calibration checkpoint; never import or run a model.

Default mode prints a plan. --execute verifies Hub metadata, Git/LFS file digests
and model dimensions, then writes a create-only acquisition receipt. No Azure
credential, Hugging Face token, alternate mirror or remote Python code is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request

REPO = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_BYTES = 18 * 1024**3
SMALL_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "model.safetensors.index.json", "README.md",
}
SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z")
METADATA_URL = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"


class AcquisitionError(Exception):
    pass


class PublicHubRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlparse(newurl)
        host = (url.hostname or "").lower()
        if url.scheme != "https" or url.username or url.password or not (
            host == "huggingface.co"
            or host.endswith((".huggingface.co", ".hf.co", ".cloudfront.net", ".amazonaws.com"))
        ):
            raise AcquisitionError("download redirected outside approved public HTTPS storage")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _json_write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def file_plan(metadata: dict) -> dict:
    if metadata.get("id") != REPO or metadata.get("sha") != REVISION:
        raise AcquisitionError("Hub metadata does not identify the pinned calibration model")
    files = {}
    for row in metadata.get("siblings", []):
        name = row.get("rfilename", "")
        if name not in SMALL_FILES and not SHARD.fullmatch(name):
            continue
        if name in files or type(row.get("size")) is not int or row["size"] <= 0:
            raise AcquisitionError("malformed or duplicate checkpoint file entry")
        if isinstance(row.get("lfs"), dict):
            expected = row["lfs"].get("sha256", "")
            if row["lfs"].get("size") != row["size"] or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise AcquisitionError("LFS size/digest is invalid")
            kind = "sha256"
        else:
            expected = row.get("blobId", "")
            if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
                raise AcquisitionError("Git blob digest is invalid")
            kind = "git-sha1"
        files[name] = {"size": row["size"], "hash_kind": kind, "expected": expected}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json"} <= files.keys():
        raise AcquisitionError("required calibration checkpoint metadata is incomplete")
    if not any(SHARD.fullmatch(name) for name in files):
        raise AcquisitionError("no safetensors weight shards in the pinned checkpoint")
    if sum(row["size"] for row in files.values()) > MAX_BYTES:
        raise AcquisitionError("checkpoint exceeds the bounded calibration download size")
    return files


def verify_file(path: Path, entry: dict) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
        raise AcquisitionError(f"file type or size differs: {path.name}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            sha.update(chunk)
            git.update(chunk)
    actual = sha.hexdigest() if entry["hash_kind"] == "sha256" else git.hexdigest()
    if actual != entry["expected"]:
        raise AcquisitionError(f"checkpoint digest mismatch: {path.name}")
    return {"name": path.name, "size": entry["size"], "sha256": sha.hexdigest()}


def validate_index(cache: Path, files: dict) -> None:
    config = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    if any(config.get(k) != v for k, v in {
        "model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28,
        "vocab_size": 152064,
    }.items()):
        raise AcquisitionError("checkpoint config is not the declared calibration model")
    index = json.loads((cache / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise AcquisitionError("empty or invalid safetensors index")
    referenced = set(mapping.values())
    if referenced != {name for name in files if SHARD.fullmatch(name)}:
        raise AcquisitionError("weight index and pinned Hub shard inventory differ")


def acquire(cache: Path, run_dir: Path, *, timeout_seconds: int = 7200, opener=None) -> dict:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
        raise AcquisitionError("acquisition timeout must be 1..7200 seconds")
    if cache.resolve().is_relative_to(run_dir.resolve()) or run_dir.resolve().is_relative_to(cache.resolve()):
        raise AcquisitionError("cache and receipt directories must be separate")
    if cache.is_symlink() or run_dir.is_symlink():
        raise AcquisitionError("cache/receipt symlinks are refused")
    run_dir.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    opener = opener or urllib.request.build_opener(PublicHubRedirects())
    started = datetime.now(timezone.utc).isoformat()
    _json_write(run_dir / "request.json", {
        "schema": "wda/acquisition-request/1", "repo_id": REPO, "revision": REVISION,
        "started_at": started, "timeout_seconds": timeout_seconds, "max_bytes": MAX_BYTES,
        "model_calls": 0, "kind": "artifact_acquisition_not_model_execution",
    })

    def open_url(url):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AcquisitionError("acquisition deadline exceeded")
        req = urllib.request.Request(url, headers={"User-Agent": "wda-calibration-acquisition/1"})
        return opener.open(req, timeout=min(30, remaining))

    def download(name: str, entry: dict):
        target = cache / name
        if target.exists():
            return {**verify_file(target, entry), "reused_verified_file": True}
        partial = cache / (name + ".partial")
        # A prior partial transfer is not silently discarded or trusted.
        if partial.exists():
            raise AcquisitionError(f"partial transfer exists; inspect before retry: {name}")
        received = 0
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}"
        with open_url(url) as source, partial.open("xb") as dest:
            while chunk := source.read(4 * 1024**2):
                if time.monotonic() >= deadline:
                    raise AcquisitionError("acquisition deadline exceeded")
                received += len(chunk)
                if received > entry["size"]:
                    raise AcquisitionError(f"download exceeds declared file size: {name}")
                dest.write(chunk)
            dest.flush()
            os.fsync(dest.fileno())
        verified = verify_file(partial, entry)
        # The cache-wide exclusive lease serializes acquisition writers. A
        # verified file is published by same-directory rename, without hardlinks.
        if target.exists():
            raise AcquisitionError("another writer created the cache destination")
        partial.rename(target)
        return {**verified, "name": name, "reused_verified_file": False}

    lease = None
    lease_path = cache / ".acquisition.lock"
    try:
        lease = lease_path.open("x", encoding="utf-8")
        lease.write(json.dumps({"run": run_dir.name, "pid": os.getpid(), "started_at": started}))
        lease.flush()
        with open_url(METADATA_URL) as source:
            raw = source.read(4 * 1024**2 + 1)
        if len(raw) > 4 * 1024**2:
            raise AcquisitionError("Hub metadata exceeds the permitted size")
        metadata = json.loads(raw)
        files = file_plan(metadata)
        _json_write(run_dir / "file-plan.json", {
            "repo_id": REPO, "revision": REVISION, "files": files,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        })
        order = ["config.json", "model.safetensors.index.json"]
        order += sorted(set(files) - set(order))
        verified = []
        for name in order:
            artifact = download(name, files[name])
            verified.append(artifact)
            _json_write(run_dir / f"file-{len(verified):02d}.json", artifact)
            print(json.dumps({"event": "file_verified", **artifact}), flush=True)
            if name == "model.safetensors.index.json":
                validate_index(cache, files)
        result = {
            "schema": "wda/calibration-acquisition/1", "status": "verified",
            "repo_id": REPO, "revision": REVISION, "files": verified,
            "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "model_calls": 0, "scientific_qualification": "not_run",
        }
        _json_write(run_dir / "result.json", result)
        return result
    except (AcquisitionError, OSError, ValueError, KeyError, TypeError) as error:
        # Do not persist signed redirect URLs or raw HTTP exception bodies.
        reason = str(error) if isinstance(error, AcquisitionError) else type(error).__name__
        _json_write(run_dir / "failure.json", {
            "schema": "wda/acquisition-failure/1", "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(), "model_calls": 0,
            "complete": False,
        })
        raise AcquisitionError(reason) from error
    finally:
        if lease is not None:
            lease.close()
            lease_path.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path(".wda-model-cache") / REVISION)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"repo_id": REPO, "revision": REVISION, "max_bytes": MAX_BYTES,
                          "model_calls": 0, "executed": False}, indent=2))
        return 0
    if args.run_dir is None:
        parser.error("--execute requires a new --run-dir for durable receipts")
    try:
        result = acquire(args.cache, args.run_dir, timeout_seconds=args.timeout_seconds)
    except (AcquisitionError, OSError) as error:
        print(f"Acquisition failed: {error}")
        return 1
    print(json.dumps({"status": result["status"], "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

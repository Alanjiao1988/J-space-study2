"""Acquisition tests with in-memory fake public responses; no network or weights."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

spec = importlib.util.spec_from_file_location(
    "acquire_calibration", Path(__file__).resolve().parents[2] / "tools/acquire_calibration.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_files():
    shard = "model-00001-of-00001.safetensors"
    files = {
        "config.json": json.dumps({
            "model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28,
            "vocab_size": 152064,
        }).encode(),
        "model.safetensors.index.json": json.dumps({"weight_map": {"toy.weight": shard}}).encode(),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        shard: b"synthetic bytes, never loaded as a model",
    }
    siblings = []
    for name, content in files.items():
        row = {
            "rfilename": name, "size": len(content),
            "blobId": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
        }
        if name.endswith(".safetensors"):
            row["lfs"] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        siblings.append(row)
    return files, {"id": module.REPO, "sha": module.REVISION, "siblings": siblings}


class FakeHub:
    def __init__(self, files, metadata):
        self.files, self.metadata, self.calls = files, metadata, []

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        assert "Authorization" not in request.headers
        if request.full_url == module.METADATA_URL:
            return io.BytesIO(json.dumps(self.metadata).encode())
        return io.BytesIO(self.files[request.full_url.rsplit("/", 1)[-1]])


def test_planning_mode_has_no_network_or_files(monkeypatch, capsys):
    monkeypatch.setattr(module, "acquire", lambda *_a, **_k: pytest.fail("plan executed acquisition"))
    assert module.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["executed"] is False and result["model_calls"] == 0


def test_complete_digest_checked_acquisition(tmp_path):
    files, metadata = fixture_files()
    hub = FakeHub(files, metadata)
    result = module.acquire(tmp_path / "cache", tmp_path / "run", opener=hub)
    assert result["status"] == "verified" and result["model_calls"] == 0
    assert result["scientific_qualification"] == "not_run"
    assert len(result["files"]) == len(files)
    assert not list((tmp_path / "cache").glob("*.partial"))
    assert json.loads((tmp_path / "run/result.json").read_text())["revision"] == module.REVISION


def test_existing_cache_is_verified_not_redownloaded(tmp_path):
    files, metadata = fixture_files()
    cache = tmp_path / "cache"
    module.acquire(cache, tmp_path / "first", opener=FakeHub(files, metadata))
    hub = FakeHub({}, metadata)
    result = module.acquire(cache, tmp_path / "second", opener=hub)
    assert hub.calls == [module.METADATA_URL]
    assert all(row["reused_verified_file"] for row in result["files"])


def test_wrong_model_revision_stops_before_file_download(tmp_path):
    files, metadata = fixture_files()
    metadata["sha"] = "b" * 40
    hub = FakeHub(files, metadata)
    with pytest.raises(module.AcquisitionError, match="pinned calibration"):
        module.acquire(tmp_path / "cache", tmp_path / "run", opener=hub)
    assert hub.calls == [module.METADATA_URL]
    assert (tmp_path / "run/failure.json").exists()
    assert not (tmp_path / "run/result.json").exists()


def test_corrupt_download_is_not_published(tmp_path):
    files, metadata = fixture_files()
    name = "model-00001-of-00001.safetensors"
    files[name] = b"x" * len(files[name])
    with pytest.raises(module.AcquisitionError, match="digest mismatch"):
        module.acquire(tmp_path / "cache", tmp_path / "run", opener=FakeHub(files, metadata))
    assert not (tmp_path / "cache" / name).exists()
    assert (tmp_path / "cache" / (name + ".partial")).exists()


def test_partial_cache_requires_explicit_recovery(tmp_path):
    files, metadata = fixture_files()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "config.json.partial").write_bytes(b"partial")
    with pytest.raises(module.AcquisitionError, match="partial transfer"):
        module.acquire(cache, tmp_path / "run", opener=FakeHub(files, metadata))
    assert (cache / "config.json.partial").read_bytes() == b"partial"


def test_receipt_directory_is_create_only(tmp_path):
    files, metadata = fixture_files()
    run = tmp_path / "run"
    module.acquire(tmp_path / "cache", run, opener=FakeHub(files, metadata))
    with pytest.raises(FileExistsError):
        module.acquire(tmp_path / "cache", run, opener=FakeHub(files, metadata))


def test_cache_lease_blocks_concurrent_acquisition(tmp_path):
    files, metadata = fixture_files()
    cache = tmp_path / "cache"
    cache.mkdir()
    lease = cache / ".acquisition.lock"
    lease.write_text("another running acquisition", encoding="utf-8")
    with pytest.raises(module.AcquisitionError, match="FileExistsError"):
        module.acquire(cache, tmp_path / "run", opener=FakeHub(files, metadata))
    assert lease.read_text(encoding="utf-8") == "another running acquisition"


def test_redirect_rejects_non_public_or_non_https_targets():
    handler = module.PublicHubRedirects()
    req = module.urllib.request.Request("https://huggingface.co/model")
    for destination in ("http://huggingface.co/file", "https://localhost/file", "https://evil.example/file"):
        with pytest.raises(module.AcquisitionError, match="approved public HTTPS"):
            handler.redirect_request(req, None, 302, "", {}, destination)


def test_dimension_mismatch_cannot_be_used_as_calibration(tmp_path):
    files, metadata = fixture_files()
    config = json.loads(files["config.json"])
    config["num_hidden_layers"] = 48
    files["config.json"] = json.dumps(config).encode()
    row = next(row for row in metadata["siblings"] if row["rfilename"] == "config.json")
    row["size"] = len(files["config.json"])
    row["blobId"] = hashlib.sha1(f"blob {row['size']}\0".encode() + files["config.json"]).hexdigest()
    with pytest.raises(module.AcquisitionError, match="declared calibration model"):
        module.acquire(tmp_path / "cache", tmp_path / "run", opener=FakeHub(files, metadata))

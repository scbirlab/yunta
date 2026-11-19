"""Single-flight, FileLock-guarded weight downloads (patch 8). No network needed."""

import io
import multiprocessing as mp
import os
import signal
import sys
import tarfile
import threading
import time
import types
from pathlib import Path

import pytest

filelock = pytest.importorskip("filelock")

from yunta import weights
from yunta.weights import DownloadLockTimeout, fetch_once

FORK = mp.get_context("fork")
PAYLOAD = b"0123456789" * 50_000  # 500 kB


def _count(path: Path) -> int:
    return len(path.read_text().split()) if path.exists() else 0


def _slow_producer(counter: Path, pause=0.4, payload=PAYLOAD):
    def produce(tmp: Path):
        fd = os.open(counter, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # atomic append across processes
        os.write(fd, b"x\n"); os.close(fd)
        with open(tmp, "wb") as f:
            for i in range(0, len(payload), len(payload) // 5):
                f.write(payload[i:i + len(payload) // 5])
                f.flush()
                time.sleep(pause / 5)
    return produce


# ------------------------------------------------------------------ fetch_once semantics

def test_threads_download_once_and_all_get_the_complete_file(tmp_path):
    target, counter = tmp_path / "w" / "params.npz", tmp_path / "count"
    produce = _slow_producer(counter)
    results, errors = [], []

    def worker():
        try:
            results.append(fetch_once(target, produce).read_bytes())
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors
    assert _count(counter) == 1
    assert len(results) == 8 and all(r == PAYLOAD for r in results)


def _proc_worker(target, counter, out):
    out.put(Path(fetch_once(target, _slow_producer(counter))).stat().st_size)


def test_processes_download_once(tmp_path):
    target, counter = tmp_path / "shared" / "model.bin", tmp_path / "count"
    q = FORK.Queue()
    procs = [FORK.Process(target=_proc_worker, args=(target, counter, q)) for _ in range(6)]
    [p.start() for p in procs]; [p.join(60) for p in procs]
    assert all(p.exitcode == 0 for p in procs)
    assert sorted(q.get() for _ in procs) == [len(PAYLOAD)] * 6
    assert _count(counter) == 1
    assert not list(target.parent.glob(".*.part*"))  # no temporary files left behind


def test_readers_never_see_a_partial_file(tmp_path):
    """The file appears only by atomic rename, so an unlocked reader sees all or nothing."""
    target, counter = tmp_path / "params.npz", tmp_path / "count"
    seen, stop = set(), threading.Event()

    def poll():
        while not stop.is_set():
            try:
                seen.add(target.stat().st_size)
            except FileNotFoundError:
                pass

    t = threading.Thread(target=poll); t.start()
    fetch_once(target, _slow_producer(counter, pause=0.5))
    time.sleep(0.05); stop.set(); t.join()
    assert seen == {len(PAYLOAD)}


def test_existing_valid_file_is_returned_without_locking_or_producing(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"ready")

    def never(_):  # pragma: no cover
        raise AssertionError("producer must not run")

    assert fetch_once(target, never) == target
    assert not (tmp_path / "x.bin.lock").exists()  # the fast path takes no lock


def test_failed_download_leaves_nothing_and_releases_the_lock(tmp_path):
    target = tmp_path / "x.bin"

    def boom(tmp):
        tmp.write_bytes(b"half")
        raise RuntimeError("network died")

    with pytest.raises(RuntimeError, match="network died"):
        fetch_once(target, boom)
    assert not target.exists() and not list(tmp_path.glob(".*.part*"))
    assert fetch_once(target, lambda t: t.write_bytes(b"ok")).read_bytes() == b"ok"  # lock was released


def test_validate_replaces_a_bad_existing_file_and_rejects_a_bad_download(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"")  # e.g. truncated by an older, non-atomic downloader
    nonempty = lambda p: p.stat().st_size > 0
    assert fetch_once(target, lambda t: t.write_bytes(b"good"), validate=nonempty).read_bytes() == b"good"
    bad = tmp_path / "y.bin"
    with pytest.raises(ValueError, match="failed validation"):
        fetch_once(bad, lambda t: t.write_bytes(b""), validate=nonempty)
    assert not bad.exists()


def test_producer_that_writes_nothing_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch_once(tmp_path / "x.bin", lambda t: None)


def test_stale_part_file_from_a_dead_worker_does_not_block(tmp_path):
    target = tmp_path / "x.bin"
    (tmp_path / f".x.bin.otherhost.99999.part").write_bytes(b"junk")
    assert fetch_once(target, lambda t: t.write_bytes(b"fresh")).read_bytes() == b"fresh"


# ------------------------------------------------------------------ lock behaviour

def _hold_lock_forever(target, ready):
    with weights.download_lock(target):
        ready.set()
        time.sleep(600)


def test_lock_held_by_killed_process_is_released_by_the_kernel(tmp_path):
    target = tmp_path / "x.bin"
    ready = FORK.Event()
    p = FORK.Process(target=_hold_lock_forever, args=(target, ready))
    p.start(); assert ready.wait(20)
    # While it lives, a short wait times out with a helpful error...
    with pytest.raises(DownloadLockTimeout, match="Timed out"):
        fetch_once(target, lambda t: t.write_bytes(b"x"), timeout=0.3)
    # ...and once it is killed (no cleanup runs), the next caller proceeds promptly.
    os.kill(p.pid, signal.SIGKILL); p.join(10)
    t0 = time.time()
    assert fetch_once(target, lambda t: t.write_bytes(b"x"), timeout=10).read_bytes() == b"x"
    assert time.time() - t0 < 5


def test_waiting_worker_logs_and_reuses_the_result(tmp_path, capsys):
    target, counter = tmp_path / "x.bin", tmp_path / "count"
    ready = FORK.Event()

    def hold():
        with weights.download_lock(target):
            ready.set(); time.sleep(0.8)
            target.with_name("x.bin").write_bytes(b"done-by-holder")

    p = FORK.Process(target=hold); p.start(); assert ready.wait(20)
    out = fetch_once(target, _slow_producer(counter))
    p.join(10)
    assert out.read_bytes() == b"done-by-holder"  # re-checked after acquiring the lock
    assert _count(counter) == 0  # we never downloaded
    assert "waiting for it" in capsys.readouterr().err


def test_soft_lock_type_and_bad_lock_type(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNTA_LOCK_TYPE", "soft")
    assert fetch_once(tmp_path / "a.bin", lambda t: t.write_bytes(b"1")).read_bytes() == b"1"
    assert not (tmp_path / "a.bin.lock").exists()  # SoftFileLock removes its lock file on release
    monkeypatch.setenv("YUNTA_LOCK_TYPE", "nfs-magic")
    with pytest.raises(ValueError, match="YUNTA_LOCK_TYPE"):
        fetch_once(tmp_path / "b.bin", lambda t: t.write_bytes(b"1"))


def test_timeout_env_var(tmp_path, monkeypatch):
    target = tmp_path / "x.bin"
    ready = FORK.Event()
    p = FORK.Process(target=_hold_lock_forever, args=(target, ready)); p.start(); assert ready.wait(20)
    monkeypatch.setenv("YUNTA_DOWNLOAD_TIMEOUT", "0.2")
    try:
        with pytest.raises(DownloadLockTimeout):
            fetch_once(target, lambda t: t.write_bytes(b"x"))
    finally:
        p.kill(); p.join(10)


# ------------------------------------------------------------------ AF2 weights (fake archive, no network)

def _fake_archive(tmp: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in {
            "./params_model_1_ptm.npz": b"NPZ-1" * 1000,
            "./params_model_3_ptm.npz": b"NPZ-3" * 1000,
            "LICENSE": b"CC-BY",
        }.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _patch_stream(monkeypatch, counter: Path):
    archive = _fake_archive(counter)

    def fake_stream(url, destination):
        fd = os.open(counter, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(fd, b"x\n"); os.close(fd)
        time.sleep(0.3)
        Path(destination).write_bytes(archive)

    monkeypatch.setattr(weights, "_stream_to", fake_stream)


def _af2_worker(home, model, q):
    q.put(weights.get_model_weights(model, path=home))


def test_af2_weights_parallel_workers_share_one_download(tmp_path, monkeypatch):
    counter = tmp_path / "count"
    _patch_stream(monkeypatch, counter)
    q = FORK.Queue()
    procs = [FORK.Process(target=_af2_worker, args=(str(tmp_path), "model_1_ptm", q)) for _ in range(5)]
    [p.start() for p in procs]; [p.join(60) for p in procs]
    assert all(p.exitcode == 0 for p in procs)
    dirs = {q.get() for _ in procs}
    assert len(dirs) == 1
    params = Path(dirs.pop()) / "params"
    assert (params / "params_model_1_ptm.npz").read_bytes() == b"NPZ-1" * 1000
    assert (params / "LICENSE").read_bytes() == b"CC-BY"
    assert _count(counter) == 1
    assert sorted(p.name for p in params.iterdir() if not p.name.endswith(".lock")) == [
        "LICENSE", "params_model_1_ptm.npz"]  # no archive or .part debris


def test_af2_weights_other_model_triggers_its_own_download(tmp_path, monkeypatch):
    counter = tmp_path / "count"
    _patch_stream(monkeypatch, counter)
    weights.get_model_weights("model_1_ptm", path=str(tmp_path))
    weights.get_model_weights("model_1_ptm", path=str(tmp_path))  # cached
    assert _count(counter) == 1
    root = weights.get_model_weights("model_3_ptm", path=str(tmp_path))
    assert _count(counter) == 2
    assert (Path(root) / "params" / "params_model_3_ptm.npz").read_bytes() == b"NPZ-3" * 1000


def test_af2_missing_member_fails_cleanly(tmp_path, monkeypatch):
    _patch_stream(monkeypatch, tmp_path / "count")
    with pytest.raises(FileNotFoundError, match="params_model_9_ptm"):
        weights.get_model_weights("model_9_ptm", path=str(tmp_path))
    assert not list((tmp_path / ".weights" / "af2-yunta" / "params").glob(".*"))


def test_extract_member_ignores_member_paths(tmp_path):
    """A hostile member name like ../../evil can never write outside the destination."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("../../evil.npz"); info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    archive = tmp_path / "a.tar"; archive.write_bytes(buf.getvalue())
    dest = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        weights._extract_member(archive, "params_x.npz", dest)
    assert not (tmp_path.parent / "evil.npz").exists() and not dest.exists()


# ------------------------------------------------------------------ OpenFold3 checkpoint (fake openfold3 module)

def _fake_openfold3(monkeypatch, cache: Path, counter: Path):
    params = types.ModuleType("openfold3.entry_points.parameters")
    params.DEFAULT_CACHE_PATH = cache
    params.OPENFOLD_MODEL_CHECKPOINT_REGISTRY = {"m": types.SimpleNamespace(file_name="m.pt")}
    params.get_default_checkpoint_dir = lambda cache_path: Path(cache_path) / "checkpoints"

    def download_model_parameters(dest, name, skip_confirmation=False):
        fd = os.open(counter, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(fd, b"x\n"); os.close(fd)
        out = Path(dest) / "m.pt"  # like the real one: written straight to its final name
        with open(out, "wb") as f:
            for _ in range(5):
                f.write(PAYLOAD[:100_000]); f.flush(); time.sleep(0.1)

    params.download_model_parameters = download_model_parameters
    for name, mod in {
        "openfold3": types.ModuleType("openfold3"),
        "openfold3.entry_points": types.ModuleType("openfold3.entry_points"),
        "openfold3.entry_points.parameters": params,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def _of3_worker(q):
    from yunta.interactions.of3.run import ensure_checkpoint
    q.put(str(ensure_checkpoint("m")))


def test_of3_checkpoint_parallel_workers_download_once_atomically(tmp_path, monkeypatch):
    counter, cache = tmp_path / "count", tmp_path / "cache"
    monkeypatch.setenv("OPENFOLD_CACHE", str(cache))
    _fake_openfold3(monkeypatch, cache, counter)
    q = FORK.Queue()
    procs = [FORK.Process(target=_of3_worker, args=(q,)) for _ in range(5)]
    [p.start() for p in procs]; [p.join(60) for p in procs]
    assert all(p.exitcode == 0 for p in procs)
    paths = {q.get() for _ in procs}
    assert len(paths) == 1
    assert Path(paths.pop()).stat().st_size == 500_000
    assert _count(counter) == 1
    left = sorted(p.name for p in (cache / "checkpoints").iterdir() if not p.name.endswith(".lock"))
    assert left == ["m.pt"]  # staging directory and part files cleaned up


def test_of3_explicit_checkpoint_bypasses_download_and_locking(tmp_path, monkeypatch):
    from yunta.interactions.of3.run import ensure_checkpoint
    ckpt = tmp_path / "mine.pt"; ckpt.write_bytes(b"w")
    assert ensure_checkpoint(checkpoint=ckpt) == ckpt
    assert not list(tmp_path.glob("*.lock"))
    with pytest.raises(FileNotFoundError):
        ensure_checkpoint(checkpoint=tmp_path / "missing.pt")

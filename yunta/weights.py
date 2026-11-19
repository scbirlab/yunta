"""Model weights: cached downloads that are safe to run from many processes at once.

Several workers (a job array, ``nf-ggi`` tasks, ``yunta coevo`` launched in parallel) often
share one cache directory on a networked filesystem. Without coordination each of them sees
"weights missing" and downloads several GB at once, and a worker that finds a
half-written file can load it. :func:`fetch_once` makes the download *single-flight*:

1. **Fast path, no lock.** If the target exists (and passes ``validate``), return it. This
   is safe because files only ever appear by an atomic rename of a completed download.
2. **Lock.** Otherwise take an exclusive :class:`filelock.FileLock` next to the target.
   Waiting workers log that they are waiting rather than starting their own download.
3. **Re-check under the lock** (double-checked locking). Whoever held the lock before may
   have finished the job; then there is nothing to do.
4. **Download to a private temporary path** in the same directory (so the final rename
   cannot cross filesystems), validate, then ``os.replace`` it into place.

Locks are released by the kernel when a process dies, so a killed download does not wedge
the others; leftover ``*.part*`` files are private to their writer and ignored. If the
filesystem does not support ``flock`` (some Lustre/GPFS mounts), set
``YUNTA_LOCK_TYPE=soft`` to use an exclusive-create lock file instead; after a hard crash a
soft lock must be removed by hand (or waited out with ``YUNTA_DOWNLOAD_TIMEOUT``).

Environment variables
---------------------
``YUNTA_DOWNLOAD_TIMEOUT``
    Seconds to wait for another process's download (default: no limit).
``YUNTA_LOCK_TYPE``
    ``flock`` (default) or ``soft``.
"""

from typing import Callable, Optional, Union

import os
import socket
import tarfile
from contextlib import contextmanager
from pathlib import Path

from carabiner import print_err

_WEIGHTS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
_TIMEOUT_ENV = "YUNTA_DOWNLOAD_TIMEOUT"
_LOCK_TYPE_ENV = "YUNTA_LOCK_TYPE"
_CONNECT_TIMEOUT, _READ_TIMEOUT = 30, 300


class DownloadLockTimeout(TimeoutError):
    """Another process held the download lock for longer than the allowed wait."""


def _timeout(timeout: Optional[float]) -> float:
    if timeout is None:
        value = os.environ.get(_TIMEOUT_ENV)
        timeout = float(value) if value else -1
    return timeout


def _make_lock(lock_path: Path, timeout: float):
    import filelock
    kind = os.environ.get(_LOCK_TYPE_ENV, "flock").lower()
    if kind not in ("flock", "soft"):
        raise ValueError(f"{_LOCK_TYPE_ENV} must be 'flock' or 'soft', not {kind!r}")
    cls = filelock.SoftFileLock if kind == "soft" else filelock.FileLock
    return cls(str(lock_path), timeout=timeout)


@contextmanager
def download_lock(
    target: Union[str, os.PathLike],
    timeout: Optional[float] = None,
    poll_interval: float = 0.1,
):
    """Hold the exclusive lock that guards downloads of ``target``.

    Logs once if the lock is contended. Raises :class:`DownloadLockTimeout` when
    ``timeout`` (seconds; default ``$YUNTA_DOWNLOAD_TIMEOUT``, else unlimited) elapses.
    """
    import filelock
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    timeout = _timeout(timeout)
    lock = _make_lock(lock_path, timeout=0)
    try:
        lock.acquire(poll_interval=poll_interval)
    except filelock.Timeout:
        print_err(f"[INFO] Another process is downloading {target.name}; waiting for it ({lock_path})...")
        lock = _make_lock(lock_path, timeout=timeout)
        try:
            lock.acquire(poll_interval=poll_interval)
        except filelock.Timeout as e:
            raise DownloadLockTimeout(
                f"Timed out after {timeout:g} s waiting for the lock on {target} ({lock_path}). "
                f"If no other process is downloading, delete the lock file."
            ) from e
    try:
        yield lock_path
    finally:
        lock.release()


def _private_part(target: Path) -> Path:
    """A temporary path only this process writes, in the target's directory."""
    return target.with_name(f".{target.name}.{socket.gethostname()}.{os.getpid()}.part")


def fetch_once(
    target: Union[str, os.PathLike],
    producer: Callable[[Path], None],
    validate: Optional[Callable[[Path], bool]] = None,
    timeout: Optional[float] = None,
) -> Path:
    """Make ``target`` exist, running ``producer`` in at most one process at a time.

    Parameters
    ----------
    target
        Final path of the cached file.
    producer
        ``producer(tmp_path)`` writes the complete file at ``tmp_path`` (a private path
        in the same directory). It is only called when ``target`` is absent or invalid, and
        never concurrently for the same ``target``.
    validate
        Optional check that an existing file is usable (e.g. non-empty). A file that fails
        is replaced.
    timeout
        Seconds to wait for another process; default ``$YUNTA_DOWNLOAD_TIMEOUT``.

    Returns
    -------
    Path
        ``target``.
    """
    target = Path(target)

    def _ok() -> bool:
        return target.is_file() and (validate is None or validate(target))

    if _ok():  # fast path: files only ever appear complete (atomic rename)
        return target
    with download_lock(target, timeout=timeout):
        if _ok():  # another process finished while we waited
            return target
        part = _private_part(target)
        try:
            part.unlink(missing_ok=True)
            producer(part)
            if not part.is_file():
                raise FileNotFoundError(f"The download for {target.name} did not produce {part}")
            if validate is not None and not validate(part):
                raise ValueError(f"Downloaded file for {target.name} failed validation")
            os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)
    return target


def _stream_to(url: str, destination: Path) -> None:
    import requests
    from tqdm.auto import tqdm
    with requests.get(url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8 * 1024**2)):
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())


def _extract_member(archive: Path, member: str, destination: Path) -> None:
    """Extract one named file from a tar into ``destination`` (never trusting member paths)."""
    with tarfile.open(archive) as tar:
        names = {m.name.lstrip("./"): m for m in tar.getmembers() if m.isfile()}
        wanted = names.get(member.lstrip("./"))
        if wanted is None:
            raise FileNotFoundError(f"{member} not found in {archive}")
        src = tar.extractfile(wanted)
        with open(destination, "wb") as out:
            while chunk := src.read(8 * 1024**2):
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())


def get_model_weights(
    model_name: Optional[str] = None,
    path: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:

    """Get the AlphaFold2 weights directory, downloading on first use.

    Weights live in ``path/.weights/af2-yunta/params`` (``path`` defaults to the home
    directory). Concurrent callers share one download per model (see :func:`fetch_once`).
    The ~3.5 GB archive is streamed to a private temporary file, the requested model's
    parameters are extracted from it, and the archive is deleted.

    Returns the parent of ``params``, which is what AF2 expects.
    """
    if path is None:
        path = os.path.expanduser("~")
    if model_name is None:
        model_name = "model_1_ptm" #"model_1"

    weight_dir = Path(os.path.realpath(path)) / ".weights" / "af2-yunta" / "params"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_file = weight_dir / f"params_{model_name}.npz"

    def produce(tmp: Path) -> None:
        print_err(f"[INFO]  Weights file {weight_file} does not exist. Downloading...")
        archive = tmp.with_name(tmp.name + ".tar")
        try:
            _stream_to(_WEIGHTS_URL, archive)
            print_err(f"[INFO]  Extracting {weight_file.name} from the archive...")
            _extract_member(archive, weight_file.name, tmp)
            license_file = weight_dir / "LICENSE"
            if not license_file.exists():  # small; identical whoever writes it, so replace is safe
                license_tmp = tmp.with_name(tmp.name + ".license")
                try:
                    _extract_member(archive, "LICENSE", license_tmp)
                    os.replace(license_tmp, license_file)
                except FileNotFoundError:
                    pass  # the licence text is optional
                finally:
                    license_tmp.unlink(missing_ok=True)
        finally:
            archive.unlink(missing_ok=True)

    fetch_once(weight_file, produce, validate=lambda p: p.stat().st_size > 0, timeout=timeout)

    print_err(f"[INFO] Model weights located at {weight_dir}.")
    return str(weight_dir.parent)  # AF2 expects the parent dir of "params" dir


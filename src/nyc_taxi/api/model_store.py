"""Thread-safe holder for the served LightGBM booster, with optional MLflow alias hot-swap."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nyc_taxi.model import load_model_dir
from nyc_taxi.registry import download_alias, get_alias_version

if TYPE_CHECKING:
    import lightgbm as lgb

    from nyc_taxi.api.metrics import ApiMetrics

log = logging.getLogger(__name__)


def is_remote_tracking(uri: str | None) -> bool:
    """True when the MLflow tracking URI is an http(s) server."""
    return uri is not None and uri.startswith(("http://", "https://"))


class ModelStore:
    """Holds the current (booster, meta) pair behind a lock and swaps it atomically."""

    def __init__(self, cache_dir: Path | str = Path("models/cache")) -> None:
        self._cache_dir = Path(cache_dir)
        self._lock = threading.Lock()
        self._booster: lgb.Booster | None = None
        self._meta: dict[str, Any] | None = None
        self._metrics: ApiMetrics | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def bind_metrics(self, metrics: ApiMetrics) -> None:
        """Attach the app's metrics and publish the already-loaded version, if any."""
        self._metrics = metrics
        version = self.version
        if version is not None:
            metrics.model_info.clear()
            metrics.model_info.labels(version=version).set(1)

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._booster is not None

    @property
    def version(self) -> str | None:
        with self._lock:
            if self._meta is None:
                return None
            return str(self._meta.get("model_version"))

    def current(self) -> tuple[lgb.Booster, dict[str, Any]] | None:
        """Return the served (booster, meta), or None when nothing is loaded."""
        with self._lock:
            if self._booster is None or self._meta is None:
                return None
            return self._booster, self._meta

    def reload(self, model_dir: Path | str) -> str:
        """Load a model directory and swap it in. Returns the new version."""
        booster, meta = load_model_dir(Path(model_dir))
        with self._lock:
            self._booster = booster
            self._meta = meta
        version = str(meta.get("model_version"))
        if self._metrics is not None:
            self._metrics.model_reloads.inc()
            self._metrics.model_info.clear()
            self._metrics.model_info.labels(version=version).set(1)
        log.info("serving model version %s from %s", version, model_dir)
        return version

    def sync_alias(self, alias: str, tracking_uri: str) -> bool:
        """Download the alias target if its version differs from the served one.

        Returns True when a swap happened.
        """
        remote = get_alias_version(alias, tracking_uri)
        if remote is None or str(remote) == self.version:
            return False
        dst = self._cache_dir / f"{alias}-{remote}"
        path = download_alias(alias, dst, tracking_uri)
        self.reload(path)
        return True

    def start_polling(self, alias: str, tracking_uri: str, interval_seconds: float) -> None:
        """Start a daemon thread that calls sync_alias every interval. Idempotent."""
        if self._thread is not None:
            return

        def _loop() -> None:
            while not self._stop.wait(interval_seconds):
                try:
                    self.sync_alias(alias, tracking_uri)
                except Exception:
                    log.exception("model poll failed; keeping current model")

        self._thread = threading.Thread(target=_loop, name="model-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the poll thread if running."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

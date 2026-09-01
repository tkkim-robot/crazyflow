import os

# Plotting examples run as assertions, not interactive desktop sessions.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["SCIPY_ARRAY_API"] = "1"
# We need multiple devices for sharding tests
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import pytest

# Leave JAX's persistent compilation cache disabled unless the caller requests an isolated cache.
# Forcing every tiny test compilation into one shared cache created thousands of entries and could
# abort long suites inside jaxlib. Reproduction runs can opt in with a fresh directory while
# retaining JAX's normal size and compile-time thresholds.
jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
if jax_cache_dir:
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
# Do not enable XLA caches, crashes PyTest
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


def available_backends() -> list[str]:
    """Return list of available JAX backends."""
    backends = []
    for backend in ["tpu", "gpu", "cpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


@pytest.fixture
def device() -> str:
    """Return GPU device if available, otherwise CPU."""
    if "gpu" in available_backends():
        return "gpu"
    return "cpu"


# Marker for conditional skip in headless environments
skip_if_headless = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None,
    reason="DISPLAY is not set, skipping test in headless environment",
)

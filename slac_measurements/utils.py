import warnings
from typing import Annotated
import logging
import numpy as np
from pydantic import BeforeValidator
import time

from slac_devices import Device

from slac_timing import Buffer


def calculate_statistics(data: np.ndarray, name):
    return {
        f"{name}_mean": np.mean(data),
        f"{name}_std": np.std(data),
        f"{name}_q05": np.quantile(data, 0.05),
        f"{name}_q95": np.quantile(data, 0.95),
    }


def ensure_numpy_array(v):
    return v if isinstance(v, np.ndarray) else np.array(v)


def collect_with_size_check(
    device: Device,
    collector_func: str,
    buffer: Buffer,
    logger: logging.Logger | None,
    max_retries: int = 3,
    delay: float = 3,
):
    """
    Deprecated: Use buffer.get(pv, retries=N, retry_delay=D) instead.

    Collects data using the provided function and checks its size.
    Retries collection if the data size does not match the expected points.
    """
    warnings.warn(
        "collect_with_size_check is deprecated. "
        "Use buffer.get(pv, retries=N) or device.method(buffer, retries=N) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    method = getattr(device, collector_func)
    for attempt in range(max_retries):
        data = method(buffer)
        size = len(data) if data is not None else 0
        expected_points = buffer.n_measurements

        if size == expected_points:
            return data

        if logger is not None:
            logger.warning(
                "Data size mismatch for %s %s: expected %d, got %d. Retrying (%d/%d)...",
                device.name,
                collector_func,
                expected_points,
                size,
                attempt + 1,
                max_retries,
            )
        else:
            print(
                f"Warning: Data size mismatch for {device.name} {collector_func}: "
                f"expected {expected_points}, got {size}. Retrying ({attempt + 1}/{max_retries})..."
            )
        if delay > 0:
            time.sleep(delay)

    raise RuntimeError(
        f"Unable to collect complete {collector_func} data for {device.name}. "
        f"Expected {expected_points} points but retrieved {size} after {max_retries} attempts."
    )


def wait_until(condition, timeout=10, period=0.1) -> bool:
    # Returns True if condition met within timeout
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return True
        time.sleep(period)
    return False


NDArrayAnnotatedType = Annotated[np.ndarray, BeforeValidator(ensure_numpy_array)]

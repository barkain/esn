# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Global subprocess concurrency limiter for batched execution."""

import os
import threading
from contextlib import contextmanager

# Limit concurrent subprocesses to CPU count to prevent thrashing
_MAX_CONCURRENT = os.cpu_count() or 4
_semaphore = threading.Semaphore(_MAX_CONCURRENT)


@contextmanager
def subprocess_slot():
    """Acquire a subprocess execution slot. Use as context manager."""
    _semaphore.acquire()
    try:
        yield
    finally:
        _semaphore.release()

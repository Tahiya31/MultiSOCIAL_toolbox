"""Compatibility layer for ``from backports import tarfile`` imports."""

from tarfile import *  # noqa: F401,F403
import tarfile as _stdlib_tarfile

data_filter = _stdlib_tarfile.data_filter

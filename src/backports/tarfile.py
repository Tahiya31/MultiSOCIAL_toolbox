"""Compatibility layer for ``from backports import tarfile`` imports."""

from tarfile import *  # noqa: F401,F403
import tarfile as _stdlib_tarfile


def _identity_filter(member, _dest_path):
    return member


# Python 3.10 builds do not consistently expose the newer tarfile filter
# helpers across environments, but jaraco.context expects at least
# ``data_filter`` to exist on ``backports.tarfile``.
data_filter = getattr(_stdlib_tarfile, "data_filter", _identity_filter)
tar_filter = getattr(_stdlib_tarfile, "tar_filter", _identity_filter)
fully_trusted_filter = getattr(_stdlib_tarfile, "fully_trusted_filter", _identity_filter)

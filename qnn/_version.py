import sys

if sys.version_info < (3, 6, 0):
    # coverage: ignore
    raise SystemError("qnn requires at least Python 3.6")

__version__ = "0.1.dev"
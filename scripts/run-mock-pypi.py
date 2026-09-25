#!/usr/bin/env python3
"""Start the pypiserver-based mock index used for SDK release rehearsal.

pypiserver vendors Bottle, whose multipart parser caps both buffered request
parts and buffered bodies at ``MEMFILE_MAX`` (100 KiB). SDK distributions are
larger, so uploads fail with ``MultipartError: Memory limit reached``. Raise
the cap before handing control to pypiserver's CLI.
"""

import sys

from pypiserver import bottle

MAX_REQUEST_BYTES = 512 * 1024 * 1024

bottle.BaseRequest.MEMFILE_MAX = MAX_REQUEST_BYTES


def main() -> None:
    from pypiserver.__main__ import main as pypiserver_main

    pypiserver_main(sys.argv[1:])


if __name__ == "__main__":
    main()

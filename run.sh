#!/bin/bash
# Start the tool. The first start on a machine runs setup.sh.
#   ./run.sh                 asks for the bucket(s) and runs
#   ./run.sh --help          all options
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
packages=$(cat requirements/*.txt | sha256sum | cut -d' ' -f1)
if [ ! -f .venv/install-mode ]; then
    echo "First start on this machine: installing."
    ./setup.sh
elif [ "$(cat .venv/requirements.sum 2>/dev/null)" != "$packages" ]; then
    # The package lists changed (for example after "git pull"): install what is new.
    echo "Package lists changed: updating .venv."
    if [ "$(cat .venv/install-mode)" = "gpu" ]; then extra=gpu-cuda12; else extra=cpu; fi
    .venv/bin/python -m pip --disable-pip-version-check install -q \
        -r requirements/base.txt -r "requirements/$extra.txt"
    echo "$packages" > .venv/requirements.sum
fi
exec .venv/bin/python app/s3_ocr.py "$@"

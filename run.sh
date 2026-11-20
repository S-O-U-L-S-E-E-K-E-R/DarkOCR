#!/bin/bash
# Start the tool. The first start on a machine runs setup.sh.
#   ./run.sh                 asks for the bucket(s) and runs
#   ./run.sh --help          all options
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -f .venv/install-mode ]; then
    echo "First start on this machine: installing."
    ./setup.sh
fi
exec .venv/bin/python app/s3_ocr.py "$@"

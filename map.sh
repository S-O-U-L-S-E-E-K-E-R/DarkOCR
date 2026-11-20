#!/bin/bash
# Rebuild the GPS map for one or more results files:  ./map.sh results/<bucket>.csv
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
exec "$here/.venv/bin/python" "$here/app/gps_map.py" "$@"

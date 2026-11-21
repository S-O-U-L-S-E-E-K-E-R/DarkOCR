#!/bin/bash
# Install everything this tool needs into .venv, matched to this machine's NVIDIA GPU.
#
#   ./setup.sh          detect the GPU and install the fastest setup that fits
#   ./setup.sh --cpu    install the CPU-only setup (no NVIDIA GPU needed)
#   ./setup.sh --gpu    install the GPU setup without asking the driver
#
# Safe to run again: it replaces .venv. Your keys/, results/ and images/ are not touched.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MIN_DRIVER=550          # oldest NVIDIA driver that runs the CUDA 12.4 libraries used here
MIN_GPU_MB=2000         # below this the GPU engine does not fit
mode="${1:-auto}"

say() { printf '\n== %s\n' "$*"; }

python_bin=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys, venv; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; then
        python_bin="$candidate"; break
    fi
done
if [ -z "$python_bin" ]; then
    echo "Python 3.10 to 3.13 with the venv module is needed (Debian/Ubuntu: sudo apt install python3 python3-venv)." >&2
    exit 1
fi

if [ "$mode" = "auto" ]; then
    mode="cpu"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "No nvidia-smi found: no NVIDIA driver on this machine. Using the CPU setup."
    elif ! gpu_line=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) || [ -z "$gpu_line" ]; then
        echo "nvidia-smi cannot reach the GPU (driver not loaded?). Using the CPU setup."
    else
        gpu_name=$(echo "$gpu_line" | cut -d, -f1 | xargs)
        driver=$(echo "$gpu_line" | cut -d, -f2 | xargs)
        gpu_mb=$(echo "$gpu_line" | cut -d, -f3 | xargs)
        echo "GPU: $gpu_name, driver $driver, $gpu_mb MB"
        if [ "${driver%%.*}" -lt "$MIN_DRIVER" ]; then
            echo "Driver $driver is older than $MIN_DRIVER. Update the NVIDIA driver for GPU speed. Using the CPU setup for now."
        elif [ "$gpu_mb" -lt "$MIN_GPU_MB" ]; then
            echo "GPU memory under $MIN_GPU_MB MB is too small. Using the CPU setup."
        else
            mode="gpu"
        fi
    fi
else
    mode="${mode#--}"
fi
case "$mode" in gpu|cpu) ;; *) echo "Usage: ./setup.sh [--gpu|--cpu]" >&2; exit 1 ;; esac

say "Creating .venv with $("$python_bin" --version) ($mode setup)"
rm -rf .venv
"$python_bin" -m venv .venv
pip() { .venv/bin/python -m pip --disable-pip-version-check "$@"; }
pip install -q --upgrade pip

say "Installing packages (the GPU setup downloads about 3 GB)"
if [ "$mode" = "gpu" ]; then
    pip install -q -r requirements/base.txt -r requirements/gpu-cuda12.txt
else
    pip install -q -r requirements/base.txt -r requirements/cpu.txt
fi
echo "$mode" > .venv/install-mode
cat requirements/*.txt | sha256sum | cut -d' ' -f1 > .venv/requirements.sum

say "Checking the text engine (a new GPU builds its engines now: several minutes, once)"
if .venv/bin/python app/s3_ocr.py --check; then
    say "Ready. Start with: ./run.sh"
else
    echo "The engine check failed. See the messages above." >&2
    exit 1
fi

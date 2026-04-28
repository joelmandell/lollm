#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR="${MODEL_DIR:-artifacts}"
python -m uvicorn serve_transformer:app --host 127.0.0.1 --port 8010

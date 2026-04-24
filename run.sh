#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Backend..."
cd "$ROOT_DIR/backend"
PYTHONPATH="$ROOT_DIR" python3 -m uvicorn app.main:app --reload &
BACK_PID=$!

echo "Starting Frontend..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONT_PID=$!

wait $BACK_PID $FRONT_PID

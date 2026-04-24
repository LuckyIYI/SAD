#!/usr/bin/env bash
set -euo pipefail

# Render a saved SAD TXT sites file through the standard runner.
# Examples:
#   ./render.sh results/foo_sites.txt --width 1024 --height 1024

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT_DIR/run.sh" --render "$@"

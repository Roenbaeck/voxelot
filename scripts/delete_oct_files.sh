#!/usr/bin/env bash
set -euo pipefail

# Deletes legacy .oct files in worlds/ directory. Use with care.
echo "Deleting legacy .oct files in worlds/..."
rm -f worlds/*.oct || true
echo "Done."

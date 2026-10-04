#!/usr/bin/env bash
# Start the viewer with a world.
#
#   ./run.sh                      # worlds/flat_city_test.toml
#   ./run.sh large_world_test     # any config in worlds/ by name
#   ./run.sh path/to/other.toml   # or a config file path
#
# Extra arguments go to the viewer. Note that the viewer saves the config it was started with
# when it exits.
set -euo pipefail
cd "$(dirname "$0")"

world="${1:-flat_city_test}"
if [ "$#" -gt 0 ]; then shift; fi
case "$world" in
  *.toml) config="$world" ;;
  *) config="worlds/$world.toml" ;;
esac

if [ ! -f "$config" ]; then
  echo "No such world config: $config" >&2
  echo "Available: $(cd worlds && ls *.toml | sed 's/\.toml$//' | tr '\n' ' ')" >&2
  exit 1
fi

export RUST_LOG="${RUST_LOG:-info}"
exec cargo run --release --bin voxelot -- "$config" "$@"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path-to-binary> <out-zip>"
  exit 2
fi

BIN_PATH="$1"
OUT_ZIP="$2"

if [ ! -f "$BIN_PATH" ]; then
  echo "Binary not found: $BIN_PATH"
  exit 3
fi

TMPDIR=$(mktemp -d)
STAGEDIR="$TMPDIR/demo"
mkdir -p "$STAGEDIR"

# Copy binary
cp "$BIN_PATH" "$STAGEDIR/"
# Optionally include dSYM bundle (set INCLUDE_DSYM=1 to include)
DSYM_DIR="$(dirname "$BIN_PATH")/$(basename "$BIN_PATH").dSYM"
if [ "${INCLUDE_DSYM:-0}" = "1" ] && [ -d "$DSYM_DIR" ]; then
  cp -R "$DSYM_DIR" "$STAGEDIR/"
fi

# Copy world and palette
cp "worlds/flat_city_test.vhc" "$STAGEDIR/flat_city_test.vhc"
cp "worlds/palette.txt" "$STAGEDIR/palette.txt"
# Copy demo config (relative paths in package)
cp "demo/config.toml" "$STAGEDIR/config.toml"
# Optionally include README
cp "demo/README.md" "$STAGEDIR/README.md" || true

# Ensure executable bit where applicable
chmod +x "$STAGEDIR/voxelot" || true

# Report size of the packaged binary
echo "Packaged binary size:"
du -h "$STAGEDIR/$(basename "$BIN_PATH")" || true

pushd "$TMPDIR" >/dev/null
zip -r "$OUT_ZIP" demo >/dev/null
popd >/dev/null

# Move zip to cwd
mv "$TMPDIR/$OUT_ZIP" "$(pwd)/$OUT_ZIP"
rm -rf "$TMPDIR"

echo "Created $OUT_ZIP"

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

# Create worlds directory inside the demo package (preserves original config paths)
mkdir -p "$STAGEDIR/worlds"
# Copy world and palette into worlds/
cp "worlds/flat_city_test.vhc" "$STAGEDIR/worlds/flat_city_test.vhc"
cp "worlds/palette.txt" "$STAGEDIR/worlds/palette.txt"
# Copy skybox into worlds/
if [ -f "worlds/skybox.hdr" ]; then
  cp "worlds/skybox.hdr" "$STAGEDIR/worlds/skybox.hdr"
fi

# Copy demo config (we'll replace this file with the tuned config)
cp "demo/config.toml" "$STAGEDIR/config.toml"
# Optionally include README
cp "demo/README.md" "$STAGEDIR/README.md" || true

# Ensure executable bit where applicable
chmod +x "$STAGEDIR/voxelot" || true

# Report size of the packaged binary
echo "Packaged binary size:"
du -h "$STAGEDIR/$(basename "$BIN_PATH")" || true

pushd "$TMPDIR" >/dev/null
# Try native zip first, then fall back to PowerShell Compress-Archive, 7z, or Python
if command -v zip >/dev/null 2>&1; then
  echo "Creating zip with zip..."
  zip -r "$OUT_ZIP" demo >/dev/null
elif command -v pwsh >/dev/null 2>&1 || command -v powershell >/dev/null 2>&1; then
  echo "Creating zip with PowerShell Compress-Archive..."
  if command -v pwsh >/dev/null 2>&1; then
    pwsh -NoProfile -Command "Compress-Archive -Path 'demo' -DestinationPath '$OUT_ZIP' -Force"
  else
    powershell -NoProfile -Command "Compress-Archive -Path 'demo' -DestinationPath '$OUT_ZIP' -Force"
  fi
elif command -v 7z >/dev/null 2>&1; then
  echo "Creating zip with 7z..."
  7z a -tzip "$OUT_ZIP" demo >/dev/null
elif command -v python3 >/dev/null 2>&1; then
  echo "Creating zip with Python shutil..."
  python3 - <<PY >/dev/null
import shutil
shutil.make_archive('$OUT_ZIP'.replace('.zip',''), 'zip', 'demo')
PY
else
  echo "No zip tool found (tried: zip, pwsh/powershell, 7z, python3); cannot create $OUT_ZIP"
  popd >/dev/null
  rm -rf "$TMPDIR"
  exit 127
fi
popd >/dev/null

# Move zip to cwd
mv "$TMPDIR/$OUT_ZIP" "$(pwd)/$OUT_ZIP"
rm -rf "$TMPDIR"

echo "Created $OUT_ZIP"

Demo package

To run the demo after extracting the zip:

- Linux/macOS: open a terminal in the extracted folder and run `./voxelot`.
- Windows: double-click `voxelot.exe` or run it from PowerShell/CMD.

The included `config.toml` points to `flat_city_test.vhc` and `palette.txt` (all files included). The viewer uses the named `config.toml` by default when run without arguments.

Notes:
- macOS Gatekeeper may require you to allow the app in System Preferences the first time you run it.
- On Linux, ensure GPU drivers are present for `wgpu`/Vulkan to work properly.

Releases & notarization:
- When a Git tag is pushed (e.g., `v1.0.0`), the CI will build demo zips for all platforms and automatically create a GitHub Release containing them.
- macOS code signing and notarization happen automatically if signing secrets are configured (see `RELEASE_SIGNING.md`).

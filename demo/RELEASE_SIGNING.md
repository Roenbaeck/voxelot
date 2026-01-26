macOS Signing & Notarization / Release publishing

This repository supports automated GitHub releases and optional macOS code signing + notarization via GitHub Actions.

What the workflow does

- Builds demo zips for Linux, macOS, and Windows on push (or workflow_dispatch).
- If a push is a tag (e.g., `v1.2.3`), the workflow will collect the three `demo-*.zip` artifacts and create a GitHub Release with those assets.
- If macOS signing/notarization secrets are provided in the repository, the macOS job will:
  1. Decode and install the provided signing identity (PKCS#12) into a temporary keychain.
  2. Strip the built `target/release/voxelot` binary (to reduce size) before signing.
  3. Codesign the stripped binary.
  4. Submit the binary to Apple's notarization service using `xcrun notarytool`.
  5. Staple the notarization ticket if available.

Notes:
- The CI now performs a best-effort strip step for each platform before packaging. On Windows, available strip tools vary, so the step is optional and may be skipped if not present on the runner.
- `split-debuginfo = "unpacked"` is enabled in `Cargo.toml` so debug info is emitted as separate dSYM/debug files rather than embedded in the binary; these are not included in the demo zip to keep package sizes small.

Required repository secrets (for macOS signing & notarization)

- `MACOS_SIGNING_CERT` — base64-encoded PKCS#12 (.p12) containing your Developer ID Application identity.
  - Example: `base64 -w0 signing_identity.p12` (macOS/Linux) then copy the output into the secret value field.
- `MACOS_SIGNING_CERT_PASSWORD` — password for the PKCS#12 file (empty string if none).
- `MACOS_SIGNING_ID` — the identity string to use with `codesign` (e.g., "Developer ID Application: Your Name (TEAMID)").
- `APPLE_API_KEY` — base64-encoded Apple API key file (AuthKey_XXXXXX.p8) for notarytool.
  - Example: `base64 -w0 AuthKey_XXXX.p8`
- `APPLE_API_KEY_ID` — the Key ID (e.g., `XXXXXX`)
- `APPLE_API_ISSUER_ID` — the Issuer ID / Team ID associated with the key

Notes & Security

- The workflow will only attempt signing/notarization if both `MACOS_SIGNING_CERT` and `APPLE_API_KEY_ID` are present (non-empty).
- Keep your signing identity and Apple API key secrets secure. Use repository-level secrets or organization secrets with limited access.
- Notarization requires a valid Apple Developer account with appropriate roles.

If you'd like, I can also:
- Add a small secret-checker workflow that validates presence & basic decoding of secrets.
- Add more robust error handling/logging in the signing/notarization steps.


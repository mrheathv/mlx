#!/usr/bin/env bash
# Builds "MLX Studio.app" and wraps it in a DMG at dist/MLXStudio-<version>.dmg.
#
# Run on an Apple Silicon Mac with Xcode or the Command Line Tools:
#   ./scripts/build-dmg.sh
#
# The app is ad-hoc signed, which is enough to run on the Mac that built
# it. A copy downloaded or AirDropped to another Mac gets quarantined by
# Gatekeeper; open it there with right-click → Open (or notarize it with a
# Developer ID, which this script doesn't do).
#
# Python isn't bundled. The app ships uv plus the backend's source, and
# installs its own Python 3.11 + dependencies on first launch (see
# Sources/MLXStudioApp/Services/BackendEnvironment.swift).
#
# Overrides: VERSION, BUNDLE_ID, UV_VERSION, UV_BINARY (path to a local
# uv binary to bundle instead of downloading one).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="MLX Studio"
EXECUTABLE="MLXStudio"
VERSION="${VERSION:-0.1.0}"
BUNDLE_ID="${BUNDLE_ID:-com.mlxstudio.MLXStudio}"
# BackendEnvironment.swift's `uv venv --managed-python` / `uv pip install -r
# pyproject.toml` invocations were validated against this version.
UV_VERSION="${UV_VERSION:-0.11.32}"
UV_ASSET="uv-aarch64-apple-darwin"

DIST="$ROOT/dist"
APP="$DIST/$APP_NAME.app"
DMG="$DIST/MLXStudio-$VERSION.dmg"

die() { echo "error: $*" >&2; exit 1; }
step() { echo "==> $*"; }

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] \
  || die "build on an Apple Silicon Mac — MLX only runs there"
command -v swift >/dev/null || die "swift not found; install Xcode or the Command Line Tools (xcode-select --install)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

step "Building release binary"
swift build -c release --package-path "$ROOT"
BIN_DIR="$(swift build -c release --package-path "$ROOT" --show-bin-path)"

step "Assembling $APP_NAME.app"
rm -rf "$APP" "$DMG"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/MLXStudioApp" "$APP/Contents/MacOS/$EXECUTABLE"
# ModelCatalog.loadBundled() reads this from Contents/Resources.
cp "$ROOT/Sources/MLXStudioApp/Resources/models.json" "$APP/Contents/Resources/"
rsync -a \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude '.pytest_cache' --exclude 'tests' \
  "$ROOT/backend/" "$APP/Contents/Resources/backend/"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>$EXECUTABLE</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

step "Bundling uv"
if [[ -n "${UV_BINARY:-}" ]]; then
  [[ -x "$UV_BINARY" ]] || die "UV_BINARY=$UV_BINARY is not an executable file"
  cp "$UV_BINARY" "$WORK/uv"
else
  base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
  curl -fsSL -o "$WORK/$UV_ASSET.tar.gz" "$base/$UV_ASSET.tar.gz"
  curl -fsSL -o "$WORK/$UV_ASSET.tar.gz.sha256" "$base/$UV_ASSET.tar.gz.sha256"
  expected="$(awk '{print $1}' "$WORK/$UV_ASSET.tar.gz.sha256")"
  actual="$(shasum -a 256 "$WORK/$UV_ASSET.tar.gz" | awk '{print $1}')"
  [[ "$expected" == "$actual" ]] || die "uv $UV_VERSION checksum mismatch (expected $expected, got $actual)"
  tar -xzf "$WORK/$UV_ASSET.tar.gz" -C "$WORK"
  cp "$WORK/$UV_ASSET/uv" "$WORK/uv"
fi
install -m 755 "$WORK/uv" "$APP/Contents/MacOS/uv"

step "Signing (ad-hoc)"
# Nested executables first, then the bundle, which seals Resources too.
codesign --force --sign - "$APP/Contents/MacOS/uv"
codesign --force --sign - "$APP"
codesign --verify --strict --verbose=1 "$APP"

step "Creating DMG"
mkdir -p "$WORK/dmg"
cp -R "$APP" "$WORK/dmg/"
ln -s /Applications "$WORK/dmg/Applications"
hdiutil create -volname "$APP_NAME" -srcfolder "$WORK/dmg" -ov -format UDZO "$DMG" >/dev/null

echo
echo "Built: $DMG"
echo "Open it, drag \"$APP_NAME\" to Applications, and launch it from there."

#!/usr/bin/env bash
# Build ReleaseFast and package a relocatable release tarball:
#
#   dist/ming-tts-serve-<version>-macos-arm64.tar.gz
#   dist/ming-tts-serve-<version>-macos-arm64.tar.gz.sha256
#
# The tarball holds one top-level folder:
#
#   bin/ming-tts
#   lib/libmlxc.dylib, libmlx.dylib, libjaccl.dylib, mlx.metallib
#   README.md, LICENSE, API.md, SHA256SUMS
#
# bin/ming-tts finds lib/ through the rpath @loader_path/../lib, and libmlx
# loads mlx.metallib from its own directory, so the folder works wherever it
# is extracted. Binaries and dylibs are ad-hoc signed; they are not notarized.
#
# Usage: scripts/package.sh [version]   (default: .version in build.zig.zon)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { echo "[package] ERROR: $*" >&2; exit 1; }

VERSION="${1:-$(sed -n 's/^ *\.version = "\(.*\)",$/\1/p' build.zig.zon)}"
[ -n "$VERSION" ] || die "no version given and none found in build.zig.zon"
VERSION="${VERSION#v}"
[ "$(uname -m)" = arm64 ] || die "must run on Apple silicon (arm64)"

NAME="ming-tts-serve-$VERSION-macos-arm64"
DIST="$REPO_ROOT/dist"
STAGE="$DIST/$NAME"
TARBALL="$DIST/$NAME.tar.gz"

zig build -Doptimize=ReleaseFast

rm -rf "$STAGE" "$TARBALL" "$TARBALL.sha256"
mkdir -p "$STAGE/bin" "$STAGE/lib"
cp zig-out/bin/ming-tts "$STAGE/bin/"
cp lib/mlx/lib/libmlxc.dylib lib/mlx/lib/libmlx.dylib lib/mlx/lib/libjaccl.dylib lib/mlx/lib/mlx.metallib "$STAGE/lib/"
cp README.md LICENSE API.md "$STAGE/"

# Replace the build-tree rpaths (../../lib/mlx/lib) with the package layout.
BIN="$STAGE/bin/ming-tts"
for rp in $(otool -l "$BIN" | awk '/cmd LC_RPATH/{f=1} f && /path /{print $2; f=0}'); do
  install_name_tool -delete_rpath "$rp" "$BIN"
done
install_name_tool -add_rpath "@loader_path/../lib" "$BIN"

# install_name_tool invalidates the linker's signature; Apple silicon refuses
# to run unsigned arm64 code, so sign everything ad hoc.
codesign --force --sign - "$BIN" "$STAGE"/lib/*.dylib

# No load command may point into the build machine. System libraries are fine.
for f in "$BIN" "$STAGE"/lib/*.dylib; do
  bad="$(otool -L "$f" | tail -n +2 | awk '{print $1}' | grep -vE '^(@rpath/|/usr/lib/|/System/Library/)' || true)"
  [ -z "$bad" ] || die "$f links a non-relocatable path: $bad"
  bad="$(otool -l "$f" | awk '/cmd LC_RPATH/{f=1} f && /path /{print $2; f=0}' | grep -v '^@' || true)"
  [ -z "$bad" ] || die "$f has an absolute rpath: $bad"
done

# Smoke test: dyld must resolve every library from the staged layout.
"$BIN" >/dev/null 2>&1 || die "bin/ming-tts does not start from the package layout"

(cd "$STAGE" && find . -type f | sed 's|^\./||' | sort | xargs shasum -a 256 > "$DIST/SHA256SUMS.tmp")
mv "$DIST/SHA256SUMS.tmp" "$STAGE/SHA256SUMS"

COPYFILE_DISABLE=1 tar -czf "$TARBALL" -C "$DIST" "$NAME"
(cd "$DIST" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

echo "[package] $TARBALL"
cat "$TARBALL.sha256"

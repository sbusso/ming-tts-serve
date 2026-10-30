#!/bin/sh
# Install ming-tts from the latest GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/sbusso/ming-tts-serve/main/install.sh | sh
#
# Environment:
#   MING_VERSION   release to install, e.g. 0.1.0 (default: latest)
#   MING_PREFIX    install root (default: ~/.local/share/ming-tts-serve)
#   MING_BIN_DIR   where the `ming-tts` command goes (default: ~/.local/bin)
#   MING_NO_MODEL  set to 1 to skip the model download (about 3 GB)
#   MING_TARBALL   install from this local tarball instead of downloading
set -eu

REPO=sbusso/ming-tts-serve
PREFIX=${MING_PREFIX:-$HOME/.local/share/ming-tts-serve}
BIN_DIR=${MING_BIN_DIR:-$HOME/.local/bin}
MODEL_DIR=$PREFIX/models/Ming-3.0-tts-0.5b

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || die "ming-tts needs macOS."
[ "$(uname -m)" = arm64 ] || die "ming-tts needs a Mac with Apple silicon."
os=$(sw_vers -productVersion)
major=${os%%.*}
minor=$(printf '%s' "$os" | cut -d. -f2)
if [ "$major" -lt 26 ] || { [ "$major" -eq 26 ] && [ "${minor:-0}" -lt 2 ]; }; then
  die "ming-tts needs macOS 26.2 or later (this Mac has $os)."
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ -n "${MING_TARBALL:-}" ]; then
  tarball=$MING_TARBALL
else
  if [ -z "${MING_VERSION:-}" ]; then
    MING_VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
      sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1)
    [ -n "$MING_VERSION" ] || die "could not find the latest release of $REPO."
  fi
  name=ming-tts-serve-$MING_VERSION-macos-arm64
  url=https://github.com/$REPO/releases/download/v$MING_VERSION/$name.tar.gz
  say "Downloading ming-tts $MING_VERSION"
  curl -fL --progress-bar -o "$tmp/$name.tar.gz" "$url"
  curl -fsSL -o "$tmp/$name.tar.gz.sha256" "$url.sha256"
  (cd "$tmp" && shasum -a 256 -c "$name.tar.gz.sha256" >/dev/null) || die "checksum mismatch."
  tarball=$tmp/$name.tar.gz
fi

say "Installing to $PREFIX"
mkdir -p "$tmp/x"
tar -xzf "$tarball" -C "$tmp/x"
src=$(find "$tmp/x" -mindepth 1 -maxdepth 1 -type d | head -n 1)
[ -x "$src/bin/ming-tts" ] || die "the tarball has no bin/ming-tts."
mkdir -p "$PREFIX"
rm -rf "${PREFIX:?}/bin" "${PREFIX:?}/lib"
cp -R "$src/." "$PREFIX/"
xattr -dr com.apple.quarantine "$PREFIX" 2>/dev/null || true

# A wrapper, not a symlink: the binary finds its libraries next to itself,
# and the wrapper points it at the installed model by default.
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/ming-tts" <<EOF
#!/bin/sh
: "\${MING_MODEL_DIR:=$MODEL_DIR}"
export MING_MODEL_DIR
exec "$PREFIX/bin/ming-tts" "\$@"
EOF
chmod +x "$BIN_DIR/ming-tts"

if [ "${MING_NO_MODEL:-0}" != 1 ] && [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  if command -v uvx >/dev/null 2>&1; then
    say "Downloading the model (about 3 GB)"
    uvx --from huggingface_hub hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir "$MODEL_DIR"
  else
    say "Skipped the model download: uv is not installed. Install uv (https://docs.astral.sh/uv/), then run:"
    printf '    uvx --from huggingface_hub hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir "%s"\n' "$MODEL_DIR"
  fi
fi

say "Installed. Try:"
printf '    ming-tts say --out hello.wav --json %s && afplay hello.wav\n' "'{\"task\":\"IP\",\"voice\":\"UK_George\",\"text\":\"Hello from Ming.\"}'"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "Add $BIN_DIR to your PATH: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc" ;;
esac

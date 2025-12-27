#!/bin/bash
# Build script for WASM target

set -e

echo "Building Ren for WebAssembly..."

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build with wasm-pack
wasm-pack build --target web --out-dir web/pkg --features web

echo "Build complete! Files are in web/pkg/"
echo ""
echo "To test locally:"
echo "  cd web && python3 -m http.server 8080"
echo "  Then open http://localhost:8080"

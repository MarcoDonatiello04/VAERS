#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_TEX="$ROOT_DIR/report/main.tex"
OUT_DIR="$ROOT_DIR/report"

usage() {
  cat <<'EOF'
Usage:
  ./build_report.sh           # Compile PDF
  ./build_report.sh clean     # Clean temporary files
  ./build_report.sh rebuild   # Clean + compile
EOF
}

require_latexmk() {
  if ! command -v latexmk >/dev/null 2>&1; then
    echo "Error: latexmk not found."
    echo "Install with: brew install --cask mactex-no-gui"
    echo "Then ensure PATH includes /Library/TeX/texbin"
    exit 1
  fi
}

build() {
  require_latexmk
  latexmk -pdf -interaction=nonstopmode -file-line-error \
    -output-directory="$OUT_DIR" "$REPORT_TEX"
  echo "PDF generated: $OUT_DIR/main.pdf"
}

clean() {
  require_latexmk
  latexmk -C -output-directory="$OUT_DIR" "$REPORT_TEX"
  echo "Temporary files cleaned in: $OUT_DIR"
}

case "${1:-build}" in
  build) build ;;
  clean) clean ;;
  rebuild)
    clean
    build
    ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac

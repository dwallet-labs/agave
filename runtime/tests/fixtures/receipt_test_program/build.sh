#!/usr/bin/env bash
#
# Rebuilds receipt_test_program.so (SBPFv0, dynamic syscalls) from
# receipt_test_program.c with the Solana platform tools. The .so is checked
# in and used by the Transaction Receipt Tree tests in
# runtime/src/bank/transaction_receipts.rs.
#
# Usage: build.sh [platform tools version, default v1.54]

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sdk="$here/../../../../programs/sbf/c"
tools_version="${1:-v1.54}"
llvm="$HOME/.cache/solana/$tools_version/platform-tools/llvm"
clang_include="$(ls -d "$llvm"/lib/clang/*/include | head -n 1)"
obj="$here/receipt_test_program.o"
trap 'rm -f "$obj"' EXIT

"$llvm/bin/clang" -Werror -O2 -fno-builtin -std=c17 \
  -isystem "$sdk/inc" -isystem "$clang_include" -I "$llvm/sbpf/include" \
  -target sbf -fPIC \
  -o "$obj" -c "$here/receipt_test_program.c"

"$llvm/bin/ld.lld" -z notext -shared --Bdynamic "$sdk/sbf.ld" \
  --entry entrypoint -L "$llvm/lib/sbpf" -z max-page-size=4096 -lc \
  -o "$here/receipt_test_program.so" "$obj"

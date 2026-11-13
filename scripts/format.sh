#!/usr/bin/env bash
# Format the C++ sources under altro/ with clang-format (style: .clang-format).
#
#   scripts/format.sh                 format every source file in place
#   scripts/format.sh --changed       only files that differ from origin/main (or the given base)
#   scripts/format.sh --changed main
#   scripts/format.sh --check         change nothing; list files that need formatting, exit 1 if any
#   scripts/format.sh --check --changed
#
# Uses the clang-format on PATH; set CLANG_FORMAT=/path/to/clang-format to pick another one.
# alilqr/ is an old snapshot kept for reference and is left untouched.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

check=false
changed=false
base=origin/main
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) check=true ;;
        --changed)
            changed=true
            if [[ $# -gt 1 && "$2" != --* ]]; then
                base="$2"
                shift
            fi
            ;;
        -h | --help)
            sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $1 (see --help)" >&2
            exit 2
            ;;
    esac
    shift
done

clang_format="${CLANG_FORMAT:-clang-format}"
if ! command -v "$clang_format" > /dev/null; then
    echo "clang-format not found. Install it, or set CLANG_FORMAT=/path/to/clang-format." >&2
    exit 2
fi

pattern='^altro/.*\.(h|hpp|cc|cpp)$'
if $changed; then
    # Tracked files changed since the base (committed or not), plus new untracked files.
    files=$({ git diff --name-only --diff-filter=d "$(git merge-base "$base" HEAD)"
              git ls-files --others --exclude-standard; } | sort -u | grep -E "$pattern" || true)
else
    files=$(git ls-files | grep -E "$pattern" || true)
fi
if [[ -z "$files" ]]; then
    echo "no files to format"
    exit 0
fi

if $check; then
    unformatted=0
    while IFS= read -r file; do
        if ! "$clang_format" --style=file "$file" | cmp -s - "$file"; then
            echo "needs formatting: $file"
            unformatted=$((unformatted + 1))
        fi
    done <<< "$files"
    if [[ $unformatted -gt 0 ]]; then
        echo "$unformatted file(s) need formatting; run scripts/format.sh"
        exit 1
    fi
    echo "all $(wc -l <<< "$files") file(s) are formatted"
else
    while IFS= read -r file; do
        "$clang_format" --style=file -i "$file"
    done <<< "$files"
    echo "formatted $(wc -l <<< "$files") file(s)"
fi

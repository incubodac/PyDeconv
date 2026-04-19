#!/usr/bin/env bash
set -euo pipefail

pick_python() {
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version_output=$("$candidate" -V 2>&1)
      # version_output is something like "Python 3.10.12"
      version=$(echo "$version_output" | awk '{print $2}')
      version_major=$(echo "$version" | cut -d. -f1)
      version_minor=$(echo "$version" | cut -d. -f2)
      if [ "$version_major" -gt 3 ] || { [ "$version_major" -eq 3 ] && [ "$version_minor" -ge 9 ]; }; then
        echo "$candidate"
        return 0
      fi
    fi
  done

  echo "Error: Python 3.9 or greater must be installed." >&2
  return 1
}

PYTHON_BIN="$(pick_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  exit 1
fi

echo "Using: ${PYTHON_BIN} ($( "${PYTHON_BIN}" -V ))"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment at .venv"
  "${PYTHON_BIN}" -m venv .venv
else
  echo "Virtual environment already exists at .venv"
fi

python -m pip install --upgrade pip
python -m pip install mne numpy scipy matplotlib pandas scikit-learn art pyside6

echo "Done. To start using it:"
echo "  source .venv/bin/activate"

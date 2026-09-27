#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Refusing to combine DRY_RUN=1 with automatic shutdown."
  echo "Run the ordinary launcher for a dry run:"
  echo "  DRY_RUN=1 ./run_full_train_strict_equivalence_annotation_v2.sh"
  exit 2
fi

echo "Requesting sudo authorization now so shutdown can run unattended later."
sudo -v

# Keep the sudo timestamp valid while the long annotation job runs. The
# background process performs no privileged mutation; it only refreshes the
# authorization timestamp once per minute.
while true; do
  sudo -n true || exit
  sleep 60
done &
SUDO_KEEPALIVE_PID=$!

cleanup() {
  kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
  wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if ./run_full_train_strict_equivalence_annotation_v2.sh; then
  cleanup
  trap - EXIT INT TERM
  echo "Annotation completed successfully. Shutting down now."
  sudo -n shutdown -h now
else
  status=$?
  echo "Annotation failed with exit code $status. Automatic shutdown cancelled." >&2
  exit "$status"
fi

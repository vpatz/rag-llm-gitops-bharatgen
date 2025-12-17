#!/bin/sh
set -e

# PATH must be set inside PID 1
export PATH="/usr/bin:/bin:/usr/local/bin"

echo "PATH=$PATH"
which git
which python3

exec "$@"

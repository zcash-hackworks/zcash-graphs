#!/bin/bash
set -efuo pipefail

REPODIR="$(dirname "$(readlink -f "$0")")"
OUTDIR="${REPODIR}/output"

[ -d "$OUTDIR" ] || mkdir "$OUTDIR"

time docker build -t zcash-graphs "$REPODIR"
time docker run -v "${OUTDIR}:/project/output" zcash-graphs "$@"

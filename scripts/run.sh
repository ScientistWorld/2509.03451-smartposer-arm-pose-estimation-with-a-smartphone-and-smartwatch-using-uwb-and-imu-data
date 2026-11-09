#!/bin/bash
# Compute job entry point (iteration 11 hardening bundle; see notes/hardening/job.sh).
set -e
cd "$(dirname "$0")/.."
bash notes/hardening/job.sh

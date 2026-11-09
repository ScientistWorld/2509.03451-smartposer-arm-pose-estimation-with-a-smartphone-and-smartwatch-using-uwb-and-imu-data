#!/bin/bash
# Compute job entry point (final hardening verification; see notes/hardening/job_v3.sh).
set -e
cd "$(dirname "$0")/.."
bash notes/hardening/job_v3.sh

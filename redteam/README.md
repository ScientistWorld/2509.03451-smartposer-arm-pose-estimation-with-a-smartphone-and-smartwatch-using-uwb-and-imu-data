Audit producers use only the supplied visible method/data interface. Captured
reference source is in `support/visible_reference/`. `support/rank.py` and
`support/summarize.py` are audit-only analysis, separate from artifact producers.

Run `bash redteam/run_all.sh` on a compute job. Every case also has a `run.sh`;
it installs the corresponding method outputs, invokes both real evaluators with
no arguments in separate processes, saves scores and evidence locally, and
restores the original checkpoints and score files. Reference artifacts are
reused as the audit request permits and rebuilt if absent.

`cache/` and logs are ignored. Result JSON files and the final report are the
persistent evidence. A shortcut that ties is explicitly labeled a shortcut.

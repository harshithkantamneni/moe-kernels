#!/usr/bin/env python
"""Every addendum-6 build record: python run.py --write (re-records every record) or --check (recomputes each and
compares; exit 1 on any difference). Each script stands alone too. About a minute on the laptop, under 0.4 GB."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ("pc_spans", "ptx_inflight", "pc_check", "pc2_check", "ds_check", "naive_check", "alternatives", "reach",
           "check_unseen", "post_gate")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    rc = 0
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, str(HERE / f"{s}.py"), mode], capture_output=True, text=True,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"})
        sys.stdout.write(r.stdout if mode == "--check" else (f"{s} written\n" if r.returncode == 0 else r.stdout))
        if r.returncode:
            sys.stderr.write(r.stderr)
            rc = 1
    sys.exit(rc)

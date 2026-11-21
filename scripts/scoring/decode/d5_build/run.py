#!/usr/bin/env python
"""Every addendum-5 build record: python run.py --write (re-records every .out) or --check (recomputes every .out and
compares; exit 1 on any difference). Each script stands alone too. About 2 minutes on the laptop, 0.3 GB."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ("ladders", "call_forms", "alternatives", "every_path", "ablation", "weak")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    rc = 0
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, str(HERE / f"{s}.py"), mode], capture_output=True, text=True)
        sys.stdout.write(r.stdout if mode == "--check" else f"{s}.out written\n" if r.returncode == 0 else r.stdout)
        if r.returncode:
            sys.stderr.write(r.stderr)
            rc = 1
    sys.exit(rc)

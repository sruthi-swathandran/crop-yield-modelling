"""
Print to the terminal and to a log file at the same time.

    from runlog import tee_to
    tee_to("features")        # writes logs/features.log

Redirecting with `> logs/x.log` puts the output in the file but hides
it from the window, so there is no way to see progress without opening
a second terminal. This sends every line to both.

The log is appended to rather than overwritten, with a header line per
run, so a rerun after a crash keeps the record of the run that crashed.
"""

import sys
from datetime import datetime
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

    def __getattr__(self, name):
        # isatty, encoding and the rest come from the terminal stream
        return getattr(self.streams[0], name)


def tee_to(name: str) -> Path:
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f"{name}.log"
    f = open(path, "a", encoding="utf-8", buffering=1)
    f.write(f"\n===== run started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)      # tracebacks land in the log too
    print(f"logging to {path}")
    return path

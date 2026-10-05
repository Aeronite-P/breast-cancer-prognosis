"""
watch_download.py — live, READ-ONLY progress screen for the full-cohort slide job (Stage C --all).

Shows, refreshing every few seconds:
  * how many new patients are saved (out of 789) and the whole cohort (out of 1,033)
  * each slide currently downloading: patient, progress bar, MB done / total, speed
  * total download speed, recent pace (patients/hour) and a rough finish estimate
  * the last few lines of the job's log

It only reads files (/tmp/*.svs sizes, data/stage_c_tiles_full/, the log) and asks GDC once for
slide sizes. It never touches the running job, so it is safe to open and close any time.

Run:  python src/watch_download.py        (Ctrl+C to quit)
"""
import glob
import json
import os
import sys
import time
from collections import deque

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DIR = os.path.join(ROOT, "data", "stage_c_tiles_full")
LOG = os.path.join(ROOT, "data", "logs", "stage_c_full.log")
NEW_TOTAL, ORIGINAL = 789, 244          # new patients in the full run; original size-matched cohort
REFRESH = 3                              # seconds between updates
WINDOW = 30                              # speeds are averaged over the last 30 s (smoother)


def slide_info():
    """file_id -> (patient, size in bytes) for every TCGA-BRCA diagnostic slide (one GDC request)."""
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
    params = {"filters": json.dumps(filt), "fields": "file_id,file_size,cases.submitter_id",
              "format": "JSON", "size": "20000"}
    hits = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120).json()["data"]["hits"]
    return {h["file_id"]: (h["cases"][0]["submitter_id"], h["file_size"]) for h in hits}


def bar(frac, width=24):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def log_tail(n=4):
    """Last n meaningful log lines (skips library warnings)."""
    try:
        lines = [l.rstrip() for l in open(LOG) if l.strip()
                 and not any(w in l for w in ("Warning", "warnings.warn", "Fast image processor"))]
        return lines[-n:]
    except OSError:
        return ["(log not found)"]


def main():
    print("Asking GDC for slide sizes...", flush=True)
    info = slide_info()
    hist = deque()                       # (time, {file_id: bytes}) samples for the speed window
    while True:
        now = time.time()
        files = {os.path.basename(p)[:-4]: os.path.getsize(p) for p in glob.glob("/tmp/*.svs")}
        hist.append((now, files))
        while now - hist[0][0] > WINDOW and len(hist) > 2:
            hist.popleft()
        t0, old = hist[0]
        dt = max(now - t0, 1e-6)
        saved = glob.glob(os.path.join(FULL_DIR, "*.npy"))
        recent = sum(os.path.getmtime(p) > now - 2 * 3600 for p in saved)      # saves in the last 2 h
        rate = recent / 2.0                                                     # patients per hour
        left = NEW_TOTAL - len(saved)

        out = ["\033[2J\033[H",   # clear the screen, cursor to top
               f"Full-cohort slide download  |  {time.strftime('%a %b %d  %H:%M:%S')}   (Ctrl+C to quit)",
               "",
               f"New patients saved : {len(saved)} / {NEW_TOTAL}  {bar(len(saved) / NEW_TOTAL)} "
               f"{100 * len(saved) / NEW_TOTAL:.0f}%",
               f"Whole cohort       : {ORIGINAL + len(saved)} / {ORIGINAL + NEW_TOTAL}",
               f"Recent pace        : {rate:.1f} patients/hour (last 2 h)"
               + (f"  ->  ~{left / rate / 24:.1f} days left" if rate > 0 else ""),
               "",
               f"Downloading now ({len(files)} slides):"]
        total_speed = 0.0
        for fid, size in sorted(files.items(), key=lambda kv: -kv[1]):
            patient, full = info.get(fid, ("?", 0))
            speed = max(size - old.get(fid, size), 0) / dt / 1e6
            total_speed += speed
            frac = size / full if full else 0
            out.append(f"  {patient:<14} {bar(frac)} {100 * frac:5.1f}%  "
                       f"{size / 1e6:7.0f} / {full / 1e6:6.0f} MB  {speed:5.2f} MB/s")
        out += ["", f"Total download speed: {total_speed:.2f} MB/s (average over last {WINDOW} s)", "", "Latest log lines:"]
        out += ["  " + l[:110] for l in log_tail()]
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        time.sleep(REFRESH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped watching (the download job keeps running).")

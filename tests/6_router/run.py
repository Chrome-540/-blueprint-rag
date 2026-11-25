"""Which action does a request go to? 1 small Gemini call per request.

python tests/6_router/run.py v1_baseline                   the requests below
python tests/6_router/run.py v1_baseline "show me A101"    just this one
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_version, save_output

from google import genai

CASES = [
    ("show me A101", "show_sheet"),
    ("open the roof framing plan", "show_sheet"),
    ("which structural drawings do we have?", "list_contents"),
    ("give me the v2 set", "show_file"),
    ("what is the difference between v1 and v2", "show_differences"),
    ("show me the differences in A101", "show_differences"),
    ("what changed on the foundation plan?", "show_differences"),
    ("what is the occupant load on the first floor?", "answer_from_context"),
    ("hi", "small_talk"),
    ("what can you do?", "small_talk"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("request", nargs="?")
    a = ap.parse_args()
    save_output("6_router", f"run_{a.version}_{a.request or "all"}")
    mod = load_version("6_router", a.version)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip(), http_options={"api_version": "v1beta", "timeout": 60_000})
    wrong = 0
    for request, want in [(a.request, None)] if a.request else CASES:
        start = time.time()
        r = mod.detect_intent(client, request)
        bad = want is not None and r["name"] != want
        wrong += bad
        args = {k: v for k, v in r["args"].items() if v}
        print(f"{'WRONG' if bad else 'ok':<6} {request!r:<50} -> {r['name']:<20} {args}  ({time.time() - start:.1f}s)")
    assert not wrong, f"{wrong} request(s) went to the wrong action"
    print("\nOK")


if __name__ == "__main__":
    main()

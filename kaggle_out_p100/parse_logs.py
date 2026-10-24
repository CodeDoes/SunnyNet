import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

for name in ["prose", "codemix"]:
    p = BASE / f"{name}.final.json"
    raw = p.read_text(encoding="utf-8", errors="replace")
    # drop the leading warning line if present
    start = raw.find("[")
    if start > 0:
        raw = raw[start:]
    events = json.loads(raw)
    lines = []
    for ev in events:
        if ev.get("stream_name") == "stdout":
            lines.append(ev.get("data", ""))
    text = "".join(lines)
    print("=" * 20, name, "=" * 20)
    print(f"total stdout events: {len(events)}")
    # Find sample-generating sections
    for marker in ["PROMPT", "prompt", ">>>", "SAMPLE", "samples"]:
        idx = [i for i, l in enumerate(lines) if marker in l]
        if idx:
            print(f"  marker '{marker}' found {len(idx)} times; first at line {idx[0]}")
    # Print tail summary region (last 60 lines)
    print("--- last 40 lines ---")
    print("\n".join(text.splitlines()[-40:]))
    print()

import math
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
for n in ["prose", "codemix"]:
    out = (BASE / f"{n}.full.log").read_text(encoding="utf-8")
    for m in re.finditer(
        r"=== 完成 wall=([\d.]+)s +([\d.]+) ms/token .*best_val=([\d.]+)", out
    ):
        wall, mst, best = m.group(1), m.group(2), float(m.group(3))
        if float(wall) > 1000:
            print(
                f"{n:8s} final: wall={wall}s | {mst} ms/tok | best_val={best:.6f} (ppl {math.exp(best):.3f})"
            )
    for m in re.finditer(
        r"(?:EN tokens|total tokens)=([\d,]+) -> (\d+) steps.*?using (\d+)", out
    ):
        print(f"{n:8s} corpus_tokens={m.group(1)} steps_used={m.group(3)}")

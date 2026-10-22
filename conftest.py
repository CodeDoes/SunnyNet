"""裸 `pytest` 只把 tests/ 加进 sys.path, 仓库根不在其中, `import sunnynet` 会失败。"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
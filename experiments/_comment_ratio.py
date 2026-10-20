"""统计 sunnynet 包注释比 (纯注释行 / 代码行, 及含 docstring 口径)."""
from __future__ import annotations

import ast
import pathlib

tot_c = tot_d = tot_code = 0
for p in sorted(pathlib.Path("sunnynet").glob("*.py")):
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    doc = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                          ast.AsyncFunctionDef)):
            b = n.body[0] if n.body else None
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) \
                    and isinstance(b.value.value, str):
                for i in range(b.lineno - 1, b.end_lineno):
                    doc.add(i)
    c = sum(1 for i, l in enumerate(lines)
            if l.strip().startswith("#") and i not in doc)
    code = sum(1 for i, l in enumerate(lines)
               if l.strip() and not l.strip().startswith("#") and i not in doc)
    tot_c += c
    tot_d += len(doc)
    tot_code += code
    print(f"{p.name:22s} 注释{c:3d} docstring{len(doc):3d} 代码{code:4d} "
          f"含doc比{(c + len(doc)) / max(1, code) * 100:5.1f}%")
print(f"TOTAL 纯注释{tot_c} docstring{tot_d} 代码{tot_code} "
      f"纯注释比{tot_c / tot_code * 100:.1f}% 含doc比{(tot_c + tot_d) / tot_code * 100:.1f}%")
*[中文版](REPO_AUDIT.md)*
# Repository gap audit

Conclusions come from measured checks against the API / filesystem / clean copy, not inference.
Clean copy = `git archive HEAD | tar -x` (equivalent to a CI checkout, without the `data/` directory).

## 1. Item-by-item verdicts

| # | Audit-table claim | Verification result (evidence) | Verdict |
|---|---|---|---|
| 1 | 0 Release / 0 Tag | `api.github.com/.../tags` → `[]`; `/releases` → `[]` | Confirmed |
| 2 | No CONTRIBUTING | `.github/` contains only `FUNDING.yml` + `workflows/tests.yml`; no CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / Issue / PR templates at all | Confirmed |
| 3 | No sample data | `git ls-tree -r HEAD` shows 219 files, zero `.bin`/`.csv`/`.pt`; `data/` is in `.gitignore` | Confirmed |
| 4 | No pretrained weights | As above, no weight file of any kind | Confirmed |
| 5 | Zero external validation | `contributors` = `sunny-cx`(20 commits) + `SunnyClaw-cx`(3 commits); the `name`/`bio` of both accounts point to the same person | Confirmed |
| 6 | No GPU / no public benchmark | All of §15 of the report is GPU measurements: GTX 1060 forward 8.0–15.2×, training step 9.9–12.5×, and it states there is no peer comparison on the GPU side | Not confirmed |
| 7 | Baseline written in-house | `sunnynet/linear_baselines.py` is indeed a hand-written implementation | Confirmed, but not a gap |
| 8 | README has no CI badge | README has only 1 `shields.io` badge (Afdian) | Confirmed |
| 9 | CI is green | All 17 CI runs are FAILURE, it has never succeeded in history; the badge measured as `failing` | Completely opposite → fixed |

## 2. The two most critical points in the table

- The one written backwards is the real defect: the table claims "CI is green", while measurement shows 17/17 all red.
  All-green local tests mask it — `data/proc/tokenizer.json` exists locally but is absent after a CI checkout.
- The two that look most alarming do not actually hold: #6 has GPU validation (merely without peer comparison),
  and #7's hand-written baseline is standard practice in architecture research (an official implementation can only be reused when a peer
  has a paper, and "no implementation regression" requires writing one's own).

## 3. Items missing from the table (more worth fixing before promotion)

| Item | Evidence | Impact |
|---|---|---|
| Zero GitHub topics | `api.github.com/.../topics` → `names: []` | Directly affects discoverability, first priority before promotion |
| No dependency declaration | No `pyproject.toml`/`setup.py`/`requirements.txt` at all | Installation experience; the version number `0.1.0` exists only in `sunnynet/__init__.py` |
| Discussions not enabled | `has_discussions: false` | Missing entry point for hosting a community |

## 4. Fixed and verified

Root cause: `train/train.py` and `train/evaluate.py` hard-code `ROOT/"data"/"proc"/"tokenizer.json"`,
bypassing `build_corpus.PROC`, which structurally defeats the test's `monkeypatch.setattr(build_corpus,"PROC",…)`.

- Clean-copy reproduction: `2 failed, 52 passed` (`FileNotFoundError` @ `test_smoke_pipeline.py:102`)
- Fix: switched to the existing `build_corpus.load_tokenizer()`; the `smoke-train` job now also installs `pytest`
- Verification: clean copy 54 passed · `bare pytest tests/` 54 passed · smoke 2 passed · 3.12 `compileall` no errors
- Result: after pushing, the official badge `failing → passing` (green for the first time)
- The current test count has grown to 59 architecture tests + 2 smoke tests, still all green.
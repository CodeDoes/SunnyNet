"""Ollama / LM Studio 接入层契约测试.

真机导入 (ollama create) 需要外部服务, 不适合放进 CI; 这里覆盖生成逻辑、
探活诊断、模板正确性与 LM Studio 目录布局。
"""
from __future__ import annotations

import json
import socket

import pytest

from sunnynet import integrate


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_modelfile_has_explicit_template(tmp_path):
    """rwkv7 必须带显式 TEMPLATE, 否则 Ollama 内置 rwkv-world 会双重包裹 prompt."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    mf = integrate.write_modelfile(gguf)
    body = mf.read_text(encoding="utf-8")
    assert "TEMPLATE" in body
    assert "User: {{ .Prompt }}" in body
    assert "Assistant:" in body


def test_modelfile_uses_absolute_posix_path(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    body = integrate.write_modelfile(gguf).read_text(encoding="utf-8")
    from_line = body.splitlines()[0]
    assert from_line.startswith("FROM ")
    assert "\\" not in from_line, "反斜杠在 Modelfile 里是转义符, 须用 /"


def test_modelfile_template_override(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    mf = integrate.write_modelfile(gguf, template="{{ .Prompt }}")
    assert "{{ .Prompt }}" in mf.read_text(encoding="utf-8")


def test_modelfile_rejects_missing_gguf(tmp_path):
    with pytest.raises(FileNotFoundError):
        integrate.write_modelfile(tmp_path / "nope.gguf")


def test_ollama_import_dry_run_does_not_execute(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    mf, cmd = integrate.ollama_import(gguf, "x", ollama_bin="/bin/echo", run=False)
    assert mf.exists()
    assert cmd[1] == "create" and "x" in cmd


def test_ollama_import_reports_offline_server(tmp_path):
    """server 离线时必须给可执行提示, 而不是让 CLI 抛 could not locate."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    with pytest.raises(RuntimeError) as ei:
        integrate.ollama_import(gguf, "x", ollama_bin="/bin/echo",
                                host=f"127.0.0.1:{_free_port()}", run=True)
    msg = str(ei.value)
    assert "server 未运行" in msg
    assert "could not locate ollama app" in msg
    assert "ollama serve" in msg


def test_server_alive_false_on_dead_port():
    ok, info = integrate.server_alive(f"127.0.0.1:{_free_port()}", timeout=1.0)
    assert ok is False and info


def test_host_url_passes_through_scheme():
    assert integrate._host_url("http://1.2.3.4:1/") == "http://1.2.3.4:1"
    assert integrate._host_url("127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_find_ollama_returns_none_or_path():
    p = integrate.find_ollama()
    assert p is None or isinstance(p, str)


def test_stage_lmstudio_gguf_layout(tmp_path):
    """LM Studio 用 <models>/<publisher>/<repo>/<file>.gguf 目录结构做索引."""
    gguf = tmp_path / "w.gguf"
    gguf.write_bytes(b"x" * 64)
    models = tmp_path / "models"
    dst = integrate.stage_lmstudio_gguf(gguf, models, "pub", "repo")
    assert dst.parent == models / "pub" / "repo"
    assert dst.name == "w.gguf" and dst.exists()
    assert dst.read_bytes() == b"x" * 64


def test_stage_lmstudio_gguf_idempotent(tmp_path):
    gguf = tmp_path / "w.gguf"
    gguf.write_bytes(b"x" * 64)
    models = tmp_path / "models"
    a = integrate.stage_lmstudio_gguf(gguf, models, "pub", "repo")
    b = integrate.stage_lmstudio_gguf(gguf, models, "pub", "repo")
    assert a == b


def test_stage_lmstudio_requires_models_dir(tmp_path, monkeypatch):
    gguf = tmp_path / "w.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(integrate, "find_lmstudio_models_dir", lambda: None)
    with pytest.raises(RuntimeError, match="LM Studio 模型目录"):
        integrate.stage_lmstudio_gguf(gguf)


def test_lmstudio_config_has_both_paths(tmp_path):
    out = integrate.write_lmstudio_config(tmp_path / "c.json",
                                          gguf_placed="E:/x/m.gguf")
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["path_a_gguf"]["gguf_placed"] == "E:/x/m.gguf"
    assert cfg["path_a_gguf"]["stop"] == ["User:", "System:"]
    assert cfg["path_b_endpoint"]["base_url"].endswith("/v1")


def test_smoke_reports_error_on_dead_port():
    ok, info = integrate._smoke(f"127.0.0.1:{_free_port()}", "x", timeout=1.0)
    assert ok is False and info
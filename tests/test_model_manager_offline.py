"""Offline tests for model discovery (GGUF + Ollama). Pure stdlib."""

import importlib.util
import tempfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_cands = [
    _here / "private_gpt" / "components" / "llm" / "model_manager.py",
    _here.parent / "private_gpt" / "components" / "llm" / "model_manager.py",
    _here / "model_manager.py",
]
_target = next((p for p in _cands if p.exists()))
_spec = importlib.util.spec_from_file_location("mm_under_test", _target)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def test_list_gguf_sorted_and_filtered():
    d = Path(tempfile.mkdtemp())
    (d / "b.gguf").write_bytes(b"x")
    (d / "a.gguf").write_bytes(b"x")
    (d / "readme.txt").write_text("x")
    assert mm.list_gguf_models(d) == ["a.gguf", "b.gguf"]
    print("PASS test_list_gguf_sorted_and_filtered")


def test_ollama_offline_returns_empty():
    # Nothing is listening; must fail fast and quietly.
    assert mm.list_ollama_models("http://127.0.0.1:11434", timeout=0.4) == []
    assert mm.list_ollama_models("", timeout=0.4) == []
    print("PASS test_ollama_offline_returns_empty")


def test_available_models_shape():
    d = Path(tempfile.mkdtemp())
    (d / "m.gguf").write_bytes(b"x")
    av = mm.available_models(d, "http://127.0.0.1:11434")
    assert av[0] == {"id": "gguf:m.gguf", "label": "GGUF · m.gguf", "source": "gguf"}
    print("PASS test_available_models_shape")


def test_parse_model_id():
    assert mm.parse_model_id("gguf:x.gguf") == ("gguf", "x.gguf")
    assert mm.parse_model_id("ollama:llama3.1:latest") == ("ollama", "llama3.1:latest")
    for bad in ["nocolon", "weird:x", "gguf:"]:
        try:
            mm.parse_model_id(bad)
            raise AssertionError(f"should reject {bad}")
        except ValueError:
            pass
    print("PASS test_parse_model_id")


if __name__ == "__main__":
    test_list_gguf_sorted_and_filtered()
    test_ollama_offline_returns_empty()
    test_available_models_shape()
    test_parse_model_id()
    print("\nALL MODEL-MANAGER TESTS PASSED")

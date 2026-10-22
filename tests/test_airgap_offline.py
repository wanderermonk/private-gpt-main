"""Offline tests for the strict air-gap guard. Pure stdlib + the real
airgap module (which has no private_gpt imports)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_here = Path(__file__).resolve().parent
_candidates = [
    _here / "private_gpt" / "utils" / "airgap.py",
    _here.parent / "private_gpt" / "utils" / "airgap.py",
    _here / "airgap.py",
]
_target = next((p for p in _candidates if p.exists()))
_spec = importlib.util.spec_from_file_location("airgap_under_test", _target)
airgap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(airgap)


def _settings(llm="llamacpp", emb="huggingface", ollama_base="http://localhost:11434",
              openai_key="", gemini_key=""):
    return SimpleNamespace(
        llm=SimpleNamespace(mode=llm),
        embedding=SimpleNamespace(mode=emb),
        ollama=SimpleNamespace(api_base=ollama_base, embedding_api_base=ollama_base),
        openai=SimpleNamespace(api_key=openai_key, embedding_api_key=""),
        azopenai=SimpleNamespace(api_key="", embedding_api_key=""),
        gemini=SimpleNamespace(api_key=gemini_key, embedding_api_key=""),
        sagemaker=SimpleNamespace(api_key="", embedding_api_key=""),
    )


def test_clean_local_passes():
    assert airgap.check_airgap(_settings()) == []
    airgap.enforce_airgap(_settings())  # must not raise
    print("PASS test_clean_local_passes")


def test_cloud_llm_blocked():
    v = airgap.check_airgap(_settings(llm="openai"))
    assert any("llm.mode" in s for s in v), v
    try:
        airgap.enforce_airgap(_settings(llm="openai"))
        raise AssertionError("should have raised")
    except airgap.AirgapViolation:
        pass
    print("PASS test_cloud_llm_blocked")


def test_remote_ollama_blocked():
    v = airgap.check_airgap(_settings(llm="ollama", ollama_base="http://10.8.0.5:11434"))
    # 10.x is private -> allowed; a public host must be flagged
    assert v == [], v  # private LAN address is acceptable
    v2 = airgap.check_airgap(_settings(llm="ollama", ollama_base="http://api.example.com:11434"))
    assert any("ollama" in s for s in v2), v2
    print("PASS test_remote_ollama_blocked")


def test_populated_cloud_key_flagged():
    v = airgap.check_airgap(_settings(openai_key="sk-xxxx"))
    assert any("openai.api_key" in s for s in v), v
    print("PASS test_populated_cloud_key_flagged")


def test_local_url_helper():
    assert airgap._is_local_url("http://127.0.0.1:11434")
    assert airgap._is_local_url("http://localhost:8001")
    assert not airgap._is_local_url("https://huggingface.co")
    print("PASS test_local_url_helper")


if __name__ == "__main__":
    test_clean_local_passes()
    test_cloud_llm_blocked()
    test_remote_ollama_blocked()
    test_populated_cloud_key_flagged()
    test_local_url_helper()
    print("\nALL AIRGAP TESTS PASSED")

import asyncio
import random
import string
from unittest.mock import patch

import pytest

from jam.core.inference import JamInferenceEngine
from jam.core.orchestrator import (
    JamOrchestrator,
    extract_json_object,
    heuristic_analyze_intent,
)
from jam.core.recommender import ModelRecommender
from jam.hardware.detector import HardwareDetector
from jam.memory.vector_db import JamMemory, LocalHashEmbeddingFunction
from jam.ui.app import JamApp, format_vram, validate_key


def _make_test_license_key() -> str:
    chars = string.ascii_uppercase + string.digits
    p1 = "".join(random.choices(chars, k=4))
    p2 = "".join(random.choices(chars, k=4))
    p3 = "".join(random.choices(chars, k=2))
    core = f"JAM-{p1}-{p2}-{p3}"
    checksum = sum(ord(c) for c in core.replace("-", "")) % 100
    return f"{core}{checksum:02d}"


def test_license_generation_and_validation():
    for _ in range(20):
        key = _make_test_license_key()
        assert len(key) == 18
        assert validate_key(key) is True
        assert validate_key(key.lower()) is True

    assert validate_key("JAM-0000-0000-0099") is False
    assert validate_key("INVALID") is False
    assert validate_key("JAM-1234-5678-12XX") is False


def test_hardware_detector_report_structure():
    report = HardwareDetector().get_full_report()
    assert "os" in report
    assert report["cpu"]["physical_cores"] >= 1
    assert report["cpu"]["total_cores"] >= 1
    assert report["ram"]["total_gb"] > 0
    assert "type" in report["gpu"]


@pytest.mark.parametrize(
    "ram_gb,gpu_type,vram_gb,expected_gen",
    [
        (3.2, "CPU Only", 0, "Dolphin3.0-Qwen2.5-0.5B-GGUF (Q4_K_M)"),
        (6.0, "CPU Only", 0, "Dolphin3.0-Qwen2.5-1.5B-GGUF (Q4_K_M)"),
        (8.0, "CPU Only", 0, "Dolphin3.0-Llama3.2-3B-GGUF (Q4_K_M)"),
        (16.0, "CPU Only", 0, "Dolphin3.0-Llama3.1-8B-GGUF (Q4_K_M)"),
        (16.0, "NVIDIA (CUDA)", 12.0, "Qwen2.5-14B-Instruct-abliterated-GGUF (Q4_K_M)"),
        (32.0, "NVIDIA (CUDA)", 24.0, "Qwen2.5-32B-Instruct-abliterated-GGUF (Q4_K_M)"),
        (32.0, "Apple Silicon (Metal)", "Compartida con RAM", "Qwen2.5-32B-Instruct-abliterated-GGUF (Q4_K_M)"),
    ],
)
def test_recommender_tiers_match_inference_mapping(ram_gb, gpu_type, vram_gb, expected_gen):
    report = {
        "os": "Linux",
        "cpu": {"physical_cores": 4, "total_cores": 8, "architecture": "x86_64"},
        "ram": {"total_gb": ram_gb, "available_gb": ram_gb / 2},
        "gpu": {"type": gpu_type, "vram_gb": vram_gb, "details": "Test"},
    }
    recs = ModelRecommender(report).get_recommendations()
    assert recs["generator"]["name"] == expected_gen
    # Every recommended model must exist explicitly in JamInferenceEngine.MODEL_MAPPING
    assert recs["generator"]["name"] in JamInferenceEngine.MODEL_MAPPING
    assert recs["manager"]["name"] in JamInferenceEngine.MODEL_MAPPING


def test_format_vram():
    assert format_vram(8.0) == "8.0 GB"
    assert format_vram("Compartida con RAM") == "Compartida con RAM"


def test_main_self_test():
    from jam.main import run_self_test

    assert run_self_test() == 0


def test_orchestrator_bilingual_heuristics_and_pipeline(tmp_path):
    mem = JamMemory(
        db_path=str(tmp_path / "mem"),
        embedding_function=LocalHashEmbeddingFunction(),
    )
    orch = JamOrchestrator(mem, {"city": "Monterrey", "temp": 26}, session_id="s1")

    assert heuristic_analyze_intent("Me llamo Diego")["is_new_fact"] is True
    assert heuristic_analyze_intent("My name is Alice")["is_new_fact"] is True
    assert heuristic_analyze_intent("I prefer concise answers")["category"] == "preference"
    assert heuristic_analyze_intent("¿Cómo me llamo?")["is_new_fact"] is False
    assert extract_json_object('```json\n{"is_new_fact": true}\n```') == {"is_new_fact": True}

    res1 = orch.process_pipeline("Me llamo Diego y trabajo en Python", lang="es")
    assert any("vectorizado" in line.lower() for line in res1["logs"])
    assert len(res1["messages"]) >= 2
    assert "Monterrey" in res1["messages"][0]["content"]
    orch.record_interaction("Me llamo Diego y trabajo en Python", "Hola Diego, gusto en conocerte.")

    res2 = orch.process_pipeline("¿En qué lenguaje trabajo?", lang="es")
    assert any("Diego" in f or "Python" in f for f in res2["recalled_facts"])
    assert len(res2["recent_chat"]) == 2


def test_tui_end_to_end_demo_flow(tmp_path):
    """Run the actual Textual TUI headless via Pilot: license -> onboarding -> RAG commands -> chat."""
    valid_key = _make_test_license_key()

    async def _run():
        with patch(
            "jam.ui.app.get_location_and_weather",
            return_value={"city": "TestCity", "temp": 22, "online": True},
        ), patch(
            "jam.memory.vector_db.JamMemory._init_default_embeddings",
            return_value=(LocalHashEmbeddingFunction(), "local_hash"),
        ):
            app = JamApp(language="es", data_dir=str(tmp_path / "jam_data"), demo_mode=True)
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                assert app.needs_license is True

                # 1. Invalid license rejected
                inp = app.query_one("#chat-input")
                inp.value = "BAD-KEY"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert app.needs_license is True

                # 2. Valid license accepted -> boots hardware & RAG memory
                inp.value = valid_key
                await pilot.press("enter")
                for _ in range(30):
                    await pilot.pause(0.1)
                    if not app.is_initializing and app.memory_ready:
                        break

                assert app.needs_license is False
                assert app.memory_ready is True
                assert app.is_first_time is True
                assert app.onboarding_step == 1

                # 3. Complete 3-step onboarding
                for answer in ("Diego", "Ingeniero de software en Python", "Directas al grano"):
                    inp.value = answer
                    await pilot.press("enter")
                    await pilot.pause(0.1)

                assert app.is_first_time is False
                assert app.memory.get_stats()["user_facts"] == 3

                # 4. Exercise slash commands & verify Chat vs Logs separation
                for cmd in ("/help", "/memory", "/remember Me gusta Linux", "/search Python", "/face status"):
                    inp.value = cmd
                    await pilot.press("enter")
                    await pilot.pause(0.1)

                assert app.memory.get_stats()["user_facts"] == 4

                # Toggle to Logs tab via /logs and back via Ctrl+L
                tabs = app.query_one("#main-tabs")
                assert tabs.active == "tab-chat"
                inp.value = "/logs"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert tabs.active == "tab-logs"
                await pilot.press("ctrl+l")
                await pilot.pause(0.1)
                assert tabs.active == "tab-chat"

                # 5. Send regular chat message and verify chat persistence + clean Chat UI
                inp.value = "¿Qué sabes sobre mi trabajo con Python?"
                await pilot.press("enter")
                for _ in range(20):
                    await pilot.pause(0.1)
                    if app.memory.get_stats()["chat_messages"] >= 2:
                        break

                assert app.memory.get_stats()["chat_messages"] >= 2

                # Verify [MNG] / [GEN] / [SYS] stay strictly in #system-log and NOT in #chat-log
                chat_lines = "\n".join(line.text for line in app.query_one("#chat-log").lines)
                sys_lines = "\n".join(line.text for line in app.query_one("#system-log").lines)
                assert "[MNG]" not in chat_lines
                assert "[GEN]" not in chat_lines
                assert "[SYS]" not in chat_lines
                assert "[MNG]" in sys_lines
                assert "[RAG]" in sys_lines

                # 6. Forget & clear commands
                inp.value = "/forget Linux"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert app.memory.get_stats()["user_facts"] == 3

                inp.value = "/clear"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert app.memory.get_stats()["chat_messages"] == 0

    asyncio.run(_run())

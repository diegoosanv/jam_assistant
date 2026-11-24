import time

import pytest

from jam.memory.vector_db import (
    JamMemory,
    LocalHashEmbeddingFunction,
    chunk_text,
    infer_fact_category,
    lexical_similarity,
)


@pytest.fixture
def memory(tmp_path):
    return JamMemory(
        db_path=str(tmp_path / "memory"),
        embedding_function=LocalHashEmbeddingFunction(),
    )


def test_cosine_space_and_stats(memory):
    stats = memory.get_stats()
    assert stats["user_facts"] == 0
    assert stats["chat_messages"] == 0
    assert stats["user_space"] == "cosine"
    assert stats["chat_space"] == "cosine"
    assert stats["embedding_backend"] == "local_hash"


def test_exact_deduplication_ignores_case_accents_whitespace(memory):
    r1 = memory.upsert_user_fact("El usuario vive en Ciudad de México", source="conversation")
    r2 = memory.upsert_user_fact("  el usuario vive en   ciudad de mexico  ", source="conversation")

    assert r1["action"] == "added"
    assert r2["action"] == "duplicate"
    assert r1["id"] == r2["id"]
    assert memory.get_stats()["user_facts"] == 1


def test_semantic_near_duplicate_merges_instead_of_duplicating(memory):
    r1 = memory.upsert_user_fact("El usuario programa en Python todos los días", dedup_threshold=0.25)
    r2 = memory.upsert_user_fact("El usuario programa en Python todos los dias con FastAPI", dedup_threshold=0.25)

    assert r1["action"] == "added"
    assert r2["action"] in ("updated", "duplicate")
    assert memory.get_stats()["user_facts"] == 1


def test_empty_inputs_are_safely_ignored(memory):
    assert memory.add_user_fact("") is None
    assert memory.add_user_fact("   ") is None
    assert memory.add_chat_message("") is None
    assert memory.recall_user_facts("") == []
    assert memory.recall_chat_history("   ") == []


def test_hybrid_recall_and_relevance_threshold_filtering(memory):
    memory.add_user_fact("El usuario se llama Diego y desarrolla software en Python", source="onboarding")
    memory.add_user_fact("Prefiere respuestas concisas y técnicas", source="onboarding")
    memory.add_user_fact("Le gusta el café expreso por las mañanas", source="conversation")

    # Spanish accent-insensitive query should find the right fact first
    hits = memory.recall_user_facts_detailed("cafe expreso mananas", n_results=2)
    assert len(hits) >= 1
    assert "café" in hits[0]["text"].lower()
    assert hits[0]["score"] >= 0.22

    # Unrelated query should be filtered out instead of returning irrelevant noise
    noise = memory.recall_user_facts("receta de sushi japones con salmon", n_results=3, min_score=0.35)
    assert noise == []


def test_category_inference_and_filtering(memory):
    assert infer_fact_category("El usuario se llama Ana") == "profile"
    assert infer_fact_category("Trabajo como ingeniero de datos") == "work"
    assert infer_fact_category("Prefiero modo oscuro") == "preference"
    assert infer_fact_category("Vivo en Guadalajara") == "location"

    memory.add_user_fact("El usuario se llama Ana", category="profile")
    memory.add_user_fact("Trabajo como ingeniero de datos en Python", category="work")

    work_hits = memory.recall_user_facts("ingeniero de datos", n_results=3, category="work")
    assert len(work_hits) == 1
    assert "ingeniero" in work_hits[0].lower()


def test_chunk_text_splits_long_documents_with_overlap():
    long_text = " ".join(f"Oracion numero {i} sobre arquitectura vectorial." for i in range(40))
    chunks = chunk_text(long_text, max_chars=200, overlap=40)
    assert len(chunks) >= 3
    assert all(len(c) <= 200 for c in chunks)


def test_chat_history_recent_window_and_episodic_recall(memory):
    memory.add_chat_message("Hablamos de configurar Docker Compose para PostgreSQL", role="user", session_id="s1")
    time.sleep(0.01)
    memory.add_chat_message("Puedes definir un volumen persistente en docker-compose.yml", role="assistant", session_id="s1")
    time.sleep(0.01)
    memory.add_chat_message("Hoy hace buen clima", role="user", session_id="s1")
    time.sleep(0.01)
    memory.add_chat_message("Sí, disfruta tu día", role="assistant", session_id="s1")

    recent = memory.get_recent_chat(n_results=2, session_id="s1")
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    assert "clima" in recent[0]["text"]
    assert recent[1]["role"] == "assistant"

    ctx = memory.build_memory_context(
        query="como configuramos Docker Compose PostgreSQL",
        session_id="s1",
        max_recent_turns=2,
        max_recalled_chats=2,
    )
    assert len(ctx["recent_chat"]) == 2
    recalled_texts = [item["text"] for item in ctx["recalled_chat"]]
    assert any("Docker" in t or "docker-compose" in t for t in recalled_texts)


def test_crud_operations_forget_and_clear(memory):
    f1 = memory.add_user_fact("Mi color favorito es el azul marino")
    f2 = memory.add_user_fact("Trabajo con Kubernetes y Linux")
    assert memory.get_stats()["user_facts"] == 2

    all_facts = memory.get_all_user_facts()
    assert len(all_facts) == 2

    assert memory.delete_user_fact(f1) is True
    assert memory.delete_user_fact("nonexistent-id") is False
    assert memory.get_stats()["user_facts"] == 1

    deleted = memory.forget_matching_facts("Kubernetes Linux", min_score=0.25)
    assert deleted == 1
    assert memory.get_stats()["user_facts"] == 0

    memory.add_chat_message("Hola", role="user", session_id="s1")
    memory.add_chat_message("Adiós", role="assistant", session_id="s1")
    assert memory.clear_chat_history(session_id="s1") == 2
    assert memory.get_stats()["chat_messages"] == 0


def test_lexical_similarity_bilingual():
    assert lexical_similarity("programación python", "Me gusta la programacion en Python") > 0.5
    assert lexical_similarity("gato negro", "arquitectura distribuida kubernetes") < 0.2

import argparse
import gc
import os
import sys
import tempfile
import traceback

# Ensure repository root is importable both via `python jam/main.py` and `python -m jam.main`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _configure_console_utf8() -> None:
    """Prevent UnicodeEncodeError on legacy Windows conhost (CP1252) terminals."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _resolve_default_data_dir() -> str:
    """
    Use a portable jam_data folder next to JAM.exe / repo root when writable;
    fall back to %LOCALAPPDATA%/JAM/jam_data (or ~/.jam_data) if read-only.
    """
    base_dir = (
        os.path.dirname(os.path.abspath(sys.executable))
        if getattr(sys, "frozen", False)
        else _PROJECT_ROOT
    )
    preferred = os.path.join(base_dir, "jam_data")
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = os.path.join(preferred, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return preferred
    except OSError:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return os.path.join(local_app_data, "JAM", "jam_data")
        return os.path.join(os.path.expanduser("~"), ".jam_data")


_DEFAULT_DATA_DIR = _resolve_default_data_dir()

from jam.ui.app import JamApp


def run_self_test() -> int:
    """
    Verify all bundled native subsystems (used in CI on Windows to verify JAM.exe
    unpacks and loads ChromaDB, llama.cpp, OpenCV cascades, and Textual cleanly).
    """
    from jam.core.inference import LLAMA_AVAILABLE, LLAMA_IMPORT_ERROR, JamInferenceEngine
    from jam.core.recommender import ModelRecommender
    from jam.hardware.detector import HardwareDetector
    from jam.memory.vector_db import JamMemory, LocalHashEmbeddingFunction
    from jam.vision.face_auth import JamFaceAuth

    report = HardwareDetector().get_full_report()
    recs = ModelRecommender(report).get_recommendations()
    assert recs["generator"]["name"] in JamInferenceEngine.MODEL_MAPPING
    assert LLAMA_AVAILABLE, f"llama-cpp-python native library failed to load: {LLAMA_IMPORT_ERROR}"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        mem = JamMemory(
            db_path=os.path.join(tmpdir, "memory"),
            embedding_function=LocalHashEmbeddingFunction(),
        )
        try:
            res = mem.upsert_user_fact("Self-test fact for JAM")
            assert res["action"] == "added"
            assert len(mem.recall_user_facts("Self-test fact")) == 1
        finally:
            mem.close()
            del mem
            gc.collect()

        face = JamFaceAuth(faces_dir=os.path.join(tmpdir, "faces"))
        assert face.is_available(), "OpenCV Haar cascade failed to load"

    print(
        f"[SELF-TEST OK] OS={report['os']} | Model={recs['generator']['name']} | "
        f"llama_cpp=OK | ChromaDB=OK | OpenCV=OK | Textual=OK"
    )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="JAM (Just Another Mind) Local AI Assistant")
    parser.add_argument(
        "--lang",
        choices=["es", "en"],
        default=None,
        help="Interface language (es or en). If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in Demo/Simulated LLM mode with live RAG vector memory (skips GGUF model download).",
    )
    parser.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        help="Directory for local memory, license, and cached models (default: portable jam_data).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run non-interactive startup verification of all native modules and exit.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    _configure_console_utf8()
    args = parse_args(argv)

    if args.self_test:
        sys.exit(run_self_test())

    try:
        if args.lang:
            language = args.lang
        else:
            print("=== JAM (Just Another Mind) ===")
            print("Select Language / Selecciona el idioma de la interfaz:")
            print("[1] Español (Predeterminado)")
            print("[2] English")

            try:
                lang_choice = input("> ").strip()
            except EOFError:
                lang_choice = "1"

            language = "en" if lang_choice == "2" else "es"

        app = JamApp(
            language=language,
            data_dir=args.data_dir,
            demo_mode=args.demo,
        )
        app.run()
    except Exception:
        traceback.print_exc()
        if getattr(sys, "frozen", False):
            try:
                input("\nJAM encontró un error inesperado. Presiona Enter para cerrar...")
            except EOFError:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
from typing import Callable, Dict, List, Optional

from huggingface_hub import hf_hub_download

# Encapsulamos la importación en un try-except porque compilar llama-cpp-python
# puede ser complejo dependiendo de la plataforma del usuario (CUDA vs Metal).
LLAMA_IMPORT_ERROR: Optional[str] = None
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except Exception as _llama_exc:
    Llama = None
    LLAMA_AVAILABLE = False
    LLAMA_IMPORT_ERROR = str(_llama_exc)


class JamInferenceEngine:
    MODEL_MAPPING = {
        "Dolphin3.0-Qwen2.5-0.5B-GGUF (Q4_K_M)": (
            "bartowski/Dolphin3.0-Qwen2.5-0.5B-GGUF",
            "Dolphin3.0-Qwen2.5-0.5B-Q4_K_M.gguf",
        ),
        "Dolphin3.0-Qwen2.5-1.5B-GGUF (Q4_K_M)": (
            "bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF",
            "Dolphin3.0-Qwen2.5-1.5B-Q4_K_M.gguf",
        ),
        "Dolphin3.0-Llama3.2-3B-GGUF (Q4_K_M)": (
            "bartowski/Dolphin3.0-Llama3.2-3B-GGUF",
            "Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf",
        ),
        "Dolphin3.0-Llama3.1-8B-GGUF (Q4_K_M)": (
            "dphn/Dolphin3.0-Llama3.1-8B-GGUF",
            "Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf",
        ),
        "Qwen2.5-14B-Instruct-abliterated-GGUF (Q4_K_M)": (
            "mradermacher/Qwen2.5-14B-Instruct-abliterated-GGUF",
            "Qwen2.5-14B-Instruct-abliterated.Q4_K_M.gguf",
        ),
        "Qwen2.5-32B-Instruct-abliterated-GGUF (Q4_K_M)": (
            "mradermacher/Qwen2.5-32B-Instruct-abliterated-GGUF",
            "Qwen2.5-32B-Instruct-abliterated.Q4_K_M.gguf",
        ),
    }

    def __init__(
        self,
        model_recommendation: Dict,
        models_dir: str = "./jam_data/models",
        use_gpu: bool = False,
        n_ctx: int = 2048,
    ):
        self.models_dir = models_dir
        self.model_info = model_recommendation
        self.use_gpu = use_gpu
        self.n_ctx = n_ctx
        self.llm = None
        self.loaded_model_path: Optional[str] = None
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_hf_details(self, model_name: str):
        """
        Mapea el nombre del recomendador a un repositorio y archivo exacto en HuggingFace Hub.
        """
        return self.MODEL_MAPPING.get(
            model_name,
            ("bartowski/Dolphin3.0-Qwen2.5-0.5B-GGUF", "Dolphin3.0-Qwen2.5-0.5B-Q4_K_M.gguf"),
        )

    def load_model(self, log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Descarga (si es necesario) y carga el modelo GGUF en memoria.
        """
        if not LLAMA_AVAILABLE or Llama is None:
            if log_callback:
                log_callback("[ERROR] llama-cpp-python no está instalado/compilado.")
            return False

        repo_id, filename = self._get_hf_details(self.model_info.get("name", ""))
        model_path = os.path.join(self.models_dir, filename)

        # 1. Descarga Inteligente (compatible con huggingface_hub >= 0.23 y 2.x)
        if not os.path.exists(model_path):
            if log_callback:
                log_callback(
                    f"[SYS] Descargando modelo: {filename} desde HuggingFace "
                    f"({self.model_info.get('memory_required', '~0.5-1GB')})..."
                )
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.models_dir,
                )
                if log_callback:
                    log_callback("[SYS] Descarga completada.")
            except Exception as e:
                if log_callback:
                    log_callback(f"[ERROR] Falló la descarga: {e}")
                return False

        if log_callback:
            log_callback(f"[SYS] Cargando {filename} en Motor de Inferencia...")

        n_gpu_layers = -1 if self.use_gpu else 0
        n_threads = max(1, min(os.cpu_count() or 2, 8))

        # 2. Inicialización de Llama.cpp con fallback seguro si flash_attn no es soportado
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                flash_attn=self.use_gpu,
                verbose=False,
            )
        except Exception:
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=min(self.n_ctx, 2048),
                    n_gpu_layers=0,
                    n_threads=n_threads,
                    flash_attn=False,
                    verbose=False,
                )
            except Exception as e:
                if log_callback:
                    log_callback(f"[ERROR] Falló la carga del modelo en llama.cpp: {e}")
                self.llm = None
                return False

        self.loaded_model_path = model_path
        if log_callback:
            log_callback("[SYS] Modelo cargado y link neuronal establecido.")
        return True

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        is_json: bool = False,
    ) -> str:
        """
        Genera una respuesta usando el chat template nativo del modelo GGUF.
        """
        if not self.llm:
            return (
                '{"is_new_fact": false, "requires_search": false}'
                if is_json
                else "Error: Motor de inferencia offline."
            )

        temp = (0.1 if is_json else 0.7) if temperature is None else temperature
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        content = response["choices"][0]["message"].get("content", "")
        return (content or "").strip()

    def generate(self, prompt: str, max_tokens: int = 256, is_json: bool = False) -> str:
        """
        Genera una respuesta por completado directo (compatibilidad hacia atrás).
        """
        if not self.llm:
            return (
                '{"is_new_fact": false, "requires_search": false}'
                if is_json
                else "Error: Motor de inferencia offline."
            )

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1 if is_json else 0.7,
            stop=["[USER]", "[JAM]", "[SYSTEM]"],
            echo=False,
        )

        return response["choices"][0]["text"].strip()

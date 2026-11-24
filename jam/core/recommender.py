class ModelRecommender:
    def __init__(self, hardware_report):
        self.report = hardware_report
        self.gpu_info = self.report["gpu"]
        self.gpu_vram = self.gpu_info["vram_gb"]
        self.ram_total = self.report["ram"]["total_gb"]
        self.is_mac = self.gpu_info["type"] == "Apple Silicon (Metal)"

        # Calcular memoria efectiva para LLMs
        if self.is_mac:
            # En Mac con Memoria Unificada, usualmente se puede asignar hasta ~75% de la RAM total a la GPU
            self.effective_vram = self.ram_total * 0.75
        elif isinstance(self.gpu_vram, (int, float)) and self.gpu_vram > 0:
            self.effective_vram = self.gpu_vram
        else:
            self.effective_vram = 0  # Fallback a CPU only

    def has_gpu_acceleration(self) -> bool:
        return self.effective_vram > 0

    def recommend_manager_model(self):
        """
        Modelo pequeño y rápido sin censura para gestionar la memoria (RAG) y orquestar.
        """
        if self.effective_vram > 4 or self.ram_total >= 8:
            return {
                "name": "Dolphin3.0-Qwen2.5-1.5B-GGUF (Q4_K_M)",
                "memory_required": "~1.0 GB",
                "role": "Orquestador RAG Uncensored",
            }
        return {
            "name": "Dolphin3.0-Qwen2.5-0.5B-GGUF (Q4_K_M)",
            "memory_required": "~0.4 GB",
            "role": "Orquestador Ligero Uncensored (Bajos Recursos)",
        }

    def recommend_generator_model(self):
        """
        Modelo principal sin censura (Dolphin 3.0 / Abliterated) escalado al hardware disponible.
        """
        # Nivel Ultra-Premium / Premium
        if self.effective_vram >= 24 or (self.is_mac and self.ram_total >= 32):
            return {
                "name": "Qwen2.5-32B-Instruct-abliterated-GGUF (Q4_K_M)",
                "memory_required": "~20 GB",
                "features": "Flash Attention, Paged KV Cache, Uncensored",
                "role": "Generador Premium Uncensored",
            }
        # Nivel Avanzado
        if self.effective_vram >= 12 or (self.is_mac and self.ram_total >= 16):
            return {
                "name": "Qwen2.5-14B-Instruct-abliterated-GGUF (Q4_K_M)",
                "memory_required": "~9 GB",
                "features": "Flash Attention, Uncensored",
                "role": "Generador Avanzado Uncensored",
            }
        # Nivel Estándar
        if self.effective_vram >= 6 or self.ram_total >= 16:
            return {
                "name": "Dolphin3.0-Llama3.1-8B-GGUF (Q4_K_M)",
                "memory_required": "~4.9 GB",
                "role": "Generador Estándar Uncensored",
            }
        # Nivel Básico (portátiles con GPU pequeña o 8+ GB RAM)
        if self.effective_vram > 0 or self.ram_total >= 8:
            return {
                "name": "Dolphin3.0-Llama3.2-3B-GGUF (Q4_K_M)",
                "memory_required": "~2.0 GB",
                "role": "Generador Básico Uncensored",
            }
        # Nivel Ligero (4-8 GB RAM sin GPU)
        if self.ram_total >= 4:
            return {
                "name": "Dolphin3.0-Qwen2.5-1.5B-GGUF (Q4_K_M)",
                "memory_required": "~1.0 GB",
                "role": "Generador CPU-Only Uncensored (Recursos Moderados)",
            }
        # Nivel Ultra-Ligero (< 4 GB RAM sin GPU)
        return {
            "name": "Dolphin3.0-Qwen2.5-0.5B-GGUF (Q4_K_M)",
            "memory_required": "~0.4 GB",
            "role": "Generador CPU-Only Uncensored (Ultra Ligero)",
        }

    def get_recommendations(self):
        return {
            "manager": self.recommend_manager_model(),
            "generator": self.recommend_generator_model(),
            "inference_engine": "llama.cpp (Soporte Multiplataforma)",
            "effective_vram_gb": round(self.effective_vram, 2),
            "use_gpu": self.has_gpu_acceleration(),
        }

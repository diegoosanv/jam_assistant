import platform
import shutil
import subprocess
import psutil


class HardwareDetector:
    def __init__(self):
        self.os_name = platform.system()
        self.arch = platform.machine()

    def get_ram_info(self):
        ram = psutil.virtual_memory()
        total_ram_gb = ram.total / (1024 ** 3)
        available_ram_gb = ram.available / (1024 ** 3)
        return {
            "total_gb": round(total_ram_gb, 2),
            "available_gb": round(available_ram_gb, 2),
        }

    def get_cpu_info(self):
        total_cores = psutil.cpu_count(logical=True) or 1
        physical_cores = psutil.cpu_count(logical=False) or total_cores
        return {
            "physical_cores": physical_cores,
            "total_cores": total_cores,
            "architecture": self.arch,
        }

    def detect_gpu(self):
        gpu_info = {
            "type": "CPU Only",
            "vram_gb": 0,
            "details": "Sin GPU detectada o faltan drivers",
        }

        # Check Apple Silicon (Metal)
        if self.os_name == "Darwin" and self.arch == "arm64":
            return {
                "type": "Apple Silicon (Metal)",
                "vram_gb": "Compartida con RAM",
                "details": "Memoria Unificada",
            }

        # Check NVIDIA (CUDA)
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total,gpu_name",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                )
                if result and result.strip():
                    total_vram_mb = 0.0
                    names = []
                    for line in result.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 2:
                            continue
                        try:
                            total_vram_mb += float(parts[0])
                        except ValueError:
                            pass
                        names.append(parts[1])

                    if names:
                        return {
                            "type": "NVIDIA (CUDA)",
                            "vram_gb": round(total_vram_mb / 1024, 2) if total_vram_mb > 0 else 0,
                            "details": " | ".join(names),
                        }
            except Exception:
                pass

        # Check AMD (ROCm)
        if shutil.which("rocm-smi"):
            try:
                result = subprocess.check_output(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                )
                if result and result.strip():
                    return {
                        "type": "AMD (ROCm)",
                        "vram_gb": "Desconocida",
                        "details": "GPU AMD detectada",
                    }
            except Exception:
                pass

        return gpu_info

    def get_full_report(self):
        return {
            "os": self.os_name,
            "cpu": self.get_cpu_info(),
            "ram": self.get_ram_info(),
            "gpu": self.detect_gpu(),
        }

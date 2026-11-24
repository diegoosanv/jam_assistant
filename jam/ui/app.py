import asyncio
import datetime
import os
import shlex
import uuid

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Input, ProgressBar, RichLog, Static, TabbedContent, TabPane

from jam.core.environment import get_location_and_weather
from jam.core.orchestrator import JamOrchestrator
from jam.core.recommender import ModelRecommender
from jam.hardware.detector import HardwareDetector
from jam.vision.face_auth import JamFaceAuth

JAM_ASCII_LOGO = (
    "[bold #00ffcc]  ╔═╗╔═╗╔╦╗[/]\n"
    "[bold #00ffcc]  ║ ║╠═╣║║║[/]\n"
    "[bold #00ffcc] ╚╝ ╩ ╩╩ ╩[/]\n"
    "[dim #666666] JUST ANOTHER MIND[/]"
)

APP_CSS = """
Screen { background: #000000; }
#sidebar { width: 32; border-right: solid #222222; padding: 1; background: #000000; color: #888888; }
#main-area { width: 1fr; height: 1fr; background: #000000; }
TabbedContent { height: 1fr; background: #000000; }
TabPane { padding: 0 1; background: #000000; height: 1fr; }
Tabs { background: #080808; border-bottom: solid #222222; }
Tab { color: #777777; }
Tab.-active { color: #00ffcc; text-style: bold; }
#chat-log { height: 1fr; border: none; background: #000000; color: #e6e6e6; scrollbar-background: #000000; scrollbar-color: #333333; padding: 0 1; }
#system-log { height: 1fr; border: none; background: #050505; color: #99aacc; scrollbar-background: #050505; scrollbar-color: #333333; padding: 0 1; }
#status-bar { height: 1; background: #080808; color: #00ffcc; padding: 0 1; }
#progress-bar { height: 1; margin: 0 1; display: none; }
#input-container { height: 3; border-top: solid #222222; background: #000000; }
Input { border: none; background: #000000; color: #ffffff; width: 1fr; }
Input:focus { border: none; }
.system-text { color: #00ffcc; margin-top: 1; }
.subtitle { color: #555555; margin-top: 1; text-style: bold; }
"""

TEXTS = {
    "es": {
        "tab_chat": "Chat",
        "tab_logs": "Logs",
        "placeholder": "Escribe un mensaje o /help...",
        "status_locked": "● Bloqueado  |  Ingresa tu clave de licencia",
        "status_init": "◌ Inicializando JAM Core... (Ctrl+L para ver logs)",
        "status_ready": "● Listo  |  Ctrl+L: Logs  |  /help: Comandos",
        "status_thinking": "◌ Pensando...",
        "status_face": "◌ Escaneando rostro...",
        "license_welcome": "Bienvenido a JAM. Ingresa tu clave de licencia para comenzar (formato JAM-XXXX-YYYY-ZZZZ).",
        "license_corrupt": "La licencia guardada no es válida. Ingresa una nueva clave de licencia.",
        "license_invalid": "Clave inválida. Intenta nuevamente.",
        "license_ok_chat": "Licencia verificada. Preparando tu asistente local...",
        "onboard_q1": "Hola, soy JAM. Para empezar a recordar tus preferencias, ¿cómo te llamas?",
        "onboard_q2": "Mucho gusto. ¿Cuál es tu profesión o interés principal?",
        "onboard_q3": "Perfecto. ¿Prefieres respuestas detalladas o directas al grano?",
        "onboard_done": "Listo, ya guardé tu perfil en mi memoria local. Escribe /help para ver comandos (incluyendo reconocimiento facial /face). ¿En qué te ayudo hoy?",
        "welcome_back": "Bienvenido de vuelta. ¿En qué te ayudo hoy?",
        "welcome_face": "¡Hola de nuevo, {user}! Rostro reconocido ({conf:.0%}). ¿En qué te ayudo hoy?",
        "sim_notice": "Estoy funcionando en Modo Simulado (sin modelo GGUF activo), pero tu memoria vectorial RAG está activa.",
        "busy_init": "Todavía estoy terminando de iniciar. Espera un momento...",
        "busy_gen": "Todavía estoy respondiendo el mensaje anterior. Espera un momento...",
        "net_online": "CONECTADA",
        "net_offline": "DESCONECTADA",
        "help_title": "Comandos disponibles",
        "help_body": (
            "  /help                       Muestra esta ayuda\n"
            "  /logs                       Abre o cierra el apartado de Logs (también Ctrl+L)\n"
            "  /memory                     Ver hechos guardados en memoria vectorial\n"
            "  /remember <dato>            Guarda un dato manualmente en memoria\n"
            "  /forget <consulta>          Elimina recuerdos que coincidan\n"
            "  /search <consulta>          Busca en tu memoria vectorial con puntajes\n"
            "  /clear                      Limpia el historial de chat\n"
            "  /face                       Estado del reconocimiento facial local\n"
            "  /face enroll <nombre> [img] Registra tu rostro (cámara o foto)\n"
            "  /face verify [img]          Verifica tu rostro (cámara o foto)\n"
            "  /face reset                 Borra los perfiles faciales guardados"
        ),
        "unknown_cmd": "Comando no reconocido. Escribe /help para ver la lista.",
    },
    "en": {
        "tab_chat": "Chat",
        "tab_logs": "Logs",
        "placeholder": "Type a message or /help...",
        "status_locked": "● Locked  |  Enter your license key",
        "status_init": "◌ Initializing JAM Core... (Ctrl+L to view logs)",
        "status_ready": "● Ready  |  Ctrl+L: Logs  |  /help: Commands",
        "status_thinking": "◌ Thinking...",
        "status_face": "◌ Scanning face...",
        "license_welcome": "Welcome to JAM. Enter your license key to begin (format JAM-XXXX-YYYY-ZZZZ).",
        "license_corrupt": "Saved license is invalid. Please enter a new valid key.",
        "license_invalid": "Invalid key. Please try again.",
        "license_ok_chat": "License verified. Preparing your local assistant...",
        "onboard_q1": "Hi, I'm JAM. To start remembering your preferences, what is your name?",
        "onboard_q2": "Nice to meet you. What is your main profession or interest?",
        "onboard_q3": "Got it. Do you prefer detailed answers or concise, direct replies?",
        "onboard_done": "All set! Your profile is stored in local vector memory. Type /help to see commands (including /face recognition). How can I help today?",
        "welcome_back": "Welcome back. How can I help you today?",
        "welcome_face": "Welcome back, {user}! Face recognized ({conf:.0%}). How can I help today?",
        "sim_notice": "Running in Simulated Mode (no GGUF model active), with live RAG vector memory enabled.",
        "busy_init": "Still finishing startup. Please wait a moment...",
        "busy_gen": "Still generating the previous reply. Please wait a moment...",
        "net_online": "CONNECTED",
        "net_offline": "OFFLINE",
        "help_title": "Available commands",
        "help_body": (
            "  /help                       Show this help message\n"
            "  /logs                       Toggle the Logs view (or press Ctrl+L)\n"
            "  /memory                     Inspect stored user facts in vector memory\n"
            "  /remember <fact>            Manually save a fact to vector memory\n"
            "  /forget <query>             Delete matching facts from memory\n"
            "  /search <query>             Test hybrid vector search with scores\n"
            "  /clear                      Clear stored chat history\n"
            "  /face                       Show local face recognition status\n"
            "  /face enroll <name> [img]   Enroll your face (webcam or image path)\n"
            "  /face verify [img]          Verify your face (webcam or image path)\n"
            "  /face reset                 Remove saved face profiles"
        ),
        "unknown_cmd": "Unknown command. Type /help to list commands.",
    },
}


def validate_key(key: str) -> bool:
    key = key.strip().upper()
    if not key.startswith("JAM-") or len(key) != 18:
        return False
    core = key[:-2]
    try:
        checksum_provided = int(key[-2:])
        checksum_calculated = sum(ord(c) for c in core.replace("-", "")) % 100
        return checksum_provided == checksum_calculated
    except ValueError:
        return False


def format_vram(vram_value) -> str:
    if isinstance(vram_value, (int, float)):
        return f"{vram_value} GB"
    return str(vram_value)


class DummyMemory:
    def add_user_fact(self, fact, **kwargs):
        return None

    def upsert_user_fact(self, fact, **kwargs):
        return {"id": None, "action": "skipped", "category": None}

    def add_chat_message(self, message, **kwargs):
        return None

    def recall_user_facts(self, q, n_results=3, **kwargs):
        return []

    def recall_user_facts_detailed(self, q, n_results=3, **kwargs):
        return []

    def get_all_user_facts(self, limit=50):
        return []

    def forget_matching_facts(self, query, **kwargs):
        return 0

    def clear_chat_history(self, session_id=None):
        return 0

    def get_stats(self):
        return {"user_facts": 0, "chat_messages": 0, "embedding_backend": "offline"}


class Sidebar(Static):
    def compose(self) -> ComposeResult:
        yield Static(JAM_ASCII_LOGO, id="jam-logo")
        self.sys_status = Static("SYSTEM: LOCKED", classes="system-text")
        yield self.sys_status

        self.face_title = Static("[USER / FACE]", classes="subtitle")
        yield self.face_title
        self.face_info = Static("UNVERIFIED")
        yield self.face_info

        self.env_title = Static("[ENV / COMMS]", classes="subtitle")
        yield self.env_title
        self.env_info = Static("OFFLINE")
        yield self.env_info

        self.hw_title = Static("[HARDWARE]", classes="subtitle")
        yield self.hw_title
        self.hw_info = Static("OFFLINE")
        yield self.hw_info

        self.model_title = Static("[MODELS]", classes="subtitle")
        yield self.model_title
        self.model_info = Static("OFFLINE")
        yield self.model_info

        self.mem_title = Static("[MEMORY / RAG]", classes="subtitle")
        yield self.mem_title
        self.mem_info = Static("OFFLINE")
        yield self.mem_info


class JamApp(App):
    CSS = APP_CSS
    BINDINGS = [
        Binding("ctrl+l", "toggle_logs", "Logs", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        language: str = "es",
        data_dir: str = "./jam_data",
        demo_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lang = language if language in TEXTS else "es"
        self.data_dir = data_dir
        self.demo_mode = demo_mode
        self.session_id = str(uuid.uuid4())[:8]
        self.memory = DummyMemory()
        self.face_auth: JamFaceAuth = JamFaceAuth(faces_dir=self._data_path("faces"))
        self.recognized_user: str | None = None
        self.engine = None
        self.engine_ready = False
        self.memory_ready = False
        self.needs_license = False
        self.is_initializing = True
        self.is_first_time = False
        self.onboarding_step = 0
        self.env_data = {"city": "OFFLINE", "temp": "--", "online": False}
        self._gen_lock = asyncio.Lock()

    def t(self, key: str) -> str:
        return TEXTS[self.lang][key]

    def _data_path(self, *parts: str) -> str:
        return os.path.join(self.data_dir, *parts)

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#status-bar", Static).update(text)
        except Exception:
            pass

    def _write_log(self, message: str, level: str = "INFO") -> None:
        """Write technical/internal telemetry ONLY to the dedicated Logs panel."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color = {
            "INFO": "#7aa2f7",
            "OK": "#00ffcc",
            "WARN": "#e0af68",
            "ERROR": "#f7768e",
        }.get(level, "#99aacc")
        try:
            sys_log = self.query_one("#system-log", RichLog)
            sys_log.write(f"[dim]{ts}[/] [{color}]{escape(message)}[/]")
        except Exception:
            pass

    def _write_chat_user(self, message: str) -> None:
        label = "Tú" if self.lang == "es" else "You"
        if self.recognized_user:
            label = self.recognized_user
        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"\n[bold #00d7ff]{escape(label)}:[/] {escape(message)}")

    def _write_chat_jam(self, message: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"[bold #00ffcc]JAM:[/] {escape(message)}")

    def _write_chat_info(self, title: str, body: str = "") -> None:
        chat = self.query_one("#chat-log", RichLog)
        if body:
            chat.write(f"\n[bold #aaaaaa]{escape(title)}[/]\n[dim]{escape(body)}[/]")
        else:
            chat.write(f"[dim #aaaaaa]{escape(title)}[/]")

    def _refresh_memory_sidebar(self) -> None:
        try:
            stats = self.memory.get_stats()
            backend = stats.get("embedding_backend", "offline")
            sidebar = self.query_one(Sidebar)
            sidebar.mem_info.update(
                f"FACTS: {stats.get('user_facts', 0)}\n"
                f"CHATS: {stats.get('chat_messages', 0)}\n"
                f"EMB: {backend}"
            )
        except Exception:
            pass

    def _refresh_face_sidebar(self, confidence: float | None = None) -> None:
        try:
            sidebar = self.query_one(Sidebar)
            status = self.face_auth.get_status()
            if self.recognized_user:
                conf_txt = f" ({confidence:.0%})" if confidence is not None else ""
                sidebar.face_info.update(f"ID: {self.recognized_user}{conf_txt}\nSTATE: VERIFIED")
            elif not status["available"]:
                sidebar.face_info.update("CV2: OFFLINE")
            elif status["profiles_count"] > 0:
                sidebar.face_info.update(f"PROFILES: {status['profiles_count']}\nSTATE: READY")
            else:
                sidebar.face_info.update("PROFILES: 0\nUSE: /face enroll")
        except Exception:
            pass

    def action_toggle_logs(self) -> None:
        """Toggle between the clean Chat tab and the technical Logs tab."""
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
            tabs.active = "tab-logs" if tabs.active == "tab-chat" else "tab-chat"
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Sidebar(id="sidebar")
            with Vertical(id="main-area"):
                with TabbedContent(initial="tab-chat", id="main-tabs"):
                    with TabPane(self.t("tab_chat"), id="tab-chat"):
                        yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
                    with TabPane(self.t("tab_logs"), id="tab-logs"):
                        yield RichLog(id="system-log", wrap=True, markup=True, highlight=False)
                yield ProgressBar(id="progress-bar", total=None, show_eta=False)
                yield Static(self.t("status_init"), id="status-bar")
                with Container(id="input-container"):
                    yield Input(placeholder=self.t("placeholder"), id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "JAM"
        self._refresh_face_sidebar()
        self.query_one("#chat-input", Input).focus()
        self.run_worker(self.check_license())

    async def check_license(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        license_path = self._data_path("license.key")

        if not os.path.exists(license_path):
            self._write_log("[SYS] Inicialización pausada: falta license.key.", "WARN")
            self._write_chat_jam(self.t("license_welcome"))
            self._set_status(self.t("status_locked"))
            self.needs_license = True
            self.is_initializing = False
        else:
            with open(license_path, "r", encoding="utf-8") as f:
                saved_key = f.read().strip()
            if validate_key(saved_key):
                self._write_log("[SYS] Licencia local validada correctamente.", "OK")
                self.query_one(Sidebar).sys_status.update("SYSTEM: ONLINE")
                self.run_worker(self.resume_init_system())
            else:
                self._write_log("[SYS] Licencia local corrupta o inválida.", "ERROR")
                self._write_chat_jam(self.t("license_corrupt"))
                self._set_status(self.t("status_locked"))
                self.needs_license = True
                self.is_initializing = False

        self.query_one("#chat-input", Input).focus()

    async def resume_init_system(self) -> None:
        self.is_initializing = True
        self._set_status(self.t("status_init"))
        sidebar = self.query_one(Sidebar)

        self._write_log("[SYS] Licencia autenticada. Inicializando JAM Core...", "OK")

        # Env Data & Hardware in worker thread (non-blocking TUI)
        env_task = asyncio.to_thread(get_location_and_weather)
        hw_task = asyncio.to_thread(lambda: HardwareDetector().get_full_report())
        self.env_data, report = await asyncio.gather(env_task, hw_task)

        net_label = self.t("net_online") if self.env_data.get("online") else self.t("net_offline")
        loc_prefix = "UBIC" if self.lang == "es" else "LOC"
        net_prefix = "RED" if self.lang == "es" else "NET"
        sidebar.env_info.update(
            f"{loc_prefix}: {self.env_data['city']}\n"
            f"TMP: {self.env_data['temp']} C\n"
            f"{net_prefix}: {net_label}"
        )

        recommender = ModelRecommender(report)
        recs = recommender.get_recommendations()

        sidebar.hw_info.update(
            f"OS: {report['os']}\n"
            f"RAM: {report['ram']['total_gb']}GB\n"
            f"VRAM: {recs['effective_vram_gb']}GB"
        )

        mng_param = recs["manager"].get("name", "LIGHT").split(" ")[0].replace("-GGUF", "")
        gen_param = recs["generator"].get("name", "HEAVY").split(" ")[0].replace("-GGUF", "")
        sidebar.model_info.update(f"MNG: {mng_param}\nGEN: {gen_param}")

        self._write_log(
            f"[HW] OS={report['os']} | CPU={report['cpu']['architecture']} "
            f"({report['cpu']['physical_cores']} cores) | RAM={report['ram']['total_gb']}GB | "
            f"GPU={report['gpu']['type']} ({format_vram(report['gpu']['vram_gb'])})"
        )
        self._write_log(f"[SYS] Modelo asignado: {recs['generator']['name']}")

        self.run_worker(self.load_ai_engine(recs))

    async def load_ai_engine(self, recs) -> None:
        pb = self.query_one("#progress-bar", ProgressBar)
        pb.display = True

        # 1. Initialize vectorized memory FIRST, independent of LLM
        self._write_log("[RAG] Inicializando memoria vectorial ChromaDB...")

        def _init_memory():
            from jam.memory.vector_db import JamMemory

            return JamMemory(db_path=self._data_path("memory"))

        try:
            self.memory = await asyncio.to_thread(_init_memory)
            self.memory_ready = True
            stats = self.memory.get_stats()
            self._write_log(
                f"[RAG] Base vectorial lista (facts={stats['user_facts']}, "
                f"chats={stats['chat_messages']}, emb={stats['embedding_backend']}).",
                "OK",
            )
            self._refresh_memory_sidebar()
        except Exception as e:
            self.memory_ready = False
            self._write_log(f"[ERROR] Fallo al iniciar ChromaDB: {e}", "ERROR")

        # 2. Load local LLM inference engine (unless demo_mode is forced)
        success = False
        if self.demo_mode:
            self._write_log("[SYS] Modo Demo activado (--demo). Omitiendo carga GGUF.", "WARN")
        else:
            self._write_log("[GEN] Preparando motor local llama.cpp...")

            def _load_llm():
                try:
                    from jam.core.inference import JamInferenceEngine

                    self.engine = JamInferenceEngine(
                        recs["generator"],
                        models_dir=self._data_path("models"),
                        use_gpu=bool(recs.get("use_gpu", False)),
                    )

                    def _log_cb(msg):
                        lvl = "ERROR" if "[ERROR]" in msg else "INFO"
                        self.call_from_thread(self._write_log, msg, lvl)

                    return self.engine.load_model(log_callback=_log_cb)
                except Exception as e:
                    self.call_from_thread(self._write_log, f"[ERROR] {e}", "ERROR")
                    return False

            success = await asyncio.to_thread(_load_llm)

        # 3. Optional automatic face recognition on startup if a profile exists
        face_match = None
        face_status = self.face_auth.get_status()
        if face_status["available"] and face_status["profiles_count"] > 0 and not self.demo_mode:
            self._set_status(self.t("status_face"))

            def _auto_face():
                return self.face_auth.recognize_from_camera(
                    timeout_sec=3.0,
                    log_callback=lambda msg: self.call_from_thread(self._write_log, msg),
                )

            face_match = await asyncio.to_thread(_auto_face)
            if face_match.get("recognized"):
                self.recognized_user = face_match["user"]
                self._refresh_face_sidebar(face_match.get("confidence"))
            else:
                self._write_log(f"[FACE] Auto-verificación omitida/no confirmada ({face_match.get('reason')}).")
        else:
            self._refresh_face_sidebar()

        pb.display = False
        self.engine_ready = success
        self.is_initializing = False
        self._set_status(self.t("status_ready"))

        os.makedirs(self.data_dir, exist_ok=True)
        self.is_first_time = not os.path.exists(self._data_path("onboarding_done"))
        self.onboarding_step = 0

        if not self.engine_ready:
            self._write_chat_info(self.t("sim_notice"))

        if self.is_first_time and self.memory_ready:
            self._write_log("[SYS] Nuevo usuario detectado. Iniciando onboarding.")
            self._write_chat_jam(self.t("onboard_q1"))
            self.onboarding_step = 1
        elif face_match and face_match.get("recognized"):
            self._write_chat_jam(
                self.t("welcome_face").format(
                    user=face_match["user"],
                    conf=face_match["confidence"],
                )
            )
        else:
            self._write_chat_jam(self.t("welcome_back"))

        self.query_one("#chat-input", Input).focus()

    async def _run_face_command(self, arg: str) -> None:
        """Handle /face subcommands asynchronously so webcam capture never freezes TUI."""
        try:
            tokens = shlex.split(arg) if arg else []
        except ValueError:
            tokens = arg.split()

        sub = tokens[0].lower() if tokens else "status"

        if sub == "status":
            st = self.face_auth.get_status()
            users = ", ".join(st["enrolled_users"]) if st["enrolled_users"] else (
                "Ninguno" if self.lang == "es" else "None"
            )
            cv_txt = "OK" if st["available"] else "NO DISPONIBLE"
            lbph_txt = "ON" if st["lbph_available"] else "OFF"
            if self.lang == "es":
                self._write_chat_info(
                    "Reconocimiento facial local",
                    f"  Motor OpenCV: {cv_txt} (LBPH: {lbph_txt})\n"
                    f"  Perfiles registrados ({st['profiles_count']}): {users}\n"
                    f"  Usuario activo: {self.recognized_user or 'No verificado'}",
                )
            else:
                self._write_chat_info(
                    "Local Face Recognition",
                    f"  OpenCV Engine: {cv_txt} (LBPH: {lbph_txt})\n"
                    f"  Enrolled profiles ({st['profiles_count']}): {users}\n"
                    f"  Active user: {self.recognized_user or 'Unverified'}",
                )
            return

        if sub == "reset":
            removed = self.face_auth.delete_profile(None)
            self.recognized_user = None
            self._refresh_face_sidebar()
            self._write_log(f"[FACE] Perfiles faciales eliminados: {removed}.", "WARN")
            self._write_chat_info(
                f"Perfiles faciales eliminados ({removed})."
                if self.lang == "es"
                else f"Removed {removed} saved face profile(s)."
            )
            return

        if sub == "enroll":
            if len(tokens) < 2:
                self._write_chat_info(
                    "Uso: /face enroll <nombre> [ruta_foto_opcional]"
                    if self.lang == "es"
                    else "Usage: /face enroll <name> [optional_image_path]"
                )
                return

            user_name = tokens[1]
            image_path = tokens[2] if len(tokens) >= 3 else None

            self._set_status(self.t("status_face"))
            if image_path:
                self._write_log(f"[FACE] Registrando rostro desde imagen '{image_path}' para {user_name}...")
                result = await asyncio.to_thread(
                    self.face_auth.enroll_from_image_file, user_name, image_path
                )
            else:
                self._write_chat_info(
                    f"Mira a la cámara unos segundos para registrar el rostro de {user_name}..."
                    if self.lang == "es"
                    else f"Look at the camera for a few seconds to enroll {user_name}..."
                )
                result = await asyncio.to_thread(
                    self.face_auth.enroll_from_camera,
                    user_name,
                    8,
                    12.0,
                    0,
                    lambda msg: self.call_from_thread(self._write_log, msg),
                )

            self._set_status(self.t("status_ready"))
            if result.get("success"):
                self.recognized_user = user_name
                self._refresh_face_sidebar(1.0)
                if self.memory_ready:
                    fact = (
                        f"El usuario se llama: {user_name}"
                        if self.lang == "es"
                        else f"The user's name is: {user_name}"
                    )
                    self.memory.upsert_user_fact(
                        fact, category="profile", importance=0.95, source="face_enroll"
                    )
                    self._refresh_memory_sidebar()
                self._write_log(
                    f"[FACE] Rostro registrado para '{user_name}' ({result.get('samples')} muestras).",
                    "OK",
                )
                self._write_chat_jam(
                    f"Rostro de {user_name} registrado localmente ({result.get('samples')} muestras)."
                    if self.lang == "es"
                    else f"Face enrolled locally for {user_name} ({result.get('samples')} samples)."
                )
            else:
                err = result.get("error", "Error desconocido")
                self._write_log(f"[FACE] Falló el registro: {err}", "ERROR")
                self._write_chat_info(
                    f"No se pudo registrar el rostro: {err}"
                    if self.lang == "es"
                    else f"Face enrollment failed: {err}"
                )
            return

        if sub == "verify":
            image_path = tokens[1] if len(tokens) >= 2 else None
            self._set_status(self.t("status_face"))
            if image_path:
                self._write_log(f"[FACE] Verificando rostro desde '{image_path}'...")
                result = await asyncio.to_thread(self.face_auth.recognize_from_image_file, image_path)
            else:
                self._write_chat_info(
                    "Escaneando cámara..." if self.lang == "es" else "Scanning camera..."
                )
                result = await asyncio.to_thread(
                    self.face_auth.recognize_from_camera,
                    5.0,
                    0,
                    0.72,
                    lambda msg: self.call_from_thread(self._write_log, msg),
                )

            self._set_status(self.t("status_ready"))
            if result.get("recognized"):
                self.recognized_user = result["user"]
                self._refresh_face_sidebar(result.get("confidence"))
                self._write_log(
                    f"[FACE] Verificado: {result['user']} ({result['confidence']:.0%}).",
                    "OK",
                )
                self._write_chat_jam(
                    self.t("welcome_face").format(
                        user=result["user"],
                        conf=result["confidence"],
                    )
                )
            else:
                reason = result.get("reason", "unknown")
                self._write_log(f"[FACE] No reconocido (motivo={reason}, sim={result.get('confidence', 0):.0%}).", "WARN")
                self._write_chat_info(
                    f"Rostro no reconocido ({reason})."
                    if self.lang == "es"
                    else f"Face not recognized ({reason})."
                )
            return

        self._write_chat_info(
            "Subcomandos de /face: status | enroll <nombre> [img] | verify [img] | reset"
            if self.lang == "es"
            else "/face subcommands: status | enroll <name> [img] | verify [img] | reset"
        )

    def _handle_slash_command(self, command_text: str) -> bool:
        """Handle built-in TUI commands for memory, logs, and face recognition."""
        parts = command_text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            self._write_chat_info(self.t("help_title"), self.t("help_body"))
            return True

        if cmd == "/logs":
            self.action_toggle_logs()
            return True

        if cmd == "/face":
            self.run_worker(self._run_face_command(arg))
            return True

        if cmd == "/memory":
            stats = self.memory.get_stats()
            facts = self.memory.get_all_user_facts(limit=20)
            header = (
                f"Memoria RAG (Hechos: {stats.get('user_facts', 0)} | "
                f"Chats: {stats.get('chat_messages', 0)} | "
                f"Embeddings: {stats.get('embedding_backend', 'unknown')})"
            )
            if not facts:
                body = "  (Sin hechos guardados aún)" if self.lang == "es" else "  (No facts stored yet)"
            else:
                body = "\n".join(
                    f"  {idx}. [{fact['category']}] {fact['text']}"
                    for idx, fact in enumerate(facts, 1)
                )
            self._write_chat_info(header, body)
            return True

        if cmd == "/remember":
            if not arg:
                self._write_chat_info(
                    "Uso: /remember <dato sobre ti>"
                    if self.lang == "es"
                    else "Usage: /remember <fact about you>"
                )
                return True
            res = self.memory.upsert_user_fact(arg, source="manual")
            self._refresh_memory_sidebar()
            self._write_log(f"[RAG] /remember -> action={res.get('action')} cat={res.get('category')}")
            self._write_chat_info(
                f"Guardado en memoria ({res.get('action')}): {arg}"
                if self.lang == "es"
                else f"Saved to memory ({res.get('action')}): {arg}"
            )
            return True

        if cmd == "/forget":
            if not arg:
                self._write_chat_info(
                    "Uso: /forget <tema a olvidar>"
                    if self.lang == "es"
                    else "Usage: /forget <topic to forget>"
                )
                return True
            deleted = self.memory.forget_matching_facts(arg)
            self._refresh_memory_sidebar()
            self._write_log(f"[RAG] /forget '{arg}' -> eliminados={deleted}")
            self._write_chat_info(
                f"Se eliminaron {deleted} recuerdo(s)."
                if self.lang == "es"
                else f"Removed {deleted} matching memory record(s)."
            )
            return True

        if cmd == "/search":
            if not arg:
                self._write_chat_info(
                    "Uso: /search <consulta>"
                    if self.lang == "es"
                    else "Usage: /search <query>"
                )
                return True
            matches = self.memory.recall_user_facts_detailed(arg, n_results=5, min_score=0.15)
            if not matches:
                self._write_chat_info(
                    "Sin coincidencias relevantes."
                    if self.lang == "es"
                    else "No relevant matches found."
                )
            else:
                lines = "\n".join(
                    f"  score={m['score']:.2f} (sem={m['semantic_score']:.2f}, lex={m['lexical_score']:.2f}) -> {m['text']}"
                    for m in matches
                )
                self._write_chat_info(f"Resultados para '{arg}':", lines)
            return True

        if cmd == "/clear":
            cleared = self.memory.clear_chat_history()
            self._refresh_memory_sidebar()
            self._write_log(f"[RAG] Historial limpiado ({cleared} mensajes).")
            self._write_chat_info(
                f"Historial de chat limpiado ({cleared} mensajes)."
                if self.lang == "es"
                else f"Chat history cleared ({cleared} messages)."
            )
            return True

        self._write_chat_info(self.t("unknown_cmd"))
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        input_widget = self.query_one("#chat-input", Input)

        if user_input == "":
            return

        input_widget.value = ""

        # Manejo de Licencia
        if self.needs_license:
            if validate_key(user_input):
                normalized_key = user_input.upper()
                with open(self._data_path("license.key"), "w", encoding="utf-8") as f:
                    f.write(normalized_key)
                masked = f"{normalized_key[:8]}-****-****"
                self._write_log(f"[SYS] Nueva licencia activada ({masked}).", "OK")
                self._write_chat_info(self.t("license_ok_chat"))
                self.needs_license = False
                self.query_one(Sidebar).sys_status.update("SYSTEM: ONLINE")
                self.run_worker(self.resume_init_system())
            else:
                self._write_log("[SYS] Intento de licencia inválida.", "WARN")
                self._write_chat_jam(self.t("license_invalid"))
            return

        if self.is_initializing:
            self._write_chat_info(self.t("busy_init"))
            return

        # Comandos internos (no se envían al LLM)
        if user_input.startswith("/"):
            self._handle_slash_command(user_input)
            return

        # Mostrar siempre en el tab de Chat al enviar un mensaje normal
        try:
            self.query_one("#main-tabs", TabbedContent).active = "tab-chat"
        except Exception:
            pass

        self._write_chat_user(user_input)

        # Protocolo de Onboarding
        if self.is_first_time and self.memory_ready:
            if self.onboarding_step == 1:
                fact = (
                    f"El usuario se llama: {user_input}"
                    if self.lang == "es"
                    else f"The user's name is: {user_input}"
                )
                self.memory.upsert_user_fact(fact, category="profile", importance=0.95, source="onboarding")
                self._refresh_memory_sidebar()
                self._write_log(f"[RAG] Onboarding paso 1 guardado: {fact}")
                self._write_chat_jam(self.t("onboard_q2"))
                self.onboarding_step = 2
            elif self.onboarding_step == 2:
                fact = (
                    f"La profesión o interés principal del usuario es: {user_input}"
                    if self.lang == "es"
                    else f"The user's main profession or interest is: {user_input}"
                )
                self.memory.upsert_user_fact(fact, category="work", importance=0.90, source="onboarding")
                self._refresh_memory_sidebar()
                self._write_log(f"[RAG] Onboarding paso 2 guardado: {fact}")
                self._write_chat_jam(self.t("onboard_q3"))
                self.onboarding_step = 3
            elif self.onboarding_step == 3:
                fact = (
                    f"Estilo de respuesta preferido del usuario: {user_input}"
                    if self.lang == "es"
                    else f"User's preferred response style: {user_input}"
                )
                self.memory.upsert_user_fact(fact, category="preference", importance=0.90, source="onboarding")
                self._refresh_memory_sidebar()
                self._write_log(f"[RAG] Onboarding completado y perfil persistido.", "OK")
                self._write_chat_jam(self.t("onboard_done"))
                with open(self._data_path("onboarding_done"), "w", encoding="utf-8") as f:
                    f.write("done")
                self.is_first_time = False
                self.onboarding_step = 0
            return

        if self._gen_lock.locked():
            self._write_chat_info(self.t("busy_gen"))
            return

        self.run_worker(self.generate_ai_response(user_input))

    async def generate_ai_response(self, user_input: str) -> None:
        async with self._gen_lock:
            self._set_status(self.t("status_thinking"))
            self._write_log("[MNG] Orquestando memoria vectorial híbrida...")

            def _process():
                orchestrator = JamOrchestrator(
                    self.memory,
                    self.env_data,
                    session_id=self.session_id,
                )
                pipeline_result = orchestrator.process_pipeline(
                    user_input,
                    lang=self.lang,
                )

                for step_log in pipeline_result["logs"]:
                    self.call_from_thread(self._write_log, step_log)

                if self.engine_ready and self.engine is not None:
                    self.call_from_thread(self._write_log, "[GEN] Generando respuesta neuronal...")
                    response = self.engine.generate_chat(
                        pipeline_result["messages"],
                        max_tokens=256,
                    )
                    self.call_from_thread(self._write_log, "[GEN] Respuesta generada correctamente.", "OK")
                else:
                    recalled = pipeline_result.get("recalled_facts", [])
                    if self.lang == "es":
                        ctx_note = (
                            f" (Recordé {len(recalled)} dato(s) tuyos)."
                            if recalled
                            else ""
                        )
                        response = f"Recibí tu mensaje en Modo Simulado: «{user_input}».{ctx_note}"
                    else:
                        ctx_note = (
                            f" (Recalled {len(recalled)} user fact(s))."
                            if recalled
                            else ""
                        )
                        response = f"Received in Simulated Mode: \"{user_input}\".{ctx_note}"

                orchestrator.record_interaction(user_input, response)
                return response

            try:
                ai_response = await asyncio.to_thread(_process)
            except Exception as e:
                self._write_log(f"[ERROR] Fallo en generación: {e}", "ERROR")
                ai_response = (
                    "Ocurrió un error al generar la respuesta. Revisa la pestaña Logs (Ctrl+L)."
                    if self.lang == "es"
                    else "An error occurred while generating the response. Check the Logs tab (Ctrl+L)."
                )

            self._refresh_memory_sidebar()
            self._set_status(self.t("status_ready"))
            self._write_chat_jam(ai_response)

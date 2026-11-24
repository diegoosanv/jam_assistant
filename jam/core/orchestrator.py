import json
import re
from typing import Any, Dict, List, Optional


_FACT_PATTERNS = [
    # Spanish
    (re.compile(r"\b(?:me llamo|mi nombre es|soy)\s+([^?.!,\n]{2,60})", re.IGNORECASE), "profile"),
    (re.compile(r"\b(?:trabajo (?:en|como|de)|soy (?:ingenier[oa]|desarrollador[a]?|programador[a]?|estudiante|doctor[a]?|profesor[a]?))\s*([^?.!\n]{0,80})", re.IGNORECASE), "work"),
    (re.compile(r"\b(?:vivo en|soy de)\s+([^?.!\n]{2,60})", re.IGNORECASE), "location"),
    (re.compile(r"\b(?:me gusta|prefiero|mi [a-zA-Záéíóúñ]+ favorito es)\s+([^?.!\n]{2,80})", re.IGNORECASE), "preference"),
    (re.compile(r"\b(?:recuerda que|guarda que)\s+(.{3,160})", re.IGNORECASE), "fact"),
    # English
    (re.compile(r"\b(?:my name is|call me)\s+([^?.!,\n]{2,60})", re.IGNORECASE), "profile"),
    (re.compile(r"\b(?:i work (?:as|at|in)|i am a|i'm a|i study)\s+([^?.!\n]{2,80})", re.IGNORECASE), "work"),
    (re.compile(r"\b(?:i live in|i am from|i'm from)\s+([^?.!\n]{2,60})", re.IGNORECASE), "location"),
    (re.compile(r"\b(?:i like|i love|i prefer|my favorite [a-z]+ is)\s+([^?.!\n]{2,80})", re.IGNORECASE), "preference"),
    (re.compile(r"\b(?:remember that)\s+(.{3,160})", re.IGNORECASE), "fact"),
]


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON object from LLM output, even if wrapped in markdown."""
    if not raw_text:
        return None
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def heuristic_analyze_intent(user_text: str) -> Dict[str, Any]:
    """
    Fast bilingual (ES/EN) heuristic intent analyzer used when no separate MNG
    model is invoked or as a fallback if JSON parsing fails.
    """
    cleaned = (user_text or "").strip()
    result: Dict[str, Any] = {
        "is_new_fact": False,
        "fact_to_save": None,
        "category": None,
        "requires_search": False,
        "search_query": None,
    }

    if len(cleaned) < 4:
        return result

    # Ignore pure questions (unless explicitly asking to remember something)
    lower = cleaned.lower()
    explicit_remember = lower.startswith(("recuerda que", "guarda que", "remember that"))
    if cleaned.endswith("?") and not explicit_remember:
        return result

    for pattern, category in _FACT_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            fact_text = match.group(1).strip() if explicit_remember else cleaned
            if len(fact_text) >= 3:
                result["is_new_fact"] = True
                result["fact_to_save"] = fact_text
                result["category"] = category
                break

    return result


class JamOrchestrator:
    def __init__(self, memory_db, env_data: Optional[Dict[str, Any]] = None, session_id: str = "default"):
        self.memory = memory_db
        self.env_data = env_data or {"city": "OFFLINE", "temp": "--"}
        self.session_id = session_id

    def analyze_intent_prompt(self, user_text: str) -> str:
        """
        Genera el prompt que leerá el modelo ligero (MNG) para clasificar la intención.
        Obliga al modelo pequeño a devolver un JSON estructurado.
        """
        return f"""[SYSTEM]
You are JAM-MNG, the fast orchestrator of the JAM system.
Your job is to analyze the user's input and extract structured data.
Do NOT answer the user's question. Output ONLY a valid JSON object.

User Input: "{user_text}"

Determine if the user is providing a new fact about themselves, or asking something that requires internet access.

Format required:
{{
    "is_new_fact": true/false,
    "fact_to_save": "The extracted fact in 3rd person or null",
    "requires_search": true/false,
    "search_query": "Optimized search query or null"
}}
"""

    def build_system_prompt(self, extracted_facts: str, recalled_chat_str: str = "", web_context: str = "") -> str:
        """Construye el mensaje de sistema enriquecido con entorno y memoria RAG."""
        city = self.env_data.get("city", "OFFLINE")
        temp = self.env_data.get("temp", "--")
        env_context = f"Location: {city} | Temp: {temp} C"

        system_prompt = (
            "You are JAM (Just Another Mind), an advanced local AI assistant "
            "running privately on the user's own hardware. Keep answers concise, "
            "helpful, technical, and accurate.\n\n"
            "CRITICAL RULES:\n"
            "1. Always reply in the exact same language the user uses in their latest message "
            "(Spanish -> Spanish, English -> English).\n"
            "2. Use [USER MEMORY / PROFILE] to know details about the human user "
            "(their name, work, preferences); your own assistant name is JAM.\n\n"
            f"[ENVIRONMENT DATA]\n{env_context}\n\n"
            f"[USER MEMORY / PROFILE]\n{extracted_facts}"
        )

        if recalled_chat_str:
            system_prompt += f"\n\n[RELEVANT PAST CONVERSATIONS]\n{recalled_chat_str}"

        if web_context:
            system_prompt += f"\n\n[WEB CONTEXT]\n{web_context}"

        return system_prompt

    def build_generator_prompt(
        self,
        user_text: str,
        extracted_facts: str,
        web_context: str = "",
        recalled_chat_str: str = "",
    ) -> str:
        """
        Construye el prompt final en formato plano (compatibilidad hacia atrás).
        """
        system_prompt = self.build_system_prompt(
            extracted_facts=extracted_facts,
            recalled_chat_str=recalled_chat_str,
            web_context=web_context,
        )
        return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_text}\n[JAM]\n"

    def build_chat_messages(
        self,
        user_text: str,
        extracted_facts: str,
        recent_chat: Optional[List[Dict[str, Any]]] = None,
        recalled_chat_str: str = "",
        web_context: str = "",
    ) -> List[Dict[str, str]]:
        """
        Construye mensajes estructurados (role/content) para llama.cpp create_chat_completion.
        """
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    extracted_facts=extracted_facts,
                    recalled_chat_str=recalled_chat_str,
                    web_context=web_context,
                ),
            }
        ]

        for turn in recent_chat or []:
            role = turn.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            text = (turn.get("text") or "").strip()
            if text:
                messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": user_text})
        return messages

    def record_interaction(self, user_text: str, assistant_response: str) -> None:
        """Persiste el intercambio actual en la memoria episódica de chat."""
        if hasattr(self.memory, "add_chat_message"):
            if user_text and user_text.strip():
                self.memory.add_chat_message(
                    user_text.strip(),
                    role="user",
                    session_id=self.session_id,
                )
            if assistant_response and assistant_response.strip():
                self.memory.add_chat_message(
                    assistant_response.strip(),
                    role="assistant",
                    session_id=self.session_id,
                )

    def process_pipeline(
        self,
        user_text: str,
        mock_mng_response: Optional[Dict[str, Any]] = None,
        lang: str = "es",
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de orquestación y memoria vectorial híbrida.
        """
        mng_data = mock_mng_response if mock_mng_response is not None else heuristic_analyze_intent(user_text)
        logs: List[str] = []

        # 1. Guardar / actualizar en memoria vectorial si hay nuevos datos personales
        if mng_data.get("is_new_fact") and mng_data.get("fact_to_save"):
            fact_text = mng_data["fact_to_save"]
            category = mng_data.get("category")
            if hasattr(self.memory, "upsert_user_fact"):
                upsert_res = self.memory.upsert_user_fact(
                    fact_text,
                    category=category,
                    source="conversation",
                )
                action = upsert_res.get("action")
                if action == "added":
                    logs.append(
                        f"[MNG] Nuevo dato vectorizado y guardado ({upsert_res.get('category')}): {fact_text}"
                        if lang == "es"
                        else f"[MNG] New fact vectorized & saved ({upsert_res.get('category')}): {fact_text}"
                    )
                elif action == "updated":
                    logs.append(
                        f"[MNG] Dato existente actualizado en memoria: {fact_text}"
                        if lang == "es"
                        else f"[MNG] Existing memory updated: {fact_text}"
                    )
                elif action == "duplicate":
                    logs.append(
                        "[MNG] Dato ya presente en memoria (duplicado evitado)."
                        if lang == "es"
                        else "[MNG] Fact already stored in memory (duplicate avoided)."
                    )
            else:
                self.memory.add_user_fact(fact_text)
                logs.append(
                    f"[MNG] Nuevo dato guardado: {fact_text}"
                    if lang == "es"
                    else f"[MNG] New fact saved: {fact_text}"
                )

        # 2. Recuperar contexto multicapa (perfil + facts relevantes + chat reciente + chat histórico)
        recalled_facts: List[str] = []
        recent_chat: List[Dict[str, Any]] = []
        recalled_chat_items: List[Dict[str, Any]] = []

        if hasattr(self.memory, "build_memory_context"):
            mem_ctx = self.memory.build_memory_context(
                query=user_text,
                session_id=self.session_id,
                max_facts=5,
                max_recent_turns=4,
                max_recalled_chats=3,
            )
            recalled_facts = mem_ctx["facts"]
            recent_chat = mem_ctx["recent_chat"]
            recalled_chat_items = mem_ctx["recalled_chat"]
        else:
            recalled_facts = self.memory.recall_user_facts(user_text, n_results=3)

        facts_str = (
            "\n".join(f"- {fact}" for fact in recalled_facts)
            if recalled_facts
            else "No prior user context available."
        )
        recalled_chat_str = "\n".join(
            f"- [{item.get('metadata', {}).get('role', 'chat')}] {item['text']}"
            for item in recalled_chat_items
        )

        if recalled_facts or recent_chat or recalled_chat_items:
            logs.append(
                f"[MNG] Memoria recuperada: {len(recalled_facts)} hecho(s), "
                f"{len(recent_chat)} turno(s) recientes, {len(recalled_chat_items)} recuerdo(s) previos."
                if lang == "es"
                else f"[MNG] Memory recalled: {len(recalled_facts)} fact(s), "
                f"{len(recent_chat)} recent turn(s), {len(recalled_chat_items)} past excerpt(s)."
            )

        # 3. Búsqueda web si es necesario (placeholder para futura integración)
        web_str = ""
        if mng_data.get("requires_search") and mng_data.get("search_query"):
            query = mng_data["search_query"]
            logs.append(
                f"[MNG] Búsqueda web solicitada (aún no conectada): '{query}'"
                if lang == "es"
                else f"[MNG] Web search requested (not yet connected): '{query}'"
            )

        # 4. Construir mensajes estructurados y prompt plano
        messages = self.build_chat_messages(
            user_text=user_text,
            extracted_facts=facts_str,
            recent_chat=recent_chat,
            recalled_chat_str=recalled_chat_str,
            web_context=web_str,
        )
        gen_prompt = self.build_generator_prompt(
            user_text=user_text,
            extracted_facts=facts_str,
            web_context=web_str,
            recalled_chat_str=recalled_chat_str,
        )
        logs.append(
            "[MNG] Prompt enriquecido transferido al modelo GEN."
            if lang == "es"
            else "[MNG] Enriched prompt transferred to GEN model."
        )

        return {
            "gen_prompt": gen_prompt,
            "messages": messages,
            "recalled_facts": recalled_facts,
            "recent_chat": recent_chat,
            "recalled_chat": recalled_chat_items,
            "logs": logs,
        }

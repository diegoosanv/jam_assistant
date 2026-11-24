import hashlib
import math
import os
import re
import time
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from chromadb.utils import embedding_functions


_STOPWORDS = {
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "sin", "sobre", "entre", "hacia", "desde",
    "y", "o", "u", "e", "que", "cual", "cuales", "quien", "quienes", "como",
    "cuando", "donde", "es", "son", "fue", "era", "ser", "estar", "esta",
    "estan", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "mi", "mis", "tu", "tus", "su", "sus", "me", "te", "se", "nos", "le", "les",
    "lo", "yo", "tu", "el", "ella", "nosotros", "ellos", "ellas", "muy", "mas",
    "menos", "pero", "si", "no", "ya", "todo", "toda", "todos", "todas", "hay",
    "usuario", "llama",
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "can", "this", "that", "these", "those", "it", "its",
    "my", "your", "his", "her", "our", "their", "me", "you", "he", "she", "we",
    "they", "i", "am", "what", "which", "who", "how", "when", "where", "why",
    "very", "not", "user",
}


def strip_accents(text: str) -> str:
    """Remove diacritics for accent-insensitive Spanish/English matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and collapse whitespace."""
    cleaned = strip_accents(text.strip().lower())
    return re.sub(r"\s+", " ", cleaned)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extract alphanumeric tokens from accent-stripped normalized text."""
    norm = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", norm)
    if not remove_stopwords:
        return tokens
    filtered = [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
    return filtered if filtered else [t for t in tokens if len(t) > 1]


def char_ngrams(text: str, n: int = 3) -> set:
    """Build character n-grams over normalized words for morphological resilience."""
    norm = re.sub(r"[^a-z0-9 ]+", "", normalize_text(text))
    grams = set()
    for word in norm.split():
        if len(word) < n:
            if len(word) > 1:
                grams.add(word)
            continue
        for i in range(len(word) - n + 1):
            grams.add(word[i : i + n])
    return grams


def lexical_similarity(query: str, document: str) -> float:
    """
    Compute hybrid lexical overlap combining token F1 and character trigram overlap.
    Helps dramatically with Spanish queries and proper nouns where English-centric
    embeddings (all-MiniLM-L6-v2) can underscore exact matches.
    """
    q_tokens = set(tokenize(query, remove_stopwords=True))
    d_tokens = set(tokenize(document, remove_stopwords=True))

    token_score = 0.0
    if q_tokens and d_tokens:
        overlap = len(q_tokens & d_tokens)
        if overlap > 0:
            precision = overlap / len(d_tokens)
            recall = overlap / len(q_tokens)
            token_score = (2 * precision * recall) / (precision + recall)
            # Boost when most query keywords appear in the document
            token_score = 0.55 * recall + 0.45 * token_score

    q_grams = char_ngrams(query, n=3)
    d_grams = char_ngrams(document, n=3)
    gram_score = 0.0
    if q_grams and d_grams:
        g_overlap = len(q_grams & d_grams)
        if g_overlap > 0:
            gram_score = g_overlap / max(len(q_grams), 1)

    return min(1.0, 0.70 * token_score + 0.30 * gram_score)


def chunk_text(text: str, max_chars: int = 480, overlap: int = 80) -> List[str]:
    """
    Split long text into overlapping chunks respecting sentence/word boundaries
    so embedding models with 256-token limits do not truncate important details.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    overlap = max(0, min(overlap, max_chars // 3))

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            # Prefer breaking on sentence boundary or space
            split_at = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if split_at > start + max_chars // 2:
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def infer_fact_category(fact_text: str) -> str:
    """Infer a lightweight semantic category for a user fact."""
    norm = normalize_text(fact_text)
    if any(k in norm for k in ("se llama", "me llamo", "nombre", "name is", "call me")):
        return "profile"
    if any(k in norm for k in ("prefier", "estilo", "gusta", "favorit", "prefer", "like", "style")):
        return "preference"
    if any(k in norm for k in ("profesion", "trabajo", "desarroll", "ingenier", "estud", "work", "engineer", "developer", "student")):
        return "work"
    if any(k in norm for k in ("vivo en", "ciudad", "pais", "live in", "from ", "location")):
        return "location"
    return "fact"


class LocalHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Deterministic offline 384-dim feature-hashing embedding function using
    accent-insensitive word unigrams, bigrams, and character trigrams.
    Used as an automatic fallback if ONNX weights cannot be downloaded offline
    or injected directly in fast offline tests.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "jam_local_hash"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "LocalHashEmbeddingFunction":
        return LocalHashEmbeddingFunction(dim=int(config.get("dim", 384)))

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text, remove_stopwords=False)

        features: List[tuple] = []
        for tok in tokens:
            weight = 0.35 if tok in _STOPWORDS else 1.2
            features.append((f"w:{tok}", weight))
        for i in range(len(tokens) - 1):
            features.append((f"b:{tokens[i]}_{tokens[i + 1]}", 0.9))
        for gram in char_ngrams(text, n=3):
            features.append((f"g:{gram}", 0.45))

        if not features:
            vec[0] = 1.0
            return vec

        for feat, weight in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(str(doc)) for doc in input]


class JamMemory:
    """
    Enhanced local vectorized memory (RAG) backed by ChromaDB:
    - Privacy-first (anonymized telemetry disabled)
    - Cosine vector space with graceful legacy collection compatibility
    - Exact + semantic deduplication (upsert instead of blind duplication)
    - Hybrid reranking: semantic cosine similarity + bilingual lexical overlap
      + importance weighting + recency decay
    - Long-text chunking with overlap
    - Separate long-term user persona facts & episodic chat history + sliding
      recent conversation window
    - Automatic offline fallback embedding function if ONNX cannot initialize
    """

    def __init__(
        self,
        db_path: str = "./jam_data/memory",
        embedding_function: Optional[EmbeddingFunction] = None,
        dedup_threshold: float = 0.12,
    ):
        self.db_path = db_path
        self.dedup_threshold = dedup_threshold
        os.makedirs(self.db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.embedding_backend = "custom"
        self.embedding_function = embedding_function
        if self.embedding_function is None:
            self.embedding_function, self.embedding_backend = self._init_default_embeddings()
        elif isinstance(self.embedding_function, LocalHashEmbeddingFunction):
            self.embedding_backend = "local_hash"

        self.user_collection = self._get_or_create_cosine_collection("user_persona")
        self.chat_collection = self._get_or_create_cosine_collection("chat_history")

    def _init_default_embeddings(self):
        """Use Chroma's default ONNX embedding function, falling back offline if needed."""
        try:
            ef = embedding_functions.DefaultEmbeddingFunction()
            # Probe a tiny string to ensure ONNX runtime & weights are operational
            ef(["jam memory init"])
            return ef, "onnx_minilm"
        except Exception:
            return LocalHashEmbeddingFunction(), "local_hash_fallback"

    def _get_or_create_cosine_collection(self, name: str):
        """
        Create collection with cosine distance space, while seamlessly opening
        legacy collections that may already exist with default L2 space.
        """
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function,
            )
        except Exception:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
            )

    @staticmethod
    def _collection_space(collection) -> str:
        meta = getattr(collection, "metadata", None) or {}
        return str(meta.get("hnsw:space", "l2")).lower()

    @staticmethod
    def _distance_to_similarity(distance: float, space: str) -> float:
        if distance is None:
            return 0.0
        if space == "cosine":
            # Cosine distance in ChromaDB is 1 - cosine_similarity in [0, 2]
            return max(0.0, min(1.0, 1.0 - float(distance)))
        # L2 distance fallback
        return 1.0 / (1.0 + max(0.0, float(distance)))

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

    @staticmethod
    def _recency_score(timestamp: Optional[float], half_life_hours: float) -> float:
        if not timestamp:
            return 0.5
        age_hours = max(0.0, (time.time() - float(timestamp)) / 3600.0)
        return math.exp(-math.log(2) * (age_hours / max(half_life_hours, 1.0)))

    def upsert_user_fact(
        self,
        fact_text: str,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        source: str = "conversation",
        dedup_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Store a user fact with metadata, exact hash deduplication, and semantic
        near-duplicate merging. Returns {'id', 'action', 'category'}.
        """
        cleaned = (fact_text or "").strip()
        if not cleaned:
            return {"id": None, "action": "skipped", "category": None}

        now = time.time()
        norm_hash = self._hash_text(cleaned)
        cat = category or infer_fact_category(cleaned)
        imp = float(importance) if importance is not None else (0.9 if source == "onboarding" or cat == "profile" else 0.7)
        imp = max(0.1, min(1.0, imp))
        threshold = self.dedup_threshold if dedup_threshold is None else dedup_threshold

        # 1. Exact deduplication by normalized hash
        try:
            existing = self.user_collection.get(
                where={"normalized_hash": norm_hash},
                include=["documents", "metadatas"],
            )
            if existing and existing.get("ids"):
                existing_id = existing["ids"][0]
                old_meta = (existing.get("metadatas") or [{}])[0] or {}
                updated_meta = {
                    "category": old_meta.get("category", cat),
                    "importance": max(float(old_meta.get("importance", 0.7)), imp),
                    "source": old_meta.get("source", source),
                    "created_at": float(old_meta.get("created_at", now)),
                    "updated_at": now,
                    "normalized_hash": norm_hash,
                }
                self.user_collection.update(
                    ids=[existing_id],
                    documents=[cleaned],
                    metadatas=[updated_meta],
                )
                return {"id": existing_id, "action": "duplicate", "category": updated_meta["category"]}
        except Exception:
            pass

        # 2. Semantic near-duplicate detection (only for single-chunk facts)
        chunks = chunk_text(cleaned, max_chars=480, overlap=80)
        if len(chunks) == 1 and self.user_collection.count() > 0 and threshold > 0:
            try:
                space = self._collection_space(self.user_collection)
                probe = self.user_collection.query(
                    query_texts=[cleaned],
                    n_results=1,
                    include=["documents", "metadatas", "distances"],
                )
                ids = probe.get("ids", [[]])[0]
                dists = probe.get("distances", [[]])[0]
                docs = probe.get("documents", [[]])[0]
                metas = probe.get("metadatas", [[]])[0]
                if ids and dists:
                    sim = self._distance_to_similarity(dists[0], space)
                    if (1.0 - sim) <= threshold:
                        existing_id = ids[0]
                        old_meta = metas[0] or {}
                        old_doc = docs[0] or ""
                        merged_doc = cleaned if len(cleaned) >= len(old_doc) else old_doc
                        updated_meta = {
                            "category": cat if cat != "fact" else old_meta.get("category", cat),
                            "importance": max(float(old_meta.get("importance", 0.7)), imp),
                            "source": source if source == "onboarding" else old_meta.get("source", source),
                            "created_at": float(old_meta.get("created_at", now)),
                            "updated_at": now,
                            "normalized_hash": self._hash_text(merged_doc),
                        }
                        self.user_collection.update(
                            ids=[existing_id],
                            documents=[merged_doc],
                            metadatas=[updated_meta],
                        )
                        return {"id": existing_id, "action": "updated", "category": updated_meta["category"]}
            except Exception:
                pass

        # 3. Insert new fact (chunked if long)
        parent_id = str(uuid.uuid4())
        ids_to_add: List[str] = []
        docs_to_add: List[str] = []
        metas_to_add: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = parent_id if len(chunks) == 1 else f"{parent_id}_c{idx}"
            ids_to_add.append(chunk_id)
            docs_to_add.append(chunk)
            metas_to_add.append({
                "category": cat,
                "importance": imp,
                "source": source,
                "created_at": now,
                "updated_at": now,
                "normalized_hash": self._hash_text(chunk),
                "chunk_index": idx,
                "chunk_total": len(chunks),
                "parent_id": parent_id,
            })

        self.user_collection.add(
            ids=ids_to_add,
            documents=docs_to_add,
            metadatas=metas_to_add,
        )
        return {"id": ids_to_add[0], "action": "added", "category": cat}

    def add_user_fact(
        self,
        fact_text: str,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        source: str = "conversation",
    ) -> Optional[str]:
        """Store key user information in long-term memory (backward-compatible API)."""
        result = self.upsert_user_fact(
            fact_text=fact_text,
            category=category,
            importance=importance,
            source=source,
        )
        return result["id"]

    def add_chat_message(
        self,
        message: str,
        role: str = "user",
        session_id: str = "default",
    ) -> Optional[str]:
        """Store a chat message in episodic vector memory with role/timestamp/chunking."""
        cleaned = (message or "").strip()
        if not cleaned:
            return None

        now = time.time()
        parent_id = str(uuid.uuid4())
        chunks = chunk_text(cleaned, max_chars=480, overlap=80)

        ids_to_add: List[str] = []
        docs_to_add: List[str] = []
        metas_to_add: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = parent_id if len(chunks) == 1 else f"{parent_id}_c{idx}"
            ids_to_add.append(chunk_id)
            docs_to_add.append(chunk)
            metas_to_add.append({
                "role": role,
                "session_id": session_id,
                "timestamp": now + idx * 1e-4,
                "chunk_index": idx,
                "chunk_total": len(chunks),
                "parent_id": parent_id,
            })

        self.chat_collection.add(
            ids=ids_to_add,
            documents=docs_to_add,
            metadatas=metas_to_add,
        )
        return ids_to_add[0]

    def _hybrid_query(
        self,
        collection,
        query: str,
        n_results: int,
        min_score: float,
        mode: str,
        where: Optional[Dict[str, Any]] = None,
        exclude_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        cleaned_query = (query or "").strip()
        if not cleaned_query or n_results <= 0:
            return []

        total = collection.count()
        if total == 0:
            return []

        candidate_k = min(total, max(n_results * 3, 10))
        query_kwargs: Dict[str, Any] = {
            "query_texts": [cleaned_query],
            "n_results": candidate_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = collection.query(**query_kwargs)
        except Exception:
            # Handle case where `where` matches fewer docs than candidate_k
            if where:
                try:
                    filtered = collection.get(where=where, include=[])
                    matched_count = len(filtered.get("ids", []))
                    if matched_count == 0:
                        return []
                    query_kwargs["n_results"] = min(candidate_k, matched_count)
                    results = collection.query(**query_kwargs)
                except Exception:
                    return []
            else:
                return []

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        space = self._collection_space(collection)
        excluded = set(exclude_ids or [])
        scored: List[Dict[str, Any]] = []
        seen_norms = set()

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            if not doc or doc_id in excluded:
                continue

            meta = meta or {}
            norm_doc = normalize_text(doc)
            if norm_doc in seen_norms:
                continue

            sem_score = self._distance_to_similarity(dist, space)
            lex_score = lexical_similarity(cleaned_query, doc)

            # Probabilistic noisy-OR fusion: preserves strong pure-semantic
            # paraphrases, rescues exact Spanish/entity lexical matches, and
            # boosts documents where both signals agree without an additive floor.
            base_relevance = 1.0 - (1.0 - sem_score) * (1.0 - 0.85 * lex_score)

            if mode == "facts":
                ts = meta.get("updated_at") or meta.get("created_at")
                rec_score = self._recency_score(ts, half_life_hours=720.0)
                imp_score = max(0.1, min(1.0, float(meta.get("importance", 0.7))))
                composite = min(1.0, base_relevance * (0.88 + 0.09 * imp_score + 0.03 * rec_score))
            else:
                ts = meta.get("timestamp")
                rec_score = self._recency_score(ts, half_life_hours=72.0)
                composite = min(1.0, base_relevance * (0.82 + 0.18 * rec_score))

            # Noise gate: filter out records weak on BOTH semantic and lexical dimensions
            if sem_score < 0.20 and lex_score < 0.20:
                continue
            if composite < min_score:
                continue

            seen_norms.add(norm_doc)
            scored.append({
                "id": doc_id,
                "text": doc,
                "score": round(composite, 4),
                "semantic_score": round(sem_score, 4),
                "lexical_score": round(lex_score, 4),
                "distance": round(float(dist), 4) if dist is not None else None,
                "metadata": meta,
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:n_results]

    def recall_user_facts_detailed(
        self,
        query: str,
        n_results: int = 3,
        min_score: float = 0.22,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid semantic + lexical recall of user facts with relevance scores."""
        where = {"category": category} if category else None
        return self._hybrid_query(
            collection=self.user_collection,
            query=query,
            n_results=n_results,
            min_score=min_score,
            mode="facts",
            where=where,
        )

    def recall_user_facts(
        self,
        query: str,
        n_results: int = 3,
        min_score: float = 0.22,
        category: Optional[str] = None,
    ) -> List[str]:
        """Recall relevant user facts as plain text list (backward-compatible API)."""
        matches = self.recall_user_facts_detailed(
            query=query,
            n_results=n_results,
            min_score=min_score,
            category=category,
        )
        return [m["text"] for m in matches]

    def recall_chat_history_detailed(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.24,
        session_id: Optional[str] = None,
        exclude_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid semantic + lexical + recency recall of episodic chat history."""
        where = {"session_id": session_id} if session_id else None
        return self._hybrid_query(
            collection=self.chat_collection,
            query=query,
            n_results=n_results,
            min_score=min_score,
            mode="chat",
            where=where,
            exclude_ids=exclude_ids,
        )

    def recall_chat_history(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.24,
        session_id: Optional[str] = None,
        exclude_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Recall relevant past chat messages as plain text list (backward-compatible API)."""
        matches = self.recall_chat_history_detailed(
            query=query,
            n_results=n_results,
            min_score=min_score,
            session_id=session_id,
            exclude_ids=exclude_ids,
        )
        return [m["text"] for m in matches]

    def get_core_profile(self, limit: int = 5) -> List[str]:
        """
        Return high-importance onboarding/profile facts so JAM always remembers
        core identity and preferred response style even when not directly mentioned.
        """
        if self.user_collection.count() == 0:
            return []

        all_data = self.user_collection.get(include=["documents", "metadatas"])
        docs = all_data.get("documents") or []
        metas = all_data.get("metadatas") or []

        candidates = []
        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            meta = meta or {}
            imp = float(meta.get("importance", 0.7))
            cat = meta.get("category", "fact")
            src = meta.get("source", "")
            if src == "onboarding" or cat in ("profile", "preference") or imp >= 0.85:
                candidates.append((imp, float(meta.get("updated_at", 0.0)), doc))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in candidates[:limit]]

    def get_recent_chat(
        self,
        n_results: int = 6,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the latest N chat messages in chronological order."""
        if self.chat_collection.count() == 0 or n_results <= 0:
            return []

        get_kwargs: Dict[str, Any] = {"include": ["documents", "metadatas"]}
        if session_id:
            get_kwargs["where"] = {"session_id": session_id}

        try:
            data = self.chat_collection.get(**get_kwargs)
        except Exception:
            return []

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        items = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            if not doc:
                continue
            meta = meta or {}
            items.append({
                "id": doc_id,
                "role": meta.get("role", "user"),
                "text": doc,
                "timestamp": float(meta.get("timestamp", 0.0)),
                "session_id": meta.get("session_id", "default"),
            })

        items.sort(key=lambda x: x["timestamp"])
        return items[-n_results:]

    def build_memory_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_facts: int = 5,
        max_recent_turns: int = 4,
        max_recalled_chats: int = 3,
    ) -> Dict[str, Any]:
        """
        Assemble clean, deduplicated multi-tier context for the LLM:
        1. Core user profile facts
        2. Query-relevant recalled user facts (hybrid scored)
        3. Recent sliding window chat turns (short-term memory)
        4. Relevant older episodic chat snippets (long-term memory)
        """
        recalled_facts = self.recall_user_facts_detailed(query, n_results=max_facts)
        core_profile = self.get_core_profile(limit=3)

        merged_facts: List[str] = []
        seen = set()
        for item in recalled_facts:
            norm = normalize_text(item["text"])
            if norm not in seen:
                seen.add(norm)
                merged_facts.append(item["text"])
        for fact in core_profile:
            if len(merged_facts) >= max_facts:
                break
            norm = normalize_text(fact)
            if norm not in seen:
                seen.add(norm)
                merged_facts.append(fact)

        recent_chat = self.get_recent_chat(n_results=max_recent_turns, session_id=session_id)
        recent_ids = [item["id"] for item in recent_chat]

        recalled_chat = self.recall_chat_history_detailed(
            query=query,
            n_results=max_recalled_chats,
            exclude_ids=recent_ids,
        )

        return {
            "facts": merged_facts,
            "recalled_facts_detailed": recalled_facts,
            "recent_chat": recent_chat,
            "recalled_chat": recalled_chat,
        }

    def get_all_user_facts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List stored user persona facts ordered by importance and recency."""
        if self.user_collection.count() == 0:
            return []

        data = self.user_collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        facts = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            if not doc:
                continue
            meta = meta or {}
            facts.append({
                "id": doc_id,
                "text": doc,
                "category": meta.get("category", "fact"),
                "importance": float(meta.get("importance", 0.7)),
                "source": meta.get("source", "unknown"),
                "updated_at": float(meta.get("updated_at", meta.get("created_at", 0.0))),
            })

        facts.sort(key=lambda x: (x["importance"], x["updated_at"]), reverse=True)
        return facts[:limit]

    def delete_user_fact(self, fact_id: str) -> bool:
        """Delete a specific user fact by ID."""
        if not fact_id:
            return False
        try:
            existing = self.user_collection.get(ids=[fact_id], include=[])
            if not existing or not existing.get("ids"):
                return False
            self.user_collection.delete(ids=[fact_id])
            return True
        except Exception:
            return False

    def forget_matching_facts(self, query: str, min_score: float = 0.35, limit: int = 3) -> int:
        """Delete user facts semantically/lexically matching the query."""
        matches = self.recall_user_facts_detailed(query, n_results=limit, min_score=min_score)
        if not matches:
            return 0
        ids = [m["id"] for m in matches]
        self.user_collection.delete(ids=ids)
        return len(ids)

    def clear_chat_history(self, session_id: Optional[str] = None) -> int:
        """Clear chat history (all sessions or a specific session)."""
        if self.chat_collection.count() == 0:
            return 0
        get_kwargs: Dict[str, Any] = {"include": []}
        if session_id:
            get_kwargs["where"] = {"session_id": session_id}
        data = self.chat_collection.get(**get_kwargs)
        ids = data.get("ids") or []
        if not ids:
            return 0
        self.chat_collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> Dict[str, Any]:
        """Return memory database counters and backend details."""
        return {
            "user_facts": self.user_collection.count(),
            "chat_messages": self.chat_collection.count(),
            "db_path": self.db_path,
            "embedding_backend": self.embedding_backend,
            "user_space": self._collection_space(self.user_collection),
            "chat_space": self._collection_space(self.chat_collection),
        }

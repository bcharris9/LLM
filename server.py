"""FastAPI app that serves circuit debugging and lab-manual chat endpoints."""

from __future__ import annotations

import inspect
import json
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
import requests
from supabase import Client, create_client
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
SUPABASE_URL = "https://mvyumvpmzcrrcwcppcea.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im12eXVtdnBtemNycmN3Y3BwY2VhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2Njk2MDQsImV4cCI6MjA3NzI0NTYwNH0.WfjqQowIt9lxKPdnWSGEOP_u7MKmetWgIPFOASuzeBw"

API_DIR = Path(__file__).resolve().parent
ASSETS_DIR = API_DIR / "assets"
HYBRID_ASSETS_DIR = API_DIR / "assets_hybrid"


def _default_chat_model_name() -> str:
    """Return the default local chat model name used for Jetson deployments."""
    return "espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16"

supabase: Optional[Client] = None
embedder: Optional[OllamaEmbeddings] = None
chat_llm: Any | None = None
chat_tokenizer: Any | None = None
chat_device: str | None = None
chat_model_lock = Lock()
conversation_history: dict[str, list[dict[str, str]]] = {}

CONTEXT_MATCH_THRESHOLD = float(os.getenv("LAB_MATCH_THRESHOLD", "0.45"))
SECOND_PASS_THRESHOLD = float(os.getenv("LAB_SECOND_PASS_THRESHOLD", "0.32"))
CONTEXT_MATCH_COUNT = int(os.getenv("LAB_MATCH_COUNT", "40"))
CONTEXT_FINAL_K = int(os.getenv("LAB_FINAL_K", "8"))
CONTEXT_SECTION_LIMIT = int(os.getenv("LAB_SECTION_LIMIT", "3"))
CONTEXT_SCORE_TOLERANCE = float(os.getenv("LAB_SCORE_TOLERANCE", "0.12"))
CONTEXT_MIN_SCORE = float(os.getenv("LAB_MIN_CONTEXT_SCORE", "0.18"))
CONTEXT_MAX_CHARS = int(os.getenv("LAB_CONTEXT_MAX_CHARS", "1800"))
CONTEXT_ANCHOR_COUNT = int(os.getenv("LAB_ANCHOR_COUNT", "3"))
CONTEXT_NEIGHBOR_WINDOW = int(os.getenv("LAB_NEIGHBOR_WINDOW", "1"))
CONTEXT_NEIGHBOR_BONUS = float(os.getenv("LAB_NEIGHBOR_BONUS", "0.06"))
MANUAL_VERSION = os.getenv("LAB_MANUAL_VERSION", "v2")
EMBED_MODEL = os.getenv("LAB_EMBED_MODEL", "mxbai-embed-large")
CHAT_LLM_MODEL = os.getenv("LAB_CHAT_MODEL", _default_chat_model_name())
CHAT_BACKEND = os.getenv("LAB_CHAT_BACKEND", "openai_compat").strip().lower()
CHAT_BASE_URL = os.getenv("LAB_CHAT_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
CHAT_API_KEY = os.getenv("LAB_CHAT_API_KEY", "").strip()
CHAT_MAX_NEW_TOKENS = int(os.getenv("LAB_CHAT_MAX_NEW_TOKENS", "256"))
CHAT_TEMPERATURE = float(os.getenv("LAB_CHAT_TEMPERATURE", "0.0"))
CHAT_TIMEOUT_SECONDS = int(os.getenv("LAB_CHAT_TIMEOUT_SECONDS", "180"))
BM25_K1 = float(os.getenv("LAB_BM25_K1", "1.2"))
BM25_B = float(os.getenv("LAB_BM25_B", "0.75"))
MAX_CHAT_LAB_NUMBER = int(os.getenv("LAB_MAX_NUMBER", "9"))

try:
    if SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        embedder = OllamaEmbeddings(model=EMBED_MODEL)
    else:
        print("WARNING: SUPABASE_KEY missing; /chat retrieval will return 503.")
except Exception as e:  # pragma: no cover - startup diagnostics only
    print(f"Startup Error: {e}")


class DebugRequest(BaseModel):
    """Request body for the circuit debug endpoint."""

    circuit_name: str = Field(..., description="Exact circuit name from GET /circuits")
    node_voltages: dict[str, float] = Field(
        default_factory=dict,
        description="Map of node name -> measured voltage (V). Example keys: N001, N002, VCC, -VCC, VOUT.",
    )
    source_currents: dict[str, float] = Field(
        default_factory=dict,
        description="Optional map of voltage source name -> measured current (A). Example key: V1.",
    )
    measurement_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Advanced: direct measurement_key -> value overrides (e.g. v_n001_max).",
    )
    temp: float | None = Field(default=27.0, description="Temperature feature (degC).")
    tnom: float | None = Field(default=27.0, description="Nominal temperature feature (degC).")
    strict: bool = Field(default=False, description="Fail if not all listed nodes are provided.")


class HealthResponse(BaseModel):
    """Response model for the lightweight service health endpoint."""

    ok: bool
    backend: str
    circuits: int
    family_pair_models: int
    pair_threshold: float
    chat_backend: str
    chat_model: str
    manual_version: str


@lru_cache(maxsize=1)
def get_runtime() -> Any:
    """Lazily construct the configured debug runtime implementation."""
    try:
        from hybrid_runtime import CircuitDebugHybridRuntime  # type: ignore[import-not-found]
        from runtime import CircuitDebugRuntime  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Circuit debug runtime modules are not available in this workspace."
        ) from e

    hybrid_dir = API_DIR / "assets_hybrid"
    hybrid_cfg = hybrid_dir / "hybrid_config.json"
    if hybrid_cfg.exists():
        return CircuitDebugHybridRuntime(
            catalog_path=ASSETS_DIR / "circuit_catalog.json",
            hybrid_assets_dir=hybrid_dir,
            auto_build_catalog_from=Path("pipeline/out_one_lab_all_v2_train"),
        )
    return CircuitDebugRuntime(
        model_bundle_path=ASSETS_DIR / "model_bundle.joblib",
        circuit_catalog_path=ASSETS_DIR / "circuit_catalog.json",
        family_pair_models_path=ASSETS_DIR / "family_pair_models.joblib",
        config_path=ASSETS_DIR / "runtime_config.json",
    )


app = FastAPI(
    title="Circuit Debug API",
    version="1.0.0",
    description=(
        "FastAPI wrapper for the LTSpice-trained circuit fault classifier. "
        "Provides circuit catalog, node schema, and debugging inference from measured node voltages/currents."
    ),
)


class ChatRequest(BaseModel):
    """Request body for both chat routes."""

    question: str
    lab_number: int | None = None


def _require_supabase() -> None:
    """Fail fast when retrieval dependencies are not available."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized; set SUPABASE_KEY.")


def _choose_chat_device() -> tuple[str, torch.dtype]:
    """Choose the device and dtype for the in-process Transformers fallback."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


def _ensure_chat_llm() -> tuple[Any, Any, str]:
    """Lazy-load the in-process chat model for the Transformers backend."""
    global chat_llm, chat_tokenizer, chat_device

    if chat_llm is not None and chat_tokenizer is not None and chat_device is not None:
        return chat_llm, chat_tokenizer, chat_device

    with chat_model_lock:
        if chat_llm is not None and chat_tokenizer is not None and chat_device is not None:
            return chat_llm, chat_tokenizer, chat_device

        device, dtype = _choose_chat_device()
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            model_kwargs["dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype

        tokenizer = AutoTokenizer.from_pretrained(CHAT_LLM_MODEL, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        model = AutoModelForCausalLM.from_pretrained(CHAT_LLM_MODEL, **model_kwargs)
        model.eval()
        model.to(device)

        chat_llm = model
        chat_tokenizer = tokenizer
        chat_device = device
        return chat_llm, chat_tokenizer, chat_device


def _require_llm() -> None:
    """Validate that whichever chat backend is selected is ready to serve requests."""
    if CHAT_BACKEND == "openai_compat":
        if not CHAT_BASE_URL:
            raise HTTPException(
                status_code=503,
                detail="LAB_CHAT_BASE_URL is not configured for the openai_compat chat backend.",
            )
        return
    try:
        _ensure_chat_llm()
    except HTTPException:
        raise
    except Exception as e:
        _raise_http_chat_model_error(e, CHAT_LLM_MODEL, "chat")


def _raise_http_embedding_model_error(error: Exception, model_name: str, role: str) -> None:
    """Translate local embedding-model failures into helpful HTTP errors."""
    message = str(error)
    if "not found" in message.lower() and model_name in message:
        raise HTTPException(
            status_code=503,
            detail=f'Ollama {role} model "{model_name}" is not installed. Run: ollama pull {model_name}',
        ) from error
    raise error


def _raise_http_chat_model_error(error: Exception, model_name: str, role: str) -> None:
    """Translate chat-model failures into a 503 response for the API caller."""
    raise HTTPException(
        status_code=503,
        detail=f'Chat backend "{CHAT_BACKEND}" could not load or invoke model "{model_name}": {error}',
    ) from error


def _invoke_transformers_chat_llm(prompt: str) -> str:
    """Generate a response with the in-process Hugging Face fallback backend."""
    model, tokenizer, device = _ensure_chat_llm()
    prompt = prompt.strip()
    if hasattr(tokenizer, "apply_chat_template"):
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered_prompt = prompt

    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CHAT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][prompt_tokens:]
    if generated_ids.numel() == 0:
        return ""
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _invoke_openai_compatible_chat(prompt: str) -> str:
    """Call a local OpenAI-compatible chat server such as vLLM on Jetson."""
    if not CHAT_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="LAB_CHAT_BASE_URL is not configured for the openai_compat chat backend.",
        )

    headers = {"Content-Type": "application/json"}
    if CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CHAT_API_KEY}"

    payload = {
        "model": CHAT_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": CHAT_TEMPERATURE,
        "max_tokens": CHAT_MAX_NEW_TOKENS,
        "stream": False,
    }
    try:
        response = requests.post(
            f"{CHAT_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=CHAT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f'Could not reach local chat model server at "{CHAT_BASE_URL}": {e}',
        ) from e

    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise HTTPException(
            status_code=503,
            detail=f'Local chat model server error ({response.status_code}): {detail}',
        )

    try:
        body = response.json()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail="Local chat model server returned invalid JSON.",
        ) from e

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(
            status_code=503,
            detail=f"Local chat model server returned an unexpected response: {body}",
        )

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise HTTPException(
            status_code=503,
            detail=f"Local chat model server returned an unexpected response: {body}",
        )

    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(text_parts).strip()
    return str(content).strip()


def _invoke_chat_llm(prompt: str) -> str:
    """Dispatch chat generation to the selected backend."""
    if CHAT_BACKEND == "openai_compat":
        return _invoke_openai_compatible_chat(prompt)
    return _invoke_transformers_chat_llm(prompt)


STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "how",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "would",
    "lab",
    "manual",
}
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]{1,}|\d+[a-z]?", re.IGNORECASE)
LAB_NUMBER_PATTERN = re.compile(r"\blab\s*[-_#:]*\s*(\d{1,2})\b", re.IGNORECASE)
REFERENCE_PATTERNS = [
    re.compile(r"\btask\s*#?\s*\d+[a-z]?\b", re.IGNORECASE),
    re.compile(r"\bfigure\.?\s*\d+(?:\s*\.\s*\d+)*(?:[a-z](?![a-z]))?(?=[^0-9]|$)", re.IGNORECASE),
    re.compile(r"\btable\.?\s*\d+(?:\s*\.\s*\d+)*(?:[a-z](?![a-z]))?(?=[^0-9]|$)", re.IGNORECASE),
    re.compile(r"\bappendix\s*[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\beq(?:uation)?\.?\s*\(?\d+(?:\s*\.\s*\d+)*\)?(?:[a-z](?![a-z]))?(?=[^0-9]|$)", re.IGNORECASE),
]
SECTION_QUERY_TERMS = {
    "goal",
    "objective",
    "theory",
    "introduction",
    "background",
    "pre-lab",
    "prelab",
    "procedure",
    "task",
    "table",
    "result",
    "analysis",
    "discussion",
    "question",
    "report",
    "deliverable",
    "checkoff",
    "conclusion",
    "appendix",
    "equipment",
    "materials",
}

SECTION_NORMALIZATIONS = [
    (r"\bgoals?\b|\bobjectives?\b|\baim\b|\bpurpose\b", "Goals"),
    (r"\bpre[- ]?lab\b", "Pre-Lab"),
    (r"\btheory\b.*\bintroduction\b|\bintroduction\b.*\btheory\b", "Theory and Introduction"),
    (r"\btheory\b", "Theory"),
    (r"\bintroduction\b", "Introduction"),
    (r"\bbackground\b", "Background"),
    (r"\bparts used\b|\bmaterials?\b|\bequipment\b|\bparts needed\b|\bparts list\b", "Materials / Parts"),
    (r"\bprocedure\b", "Procedure"),
    (r"\btables?\b.*\bresults?\b|\bdata tables?\b|\bresults?\b", "Results"),
    (r"\banalysis\b", "Analysis"),
    (r"\bdiscussion\b", "Discussion"),
    (r"\bquestions?\b", "Questions"),
    (r"\breport\b", "Report"),
    (r"\bdeliverables?\b", "Deliverables"),
    (r"\bcheck[- ]?off\b", "Checkoff"),
    (r"\bconclusion\b", "Conclusion"),
]

INTENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "objective": {
        "triggers": ("objective", "goal", "goals", "purpose", "aim"),
        "aliases": ("goals", "goal", "objective", "objectives", "theory and introduction", "introduction"),
        "prefer_early_pages": True,
    },
    "materials": {
        "triggers": ("materials", "material", "equipment", "parts", "components", "supplies"),
        "aliases": ("materials / parts", "materials", "equipment", "parts used", "parts list", "parts"),
        "prefer_early_pages": True,
    },
    "prelab": {
        "triggers": ("pre-lab", "prelab", "pre lab"),
        "aliases": ("pre-lab", "prelab"),
        "prefer_early_pages": True,
    },
    "theory": {
        "triggers": ("theory", "background", "introduction"),
        "aliases": ("theory and introduction", "theory", "background", "introduction"),
        "prefer_early_pages": True,
    },
    "procedure": {
        "triggers": ("procedure", "steps", "instructions"),
        "aliases": ("procedure",),
        "prefer_early_pages": False,
    },
    "results": {
        "triggers": ("results", "result", "data table", "data tables"),
        "aliases": ("results", "table"),
        "prefer_early_pages": False,
    },
    "analysis": {
        "triggers": ("analysis", "analyze", "calculation", "calculations"),
        "aliases": ("analysis", "results"),
        "prefer_early_pages": False,
    },
    "questions": {
        "triggers": ("questions", "question"),
        "aliases": ("questions",),
        "prefer_early_pages": False,
    },
    "report": {
        "triggers": ("report",),
        "aliases": ("report",),
        "prefer_early_pages": False,
    },
    "discussion": {
        "triggers": ("discussion", "discuss", "comment", "comments"),
        "aliases": ("discussion",),
        "prefer_early_pages": False,
    },
    "deliverable": {
        "triggers": ("deliverable", "deliverables"),
        "aliases": ("deliverables",),
        "prefer_early_pages": False,
    },
    "checkoff": {
        "triggers": ("checkoff", "check-off", "check off"),
        "aliases": ("checkoff",),
        "prefer_early_pages": False,
    },
    "conclusion": {
        "triggers": ("conclusion",),
        "aliases": ("conclusion",),
        "prefer_early_pages": False,
    },
}


@dataclass(frozen=True)
class QueryProfile:
    """Parsed query features used to steer manual retrieval and reranking."""

    intent_names: tuple[str, ...]
    section_aliases: tuple[str, ...]
    reference_terms: tuple[str, ...]
    task_numbers: tuple[str, ...]
    broad_section_query: bool


def _format_lab_name(lab_number: int | str) -> str:
    """Normalize a numeric lab identifier into the stored manual name format."""
    return f"Lab {int(str(lab_number))}"


def _extract_lab_number(text: str) -> int | None:
    """Extract an explicit lab number mention from freeform user text."""
    match = LAB_NUMBER_PATTERN.search(text)
    if not match:
        return None
    try:
        lab_number = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return lab_number if lab_number > 0 else None


def _tokenize_list(text: str) -> list[str]:
    """Tokenize text for keyword scoring while dropping lightweight stopwords."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0).lower()
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _tokenize(text: str) -> set[str]:
    """Return a unique token set for overlap calculations."""
    return set(_tokenize_list(text))


def _extract_reference_terms(text: str) -> set[str]:
    """Extract explicit manual references such as tasks, figures, and tables."""
    normalized = re.sub(r"\s+", " ", text.lower())
    refs: set[str] = set()
    for pattern in REFERENCE_PATTERNS:
        for match in pattern.finditer(normalized):
            refs.add(re.sub(r"\s+", " ", match.group(0)).strip())
    return refs


def _extract_section_terms(text: str) -> set[str]:
    """Extract broad section keywords that hint at the user's intent."""
    normalized = text.lower()
    return {term for term in SECTION_QUERY_TERMS if term in normalized}


def _normalize_section_label(label: str | None) -> str:
    """Collapse section and heading labels into stable retrieval-friendly names."""
    if not label:
        return "General"
    cleaned = re.sub(r"\s+", " ", label).strip(" :-")
    normalized = cleaned.lower()

    task_match = re.search(r"task\s*#?\s*(\d+)[\s:-]*([^\n\.]+)?", cleaned, re.IGNORECASE)
    if task_match:
        title = (task_match.group(2) or "").strip(" :-#")
        return f"Task {task_match.group(1)}" + (f": {title}" if title else "")

    for pattern, canonical in SECTION_NORMALIZATIONS:
        if re.search(pattern, normalized):
            return canonical

    if normalized.startswith(("figure ", "table ", "appendix ")):
        return cleaned

    return cleaned or "General"


def _is_reference_heavy_section(section_label: str, heading_label: str) -> bool:
    """Identify sections that mainly serve as figure/table reference material."""
    section_lower = section_label.lower()
    heading_lower = heading_label.lower()
    return section_lower.startswith(("figure ", "table ")) or heading_lower.startswith(("figure ", "table "))


def _build_query_profile(query: str) -> QueryProfile:
    """Parse one user query into intent, section, task, and reference signals."""
    normalized = query.lower()
    intent_names: list[str] = []
    section_aliases: set[str] = set(_extract_section_terms(query))

    for intent_name, config in INTENT_DEFINITIONS.items():
        if any(trigger in normalized for trigger in config["triggers"]):
            intent_names.append(intent_name)
            section_aliases.update(config["aliases"])

    task_numbers = sorted(set(re.findall(r"\btask\s*#?\s*(\d+)\b", normalized)))
    reference_terms = sorted(_extract_reference_terms(query))
    broad_section_query = bool(intent_names) and not task_numbers and not reference_terms

    return QueryProfile(
        intent_names=tuple(intent_names),
        section_aliases=tuple(sorted(section_aliases)),
        reference_terms=tuple(reference_terms),
        task_numbers=tuple(task_numbers),
        broad_section_query=broad_section_query,
    )


def _build_row_search_text(row: dict[str, Any]) -> str:
    """Assemble the text fields used for lexical and reference matching."""
    normalized_section = _normalize_section_label(str(row.get("section_name") or ""))
    normalized_heading = _normalize_section_label(str(row.get("heading") or ""))
    parts = [
        normalized_section,
        normalized_heading,
        str(row.get("section_name") or ""),
        str(row.get("heading") or ""),
        str(row.get("content") or ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def _row_identity(row: dict[str, Any]) -> str:
    """Return a stable identifier for a manual chunk row."""
    if row.get("id"):
        return str(row["id"])
    content = str(row.get("content") or "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _bm25_score(
    query_tokens: list[str],
    doc_token_counts: Counter[str],
    doc_length: int,
    avg_dl: float,
    df_counts: Counter[str],
    n_docs: int,
) -> float:
    """Compute a BM25-style lexical score for one manual chunk."""
    if not query_tokens or not n_docs:
        return 0.0

    score = 0.0
    safe_avg_dl = avg_dl or 1.0
    for term in query_tokens:
        frequency = doc_token_counts.get(term, 0)
        if frequency == 0:
            continue
        df = df_counts.get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        denom = frequency + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / safe_avg_dl)
        score += idf * (frequency * (BM25_K1 + 1) / denom)
    return score


def _load_lab_rows(lab_name: str) -> list[dict[str, Any]]:
    """Load all manual chunks for one lab, preferring the configured manual version."""
    if not supabase:
        return []

    def _select_rows(manual_version: Optional[str]) -> list[dict[str, Any]]:
        """Fetch manual chunks for one lab and optional manual-version filter."""
        query = (
            supabase.table("lab_sections")
            .select("id, lab_name, manual_version, section_name, heading, content, page_num, chunk_order, token_count")
            .filter("lab_name", "ilike", f"%{lab_name}%")
            .order("chunk_order")
        )
        if manual_version:
            query = query.eq("manual_version", manual_version)
        response = query.execute()
        return response.data or []

    rows = _select_rows(MANUAL_VERSION)
    if not rows and MANUAL_VERSION:
        rows = _select_rows(None)
    return [dict(row) for row in rows]


def _vector_search_scores(query: str, lab_name: str, profile: QueryProfile) -> dict[str, float]:
    """Fetch semantic-search scores for one lab's manual chunks."""
    if not embedder or not supabase:
        return {}

    search_parts = [f"Question about the {lab_name} manual.", f"Question: {query}"]
    if profile.section_aliases:
        search_parts.append("Relevant section types: " + ", ".join(profile.section_aliases))
    if profile.task_numbers:
        search_parts.append("Referenced tasks: " + ", ".join(profile.task_numbers))
    search_query = "\n".join(search_parts)
    try:
        vector = embedder.embed_query(search_query)
    except Exception as e:
        _raise_http_embedding_model_error(e, EMBED_MODEL, "embedding")

    def _call_match_rpc(manual_version: Optional[str], threshold: float):
        """Call the vector-match RPC with the current embedding and filters."""
        payload = {
            "query_embedding": vector,
            "match_threshold": threshold,
            "match_count": CONTEXT_MATCH_COUNT,
            "filter_lab_name": lab_name,
            "filter_manual_version": manual_version,
        }
        return supabase.rpc("match_lab_manuals", payload).execute()

    scores: dict[str, float] = {}
    manual_versions = [MANUAL_VERSION] if MANUAL_VERSION else [None]
    if MANUAL_VERSION:
        manual_versions.append(None)

    thresholds = [CONTEXT_MATCH_THRESHOLD]
    if SECOND_PASS_THRESHOLD < CONTEXT_MATCH_THRESHOLD:
        thresholds.append(SECOND_PASS_THRESHOLD)

    for manual_version in manual_versions:
        for threshold in thresholds:
            response = _call_match_rpc(manual_version, threshold)
            for row in response.data or []:
                identity = _row_identity(row)
                score = float(row.get("similarity", row.get("score", 0.0)) or 0.0)
                if score > scores.get(identity, 0.0):
                    scores[identity] = score
            if len(scores) >= max(CONTEXT_FINAL_K * 2, 6):
                return scores

    return scores


def _score_lab_rows(
    query: str,
    lab_rows: list[dict[str, Any]],
    vector_scores: dict[str, float],
    profile: QueryProfile,
) -> list[dict[str, Any]]:
    """Fuse semantic, lexical, and structural heuristics into a ranked chunk list."""
    query_tokens = _tokenize_list(query)
    query_token_set = set(query_tokens)
    query_references = set(profile.reference_terms)
    query_sections = set(profile.section_aliases)

    prepared_rows: list[dict[str, Any]] = []
    df_counts: Counter[str] = Counter()
    total_doc_length = 0
    section_first_orders: dict[str, int] = {}

    for row in lab_rows:
        row_copy = dict(row)
        section_display = _normalize_section_label(str(row_copy.get("section_name") or ""))
        heading_raw = str(row_copy.get("heading") or "").strip()
        heading_display = heading_raw or section_display
        if _normalize_section_label(heading_display) == section_display:
            heading_display = section_display

        search_text = _build_row_search_text(row_copy)
        doc_tokens = _tokenize_list(search_text)
        token_counts = Counter(doc_tokens)
        doc_length = len(doc_tokens) or 1
        section_key = section_display.lower()
        chunk_order = row_copy.get("chunk_order")
        if isinstance(chunk_order, int):
            existing = section_first_orders.get(section_key)
            if existing is None or chunk_order < existing:
                section_first_orders[section_key] = chunk_order

        prepared_rows.append(
            {
                "row": row_copy,
                "search_text": search_text,
                "token_counts": token_counts,
                "doc_length": doc_length,
                "section_display": section_display,
                "heading_display": heading_display,
                "section_key": section_key,
            }
        )
        total_doc_length += doc_length
        df_counts.update(token_counts.keys())

    if not prepared_rows:
        return []

    avg_dl = total_doc_length / len(prepared_rows)
    query_terms = sorted(query_token_set)
    max_bm25 = 0.0
    exact_reference_available = bool(query_references) and any(
        any(reference in item["search_text"].lower() for reference in query_references)
        for item in prepared_rows
    )

    for item in prepared_rows:
        bm25 = _bm25_score(
            query_tokens=query_terms,
            doc_token_counts=item["token_counts"],
            doc_length=item["doc_length"],
            avg_dl=avg_dl,
            df_counts=df_counts,
            n_docs=len(prepared_rows),
        )
        item["bm25"] = bm25
        if bm25 > max_bm25:
            max_bm25 = bm25

    ranked_rows: list[dict[str, Any]] = []
    for item in prepared_rows:
        row = item["row"]
        search_text_lower = item["search_text"].lower()
        section_display = item["section_display"]
        heading_display = item["heading_display"]
        section_lower = section_display.lower()
        heading_lower = heading_display.lower()
        doc_terms = set(item["token_counts"].keys())
        chunk_order = row.get("chunk_order")
        page_num = row.get("page_num")
        section_key = item["section_key"]
        is_section_start = isinstance(chunk_order, int) and section_first_orders.get(section_key) == chunk_order
        is_reference_section = _is_reference_heavy_section(section_display, heading_display)
        has_exact_reference = any(reference in search_text_lower for reference in query_references)

        overlap = len(query_token_set & doc_terms)
        coverage = overlap / max(len(query_token_set), 1)
        bm25_score = item["bm25"] / max_bm25 if max_bm25 else 0.0
        vector_score = vector_scores.get(_row_identity(row), 0.0)
        intent_bonus = 0.0
        mismatch_penalty = 0.0

        reference_bonus = 0.0
        # Exact figure/task/table mentions deserve a large bump because they usually indicate
        # the user is asking for one specific part of the manual rather than a broad topic.
        for reference in query_references:
            if reference in heading_lower or reference in section_lower:
                reference_bonus += 0.2
            else:
                ref_pos = search_text_lower.find(reference)
                if 0 <= ref_pos < 120:
                    reference_bonus += 0.12
                elif 0 <= ref_pos < 320:
                    reference_bonus += 0.07
                elif ref_pos >= 0:
                    reference_bonus += 0.03
        if exact_reference_available:
            if has_exact_reference:
                reference_bonus += 0.1
            elif query_references:
                mismatch_penalty += 0.18
        reference_bonus = min(reference_bonus, 0.28)

        section_bonus = 0.0
        # Section aliases let broad questions such as "what are the objectives" land on the
        # start of the right section even when the body text uses slightly different wording.
        for term in query_sections:
            if term in heading_lower or term in section_lower:
                section_bonus += 0.11
            elif term in search_text_lower[:500]:
                section_bonus += 0.03
        section_bonus = min(section_bonus, 0.34)

        heading_bonus = 0.0
        if heading_lower:
            heading_bonus += 0.02 * len(query_token_set & _tokenize(heading_lower))
        if section_lower:
            heading_bonus += 0.015 * len(query_token_set & _tokenize(section_lower))
        heading_bonus = min(heading_bonus, 0.12)

        for intent_name in profile.intent_names:
            config = INTENT_DEFINITIONS[intent_name]
            aliases = config["aliases"]
            matched_intent = any(alias in section_lower or alias in heading_lower for alias in aliases)
            if matched_intent:
                intent_bonus += 0.12
                if is_section_start:
                    intent_bonus += 0.08
                if config.get("prefer_early_pages") and isinstance(page_num, int):
                    intent_bonus += max(0.0, 0.06 - 0.012 * max(page_num - 1, 0))
            elif any(alias in search_text_lower[:300] for alias in aliases):
                intent_bonus += 0.04

        if profile.broad_section_query and is_section_start:
            intent_bonus += 0.06

        if profile.broad_section_query and not query_references:
            if is_reference_section and not section_bonus:
                mismatch_penalty += 0.14
            if section_lower.startswith("task ") and "procedure" not in profile.intent_names:
                mismatch_penalty += 0.08
            if section_lower == "results" and any(
                intent in profile.intent_names for intent in ("objective", "materials", "prelab")
            ):
                mismatch_penalty += 0.06

        task_bonus = 0.0
        for task_number in profile.task_numbers:
            if f"task {task_number}" in section_lower or f"task {task_number}" in heading_lower:
                task_bonus += 0.18

        token_count = row.get("token_count")
        if not isinstance(token_count, int):
            token_count = len(doc_terms)
        information_bonus = 0.0
        if token_count >= 35:
            information_bonus += 0.05
        elif token_count >= 18:
            information_bonus += 0.02
        elif profile.broad_section_query:
            mismatch_penalty += 0.08
            if is_section_start:
                mismatch_penalty += 0.04

        row["_display_section_name"] = section_display
        row["_display_heading"] = heading_display
        row["_section_key"] = section_key
        row["_section_start"] = is_section_start
        row["_intent_match"] = section_bonus > 0 or intent_bonus > 0
        row["_combined_score"] = (
            0.52 * vector_score
            + 0.22 * bm25_score
            + 0.14 * coverage
            + reference_bonus
            + section_bonus
            + heading_bonus
            + intent_bonus
            + task_bonus
            + information_bonus
            - mismatch_penalty
        )
        ranked_rows.append(row)

    ranked_rows.sort(
        key=lambda row: (
            row.get("_combined_score", 0.0) + (0.04 if profile.broad_section_query and row.get("_section_start") else 0.0)
        ),
        reverse=True,
    )
    anchors = ranked_rows[:CONTEXT_ANCHOR_COUNT]
    anchor_positions = [
        (anchor.get("chunk_order"), str(anchor.get("section_name") or "").lower())
        for anchor in anchors
        if isinstance(anchor.get("chunk_order"), int)
    ]
    if anchor_positions:
        # Small neighbor bonus helps pull in adjacent chunks when the best answer spans a section.
        for row in ranked_rows:
            row_order = row.get("chunk_order")
            row_section = str(row.get("section_name") or "").lower()
            if not isinstance(row_order, int):
                continue
            if any(
                row_section == anchor_section and 0 < abs(row_order - anchor_order) <= CONTEXT_NEIGHBOR_WINDOW
                for anchor_order, anchor_section in anchor_positions
            ):
                row["_combined_score"] += CONTEXT_NEIGHBOR_BONUS

    ranked_rows.sort(
        key=lambda row: (
            row.get("_combined_score", 0.0)
            + (0.04 if profile.broad_section_query and row.get("_section_start") else 0.0)
            + (0.03 if row.get("_intent_match") else 0.0)
        ),
        reverse=True,
    )
    return ranked_rows


def _select_context_rows(ranked_rows: list[dict[str, Any]], profile: QueryProfile) -> list[dict[str, Any]]:
    """Pick a compact, diverse set of top chunks to feed into the chat model."""
    if not ranked_rows:
        return []
    if ranked_rows[0].get("_combined_score", 0.0) < CONTEXT_MIN_SCORE:
        return []

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    section_counts: defaultdict[str, int] = defaultdict(int)
    best_score = ranked_rows[0].get("_combined_score", 0.0)
    section_starts = {
        str(row.get("_section_key") or row.get("section_name") or ""): row
        for row in ranked_rows
        if row.get("_section_start")
    }
    section_limit = 2 if profile.broad_section_query else CONTEXT_SECTION_LIMIT

    for row in ranked_rows:
        identity = _row_identity(row)
        if identity in seen_ids:
            continue

        score = row.get("_combined_score", 0.0)
        if score <= 0:
            continue
        if (
            selected
            and len(selected) >= min(4, CONTEXT_FINAL_K)
            and score < (best_score - CONTEXT_SCORE_TOLERANCE)
        ):
            break

        section_key = str(row.get("_section_key") or row.get("section_name") or row.get("page_num") or "general")
        if (
            profile.broad_section_query
            and section_counts[section_key] == 0
            and not row.get("_section_start")
        ):
            starter = section_starts.get(section_key)
            if starter and starter.get("_combined_score", 0.0) >= score - 0.08:
                continue

        if section_counts[section_key] >= section_limit:
            continue

        selected.append(row)
        seen_ids.add(identity)
        section_counts[section_key] += 1
        if len(selected) >= CONTEXT_FINAL_K:
            break

    return selected


def _format_context_row(row: dict[str, Any]) -> str:
    """Render one selected manual chunk into the prompt context format."""
    section = str(row.get("_display_section_name") or row.get("section_name") or "Section ?")
    heading = str(row.get("_display_heading") or row.get("heading") or "").strip()
    page_num = row.get("page_num")
    tag_parts = [str(row.get("lab_name") or "Lab ?"), section]
    if heading and heading.lower() != section.lower():
        tag_parts.append(heading)
    if page_num not in (None, ""):
        tag_parts.append(f"p.{page_num}")

    content = str(row.get("content") or "").strip()
    if len(content) > CONTEXT_MAX_CHARS:
        content = content[:CONTEXT_MAX_CHARS].rstrip() + "..."

    return f"[{' | '.join(tag_parts)}]\n{content}"


def _strip_answer_metadata(answer: str) -> str:
    """Remove retrieval tags and excess whitespace from a model answer."""
    cleaned = re.sub(r"\[(?:Lab|Appendix)[^\]]+\]", "", answer)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _reference_context_rows(
    lab_rows: list[dict[str, Any]],
    reference_terms: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Short-circuit retrieval when the query explicitly names figures, tables, or tasks."""
    if not reference_terms:
        return []

    scored_rows: list[dict[str, Any]] = []
    for row in lab_rows:
        search_text = _build_row_search_text(row)
        search_text_lower = search_text.lower()
        heading_lower = str(row.get("heading") or "").lower()
        section_lower = str(row.get("section_name") or "").lower()

        positions = [search_text_lower.find(reference) for reference in reference_terms if reference in search_text_lower]
        if not positions:
            continue

        best_pos = min(positions)
        score = 0.0
        if any(reference in heading_lower or reference in section_lower for reference in reference_terms):
            score += 2.0
        if best_pos < 80:
            score += 1.6
        elif best_pos < 220:
            score += 1.0
        else:
            score += 0.4

        token_count = row.get("token_count")
        if isinstance(token_count, int) and token_count >= 12:
            score += 0.4
        if _is_reference_heavy_section(str(row.get("section_name") or ""), str(row.get("heading") or "")):
            score += 0.3

        row_copy = dict(row)
        row_copy["_display_section_name"] = _normalize_section_label(str(row.get("section_name") or ""))
        row_copy["_display_heading"] = str(row.get("heading") or row_copy["_display_section_name"]).strip()
        row_copy["_reference_score"] = score
        scored_rows.append(row_copy)

    if not scored_rows:
        return []

    scored_rows.sort(key=lambda row: row.get("_reference_score", 0.0), reverse=True)
    best_score = scored_rows[0].get("_reference_score", 0.0)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in scored_rows:
        if row.get("_reference_score", 0.0) < best_score - 0.45:
            break
        identity = _row_identity(row)
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        selected.append(row)
        if len(selected) >= min(4, CONTEXT_FINAL_K):
            break
    return selected


def retrieve_context(query: str, lab_number: int) -> list[str]:
    """Return only the formatted context snippets for one lab."""
    return _retrieve_context_for_lab(query, lab_number)[1]


def _retrieve_context_for_lab(query: str, lab_number: int) -> tuple[float, list[str]]:
    """Retrieve and rank context snippets for one candidate lab."""
    if not embedder or not supabase:
        return 0.0, []

    lab_name = _format_lab_name(lab_number)
    profile = _build_query_profile(query)
    lab_rows = _load_lab_rows(lab_name)
    if not lab_rows:
        return 0.0, []

    reference_rows = _reference_context_rows(lab_rows, profile.reference_terms)
    if reference_rows:
        return (
            float(reference_rows[0].get("_reference_score", 0.0) or 0.0),
            [_format_context_row(row) for row in reference_rows],
        )

    vector_scores = _vector_search_scores(query, lab_name, profile)
    ranked_rows = _score_lab_rows(query, lab_rows, vector_scores, profile)
    selected_rows = _select_context_rows(ranked_rows, profile)
    if not selected_rows:
        return 0.0, []
    return (
        float(selected_rows[0].get("_combined_score", 0.0) or 0.0),
        [_format_context_row(row) for row in selected_rows],
    )


def _resolve_chat_lab(question: str, requested_lab_number: int | None) -> tuple[str | None, list[str]]:
    """Resolve the best lab to answer from, using explicit or inferred lab hints first."""
    inferred_lab_number = _extract_lab_number(question)

    if requested_lab_number is not None:
        return _format_lab_name(requested_lab_number), retrieve_context(question, requested_lab_number)

    if inferred_lab_number is not None:
        return _format_lab_name(inferred_lab_number), retrieve_context(question, inferred_lab_number)

    best_lab_number: int | None = None
    best_context: list[str] = []
    best_score = 0.0

    for candidate_lab_number in range(1, MAX_CHAT_LAB_NUMBER + 1):
        score, context = _retrieve_context_for_lab(question, candidate_lab_number)
        if not context:
            continue
        if best_lab_number is None or score > best_score:
            best_lab_number = candidate_lab_number
            best_context = context
            best_score = score

    if best_lab_number is None:
        return None, []
    return _format_lab_name(best_lab_number), best_context


def _chat_response(request: ChatRequest, path_lab_number: int | None = None) -> dict[str, str]:
    """Handle a chat request by retrieving context, prompting the local chat model, and caching short history."""
    _require_supabase()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if path_lab_number is not None and request.lab_number is not None and path_lab_number != request.lab_number:
        raise HTTPException(status_code=400, detail="Conflicting lab numbers in path and request body.")

    explicit_lab_number = path_lab_number if path_lab_number is not None else request.lab_number
    lab_name, context = _resolve_chat_lab(question, explicit_lab_number)
    if not context:
        return {"answer": "I cannot find that information in the lab manual."}

    _require_llm()
    assert lab_name is not None
    lab_history = conversation_history.setdefault(lab_name, [])
    # Keep only short recent history per lab so follow-up questions have continuity without
    # letting prompt size grow unbounded.
    history_txt = "\n".join(
        [f"Q: {turn.get('user', '')}\nA: {turn.get('ai', '')}" for turn in lab_history[-2:]]
    )
    context_txt = "\n---\n".join(context)

    prompt = f"""
    You are a helpful electrical engineering lab assistant for {lab_name}.
    Answer only using the facts explicitly stated in the context snippets below.
    Be specific about required actions, component values, figures, tables, and deliverables; do not answer with vague labels alone.
    Do not mention snippet tags, page numbers, metadata, or bracketed citations in the final answer.
    If the answer is not in the context, reply exactly: "I cannot find that information in the lab manual." Do NOT guess.

    Context (do not use outside knowledge):
    ---
    {context_txt}
    ---

    Recent conversation for the same lab (for continuity, avoid repeating):
    {history_txt}

    Question: {question}
    Answer:
    """

    try:
        answer = _invoke_chat_llm(prompt)
    except Exception as e:
        _raise_http_chat_model_error(e, CHAT_LLM_MODEL, "chat")
    final_answer = _strip_answer_metadata(str(answer))
    if not final_answer:
        final_answer = "I cannot find that information in the lab manual."
    lab_history.append({"user": question, "ai": final_answer})
    if len(lab_history) > 6:
        del lab_history[:-6]
    return {"answer": final_answer}


@app.post("/chat")
def chat(request: ChatRequest):
    """Generic chat route that infers or auto-selects the relevant lab."""
    return _chat_response(request)


@app.post("/chat/{lab_number}")
def chat_for_lab(lab_number: int, request: ChatRequest):
    """Lab-specific chat route that skips lab inference."""
    return _chat_response(request, path_lab_number=lab_number)



@app.get("/circuits")
def list_circuits() -> dict[str, Any]:
    """List the packaged circuits available to the debug runtime."""
    rt = get_runtime()
    names = rt.list_circuits()
    return {"count": len(names), "circuits": names}


@app.get("/circuits/{circuit_name}/nodes")
def get_circuit_nodes(circuit_name: str) -> dict[str, Any]:
    """Return the expected node and optional source-current inputs for one circuit."""
    rt = get_runtime()
    if not rt.has_circuit(circuit_name):
        raise HTTPException(status_code=404, detail=f"Unknown circuit: {circuit_name}")
    spec = rt.circuit_spec(circuit_name)
    return {
        "circuit_name": circuit_name,
        "node_count": len(spec.get("nodes", [])),
        "nodes": spec.get("nodes", []),
        "source_current_count": len(spec.get("source_currents", [])),
        "source_currents": spec.get("source_currents", []),
        "golden_defaults": spec.get("golden_defaults", {}),
        "notes": {
            "recommended": "Provide all listed nodes in POST /debug for best accuracy. Source currents are optional but improve accuracy."
        },
    }


@app.post("/debug")
def debug_circuit(req: DebugRequest) -> dict[str, Any]:
    """Run circuit fault inference for the provided measurement payload."""
    rt = get_runtime()
    if not rt.has_circuit(req.circuit_name):
        raise HTTPException(status_code=404, detail=f"Unknown circuit: {req.circuit_name}")
    try:
        result = rt.predict_fault(
            circuit_name=req.circuit_name,
            node_voltages=req.node_voltages,
            source_currents=req.source_currents,
            measurement_overrides=req.measurement_overrides,
            temp=req.temp,
            tnom=req.tnom,
            strict=req.strict,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    return result.to_dict()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Expose a lightweight readiness and configuration summary."""
    rt = get_runtime()
    backend = "llm_knn_hybrid" if rt.__class__.__name__.endswith("HybridRuntime") else "tabular_xgboost"
    return HealthResponse(
        ok=True,
        backend=backend,
        circuits=len(rt.list_circuits()),
        family_pair_models=len(getattr(rt, "family_pair_models", {})),
        pair_threshold=float(getattr(rt, "pair_threshold", 0.0)),
        chat_backend=CHAT_BACKEND,
        chat_model=CHAT_LLM_MODEL,
        manual_version=MANUAL_VERSION,
    )

# Root helper for quick manual check
@app.get("/")
def root():
    """Return a minimal human-readable summary of the running API."""
    return {
        "message": "SPICE Lab Assistant API is running",
        "chat_backend": CHAT_BACKEND,
        "chat_model": CHAT_LLM_MODEL,
        "chat_base_url": CHAT_BASE_URL or None,
        "routes": [
            "/chat",
            "/chat/{lab_number}",
            "/circuits",
            "/circuits/{circuit_name}/nodes",
            "/debug",
            "/health",
        ],
    }

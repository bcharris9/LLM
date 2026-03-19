from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except ModuleNotFoundError:
    OllamaLLM = None  # type: ignore[assignment]
    OllamaEmbeddings = None  # type: ignore[assignment]

try:
    from supabase import Client, create_client
except ModuleNotFoundError:
    Client = Any  # type: ignore[misc,assignment]

    def create_client(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("supabase package not installed")

load_dotenv()
SUPABASE_URL = "https://mvyumvpmzcrrcwcppcea.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im12eXVtdnBtemNycmN3Y3BwY2VhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2Njk2MDQsImV4cCI6MjA3NzI0NTYwNH0.WfjqQowIt9lxKPdnWSGEOP_u7MKmetWgIPFOASuzeBw"

supabase: Optional[Client] = None
llm: Optional[OllamaLLM] = None
embedder: Optional[OllamaEmbeddings] = None
conversation_history: list[dict[str, str]] = []

CONTEXT_MATCH_THRESHOLD = float(os.getenv("LAB_MATCH_THRESHOLD", "0.58"))
SECOND_PASS_THRESHOLD = float(os.getenv("LAB_SECOND_PASS_THRESHOLD", "0.48"))
CONTEXT_MATCH_COUNT = int(os.getenv("LAB_MATCH_COUNT", "30"))
CONTEXT_FINAL_K = int(os.getenv("LAB_FINAL_K", "10"))
CONTEXT_SECTION_LIMIT = int(os.getenv("LAB_SECTION_LIMIT", "4"))
CONTEXT_SCORE_TOLERANCE = float(os.getenv("LAB_SCORE_TOLERANCE", "0.08"))
CONTEXT_MAX_CHARS = int(os.getenv("LAB_CONTEXT_MAX_CHARS", "1800"))
MANUAL_VERSION = os.getenv("LAB_MANUAL_VERSION")
BM25_K1 = float(os.getenv("LAB_BM25_K1", "1.2"))
BM25_B = float(os.getenv("LAB_BM25_B", "0.75"))

try:
    if SUPABASE_KEY and OllamaLLM is not None and OllamaEmbeddings is not None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        llm = OllamaLLM(model="gpt-oss:120b-cloud")
        embedder = OllamaEmbeddings(model="mxbai-embed-large")
    else:
        print("WARNING: chat dependencies not fully available; /chat will return 503.")
except Exception as e:  # pragma: no cover - startup diagnostics only
    print(f"Startup Error: {e}")


from hybrid_runtime import CircuitDebugHybridRuntime  # type: ignore[no-redef]
from runtime import CircuitDebugRuntime  # type: ignore[no-redef]

API_DIR = Path(__file__).resolve().parent
ASSETS_DIR = API_DIR / "assets"
HYBRID_DIR = API_DIR / "assets_hybrid"
HYBRID_CONFIG_PATH = HYBRID_DIR / "hybrid_config.json"
PACKAGED_GOLDEN_ROOT = API_DIR / "packaged_golden_root"

class DebugRequest(BaseModel):
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
    strict: bool = Field(default=True, description="Fail if not all listed nodes are provided.")

class HealthResponse(BaseModel):
    ok: bool
    backend: str
    circuits: int
    family_pair_models: int
    pair_threshold: float
    selected_model_report: str | None = None
    selected_model_metric: str | None = None
    selected_model_metric_value: float | None = None


class ModelInfoResponse(BaseModel):
    ok: bool
    backend: str
    adapter_dir: str | None = None
    knn_ref_file: str | None = None
    knn_index_file: str | None = None
    model_name: str
    selected_model_report: str | None = None
    selected_model_metric: str | None = None
    selected_model_metric_value: float | None = None


def _load_hybrid_config() -> dict[str, Any] | None:
    if not HYBRID_CONFIG_PATH.exists():
        return None
    return json.loads(HYBRID_CONFIG_PATH.read_text(encoding="utf-8"))


def _refresh_hybrid_assets() -> None:
    builder = API_DIR / "build_hybrid_assets.py"
    if not builder.exists():
        raise FileNotFoundError(f"Missing builder: {builder}")

    cfg = _load_hybrid_config() or {}
    bool_flags = {
        "include-lab-id-in-prompt": cfg.get("include_lab_id_in_prompt", False),
        "prefer-voltage-keys": cfg.get("prefer_voltage_keys", True),
        "voltage-only": cfg.get("voltage_only", False),
        "knn-weighted-vote": cfg.get("knn_weighted_vote", True),
        "knn-standardize": cfg.get("knn_standardize", False),
        "use-prerules": cfg.get("use_prerules", True),
    }

    cmd = [sys.executable, str(builder)]
    if "measurement_stat_mode" in cfg:
        cmd += ["--measurement-stat-mode", str(cfg["measurement_stat_mode"])]
    if "response_style" in cfg:
        cmd += ["--response-style", str(cfg["response_style"])]
    if "knn_k" in cfg:
        cmd += ["--knn-k", str(int(cfg["knn_k"]))]
    if "knn_alpha" in cfg:
        cmd += ["--knn-alpha", str(float(cfg["knn_alpha"]))]
    if "knn_eps" in cfg:
        cmd += ["--knn-eps", str(float(cfg["knn_eps"]))]
    if "knn_weighted_vote" in cfg:
        cmd += ["--knn-weighted-vote" if bool_flags["knn-weighted-vote"] else "--no-knn-weighted-vote"]
    if "knn_standardize" in cfg:
        cmd += ["--knn-standardize" if bool_flags["knn-standardize"] else "--no-knn-standardize"]
    if "use_prerules" in cfg:
        cmd += ["--use-prerules" if bool_flags["use-prerules"] else "--no-use-prerules"]
    if "include_lab_id_in_prompt" in cfg:
        cmd += [
            "--include-lab-id-in-prompt"
            if bool_flags["include-lab-id-in-prompt"]
            else "--no-include-lab-id-in-prompt"
        ]
    if "prefer_voltage_keys" in cfg:
        cmd += [
            "--prefer-voltage-keys"
            if bool_flags["prefer-voltage-keys"]
            else "--no-prefer-voltage-keys"
        ]
    if "voltage_only" in cfg:
        cmd += ["--voltage-only" if bool_flags["voltage-only"] else "--no-voltage-only"]
    if "max_measurements" in cfg:
        cmd += ["--max-measurements", str(int(cfg["max_measurements"]))]
    if "max_deltas" in cfg:
        cmd += ["--max-deltas", str(int(cfg["max_deltas"]))]

    subprocess.run(cmd, check=True, cwd=str(API_DIR.parent), capture_output=True, text=True)


@lru_cache(maxsize=1)
def get_runtime() -> Any:
    hybrid_dir = HYBRID_DIR
    hybrid_cfg = HYBRID_CONFIG_PATH
    if hybrid_cfg.exists():
        return CircuitDebugHybridRuntime(
            catalog_path=ASSETS_DIR / "circuit_catalog.json",
            hybrid_assets_dir=hybrid_dir,
            auto_build_catalog_from=PACKAGED_GOLDEN_ROOT if PACKAGED_GOLDEN_ROOT.exists() else None,
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
    question: str


def _require_supabase():
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase client not initialized; set SUPABASE_KEY.")


def _require_llm():
    if not llm:
        raise HTTPException(status_code=503, detail="LLM client not initialized.")


STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "for", "with", "in", "on", "at", "by", "from"}


def _normalize_lab_filter(query: str) -> Optional[str]:
    match = re.search(r"lab\s*0?(\d+)", query, re.IGNORECASE)
    return f"Lab {match.group(1)}" if match else None


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", text.lower()) if t not in STOPWORDS}


def _tokenize_list(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", text.lower()) if t not in STOPWORDS]


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    df_counts: dict[str, int],
    n_docs: int,
) -> float:
    score = 0.0
    dl = len(doc_tokens) or 1
    for term in query_tokens:
        f = doc_tokens.count(term)
        if f == 0:
            continue
        df = df_counts.get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
        score += idf * (f * (BM25_K1 + 1) / denom)
    return score


def retrieve_context(query: str) -> list[str]:
    if not embedder or not supabase:
        return []

    lab_filter = _normalize_lab_filter(query)
    vec = embedder.embed_query(query)
    query_tokens = _tokenize(query)

    def _call_match_rpc(
        vector: list[float],
        lab_filter_local: Optional[str],
        manual_version: Optional[str],
        threshold: float,
        count: int,
    ):
        payload = {
            "query_embedding": vector,
            "match_threshold": threshold,
            "match_count": count,
            "filter_lab_name": lab_filter_local,
            "filter_manual_version": manual_version,
        }
        return supabase.rpc("match_lab_manuals", payload).execute()

    rows: list[dict] = []
    if lab_filter:
        res = _call_match_rpc(vec, lab_filter, MANUAL_VERSION, CONTEXT_MATCH_THRESHOLD, CONTEXT_MATCH_COUNT)
        rows = res.data or []
    if len(rows) < 6:
        res = _call_match_rpc(vec, None, MANUAL_VERSION, CONTEXT_MATCH_THRESHOLD, CONTEXT_MATCH_COUNT)
        rows = (rows or []) + (res.data or [])
    if not rows and SECOND_PASS_THRESHOLD < CONTEXT_MATCH_THRESHOLD:
        res = _call_match_rpc(vec, None, MANUAL_VERSION, SECOND_PASS_THRESHOLD, CONTEXT_MATCH_COUNT)
        rows = res.data or []
    if not rows and MANUAL_VERSION:
        res = _call_match_rpc(vec, lab_filter, None, CONTEXT_MATCH_THRESHOLD, CONTEXT_MATCH_COUNT)
        rows = res.data or []
        if not rows and SECOND_PASS_THRESHOLD < CONTEXT_MATCH_THRESHOLD:
            res = _call_match_rpc(vec, lab_filter, None, SECOND_PASS_THRESHOLD, CONTEXT_MATCH_COUNT)
            rows = res.data or []
    if not rows:
        return []

    score_key = "similarity" if "similarity" in rows[0] else ("score" if "score" in rows[0] else None)
    seen_hashes = set()
    reranked = []
    doc_tokens_list = []

    for r in rows:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        sig = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if sig in seen_hashes:
            continue
        seen_hashes.add(sig)
        base_score = float(r.get(score_key, 0.0)) if score_key else 0.0
        overlap = len(query_tokens & _tokenize(content))
        bonus = 0.02 * overlap
        lab_bonus = 0.05 if (lab_filter and r.get("lab_name") and lab_filter.lower() in str(r.get("lab_name", "")).lower()) else 0.0
        section = (r.get("section_name") or "").lower()
        heading = (r.get("heading") or "").lower()
        text_lower = query.lower()
        figure_bonus = 0.0
        for tok in re.findall(r"figure\s*\d+", text_lower):
            if tok in heading:
                figure_bonus = 0.06
                break
        task_bonus = 0.05 if ("task" in text_lower and "task" in section) else 0.0
        page_num = r.get("page_num")
        position_bonus = 0.0
        if isinstance(page_num, int):
            position_bonus = max(0.0, 0.03 - 0.002 * page_num)
        r["_combined_score"] = base_score + bonus + lab_bonus + position_bonus + figure_bonus + task_bonus
        doc_tokens = _tokenize_list(content)
        doc_tokens_list.append(doc_tokens)
        reranked.append(r)

    if not reranked:
        return []

    if doc_tokens_list:
        avg_dl = sum(len(toks) for toks in doc_tokens_list) / len(doc_tokens_list)
        df_counts = {}
        for toks in doc_tokens_list:
            for term in set(toks):
                df_counts[term] = df_counts.get(term, 0) + 1
        for r, toks in zip(reranked, doc_tokens_list):
            bm25 = _bm25_score(sorted(query_tokens), toks, avg_dl, df_counts, len(doc_tokens_list))
            r["_combined_score"] += 0.05 * bm25

    reranked.sort(key=lambda x: x.get("_combined_score", 0), reverse=True)
    best_candidate_score = reranked[0].get("_combined_score", 0)

    filtered = []
    section_counts = {}
    for r in reranked:
        if r.get("_combined_score", 0) < (best_candidate_score - CONTEXT_SCORE_TOLERANCE):
            continue
        key = (r.get("lab_name"), r.get("section_name"), r.get("page_num"))
        section_counts.setdefault(key, 0)
        if section_counts[key] >= CONTEXT_SECTION_LIMIT:
            continue
        section_counts[key] += 1
        filtered.append(r)
        if len(filtered) >= CONTEXT_FINAL_K:
            break

    filtered.sort(key=lambda x: (x.get("section_name", ""), x.get("page_num", 0)))
    formatted = []
    for r in filtered:
        tag = f"{r.get('lab_name', 'Lab ?')} • {r.get('section_name', 'Section ?')} (p.{r.get('page_num', '?')})"
        content = (r.get("content") or "").strip()
        if len(content) > CONTEXT_MAX_CHARS:
            content = content[:CONTEXT_MAX_CHARS] + "..."
        formatted.append(f"[{tag}]\n{content}")

    return formatted


@app.post("/chat/{lab_number}")
def chat(request: ChatRequest):
    _require_supabase()
    _require_llm()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    context = retrieve_context(question)
    if not context:
        return {"answer": "I cannot find that information in the lab manual."}

    history_txt = "\n".join([f"Q: {t.get('user','')}\nA: {t.get('ai','')}" for t in conversation_history[-2:]])
    context_txt = "\n---\n".join(context)

    prompt = f"""
    You are a helpful electrical engineering lab assistant.
    Answer only using the facts explicitly stated in the context snippets below.
    When you use a fact, cite its tag in brackets (e.g., [Lab 1 • Procedure (p.3)]).
    Every sentence of your answer should include a citation to a provided snippet.
    If the answer is not in the context, reply exactly: "I cannot find that information in the lab manual." Do NOT guess.

    Context (do not use outside knowledge):
    ---
    {context_txt}
    ---

    Recent conversation (for continuity, avoid repeating):
    {history_txt}

    Question: {question}
    Answer:
    """

    answer = llm.invoke(prompt)
    conversation_history.append({"user": question, "ai": str(answer)})
    return {"answer": str(answer)}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rt = get_runtime()
    selected_report = None
    selected_metric = None
    selected_metric_value = None
    if isinstance(rt, CircuitDebugHybridRuntime):
        cfg = _load_hybrid_config()
        if cfg:
            selected_report = cfg.get("auto_selected_report")
            selected_metric = cfg.get("auto_selected_metric")
            selected_metric_value = cfg.get("auto_selected_metric_value")
    backend = "llm_knn_hybrid" if rt.__class__.__name__.endswith("HybridRuntime") else "tabular_xgboost"
    return HealthResponse(
        ok=True,
        backend=backend,
        circuits=len(rt.list_circuits()),
        family_pair_models=len(rt.family_pair_models),
        pair_threshold=float(rt.pair_threshold),
        selected_model_report=selected_report,
        selected_model_metric=selected_metric,
        selected_model_metric_value=selected_metric_value,
    )


@app.get("/model", response_model=ModelInfoResponse)
def model() -> ModelInfoResponse:
    rt = get_runtime()
    backend = "llm_knn_hybrid" if rt.__class__.__name__.endswith("HybridRuntime") else "tabular_xgboost"
    if backend != "llm_knn_hybrid":
        return ModelInfoResponse(
            ok=True,
            backend=backend,
            adapter_dir=None,
            knn_ref_file=None,
            knn_index_file=None,
            model_name="N/A",
        )

    cfg = _load_hybrid_config() or {}
    effective_model_name = str(os.environ.get("CIRCUIT_DEBUG_BASE_MODEL", cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")))
    return ModelInfoResponse(
        ok=True,
        backend=backend,
        adapter_dir=cfg.get("adapter_dir"),
        knn_ref_file=cfg.get("knn_ref_file"),
        knn_index_file=cfg.get("knn_index_file"),
        model_name=effective_model_name,
        selected_model_report=cfg.get("auto_selected_report"),
        selected_model_metric=cfg.get("auto_selected_metric"),
        selected_model_metric_value=cfg.get("auto_selected_metric_value"),
    )


@app.post("/admin/refresh-model", include_in_schema=True)
def refresh_model() -> dict[str, Any]:
    _refresh_hybrid_assets()
    get_runtime.cache_clear()
    return {
        "ok": True,
        "message": "Hybrid assets rebuilt using best report and runtime reloaded.",
        "model": model().dict(),
    }


@app.get("/circuits")
def list_circuits() -> dict[str, Any]:
    rt = get_runtime()
    names = rt.list_circuits()
    return {"count": len(names), "circuits": names}


@app.get("/circuits/{circuit_name}/nodes")
def get_circuit_nodes(circuit_name: str) -> dict[str, Any]:
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

# Root helper for quick manual check
@app.get("/")
def root():
    return {
        "message": "SPICE Lab Assistant API is running",
        "routes": [
            "/chat/{lab_number}",
            "/circuits",
            "/circuits/{circuit_name}/nodes",
            "/debug",
            "/health",
        ],
    }

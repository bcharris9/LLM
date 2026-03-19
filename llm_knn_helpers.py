"""Shared helpers for the hybrid debug runtime's LLM and KNN scoring logic."""

from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def choose_device() -> tuple[str, torch.dtype]:
    """Pick a reasonable inference device and dtype for the current machine."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(instruction: str, input_text: str, response_style: str) -> str:
    """Render the prompt format expected by the fine-tuned debug model."""
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if response_style == "faulttype_diag_fix":
        response_header = (
            "### Response:\n"
            "Return exactly two lines.\n"
            "FaultType: one of param_drift|missing_component|pin_open|swapped_nodes|"
            "short_between_nodes|resistor_value_swap|resistor_wrong_value.\n"
            "Diagnosis: concise sentence. Fix: concise sentence.\n"
        )
    else:
        response_header = (
            "### Response:\n"
            "Return exactly one line in this format: Diagnosis: <text>. Fix: <text>.\n"
            "Choose the single best fault family: parameter drift, missing component, open connection, "
            "swapped terminals, short between nodes, resistor values swapped, wrong resistor value.\n"
        )

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            f"{response_header}"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        f"{response_header}"
    )


def normalize_text(text: str) -> str:
    """Lowercase and whitespace-normalize freeform text."""
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def has_diag_and_fix(text: str) -> bool:
    """Check whether a model output already contains both required fields."""
    low = (text or "").lower()
    return ("diagnosis:" in low) and ("fix:" in low)


def normalize_fault_type_label(raw: str) -> str:
    """Map label aliases onto the canonical runtime fault-type names."""
    key = (raw or "").strip().lower()
    alias = {
        "param_drift": "param_drift",
        "parameter_drift": "param_drift",
        "parameter drift": "param_drift",
        "param drift": "param_drift",
        "missing_component": "missing_component",
        "missing component": "missing_component",
        "pin_open": "pin_open",
        "open_connection": "pin_open",
        "open connection": "pin_open",
        "swapped_nodes": "swapped_nodes",
        "swapped_terminals": "swapped_nodes",
        "swapped terminals": "swapped_nodes",
        "nodes swapped": "swapped_nodes",
        "short_between_nodes": "short_between_nodes",
        "short between nodes": "short_between_nodes",
        "resistor_value_swap": "resistor_value_swap",
        "resistor value swap": "resistor_value_swap",
        "resistor values swapped": "resistor_value_swap",
        "resistor_wrong_value": "resistor_wrong_value",
        "wrong resistor value": "resistor_wrong_value",
    }
    return alias.get(key, "unknown")


def force_diag_fix_format(text: str, response_style: str) -> str:
    """Coerce arbitrary model output into the stable diagnosis/fix response format."""
    raw = (text or "").strip()
    if not raw:
        if response_style == "faulttype_diag_fix":
            return (
                "FaultType: unknown\n"
                "Diagnosis: unknown fault. Fix: inspect wiring and component values."
            )
        return "Diagnosis: unknown fault. Fix: inspect wiring and component values."

    # Normalize whitespace for robust extraction.
    clean = re.sub(r"\s+", " ", raw).strip()

    # Optional explicit fault type line.
    fault_type = "unknown"
    m = re.search(r"faulttype:\s*([a-z_ \-]+)", clean, flags=re.IGNORECASE)
    if m:
        fault_type = normalize_fault_type_label(m.group(1))

    low = clean.lower()
    dpos = low.find("diagnosis:")
    fpos = low.find("fix:")
    if dpos >= 0 and fpos >= 0:
        if fpos > dpos:
            diagnosis = clean[dpos + len("diagnosis:"):fpos].strip(" .;")
            fix = clean[fpos + len("fix:"):].strip(" .;")
        else:
            fix = clean[fpos + len("fix:"):dpos].strip(" .;")
            diagnosis = clean[dpos + len("diagnosis:"):].strip(" .;")
        diagnosis = diagnosis or "unknown fault"
        fix = fix or "inspect wiring and component values"
        if response_style == "faulttype_diag_fix":
            if fault_type == "unknown":
                fault_type = classify_fault_text(f"Diagnosis: {diagnosis}. Fix: {fix}.")
            return f"FaultType: {fault_type}\nDiagnosis: {diagnosis}. Fix: {fix}."
        return f"Diagnosis: {diagnosis}. Fix: {fix}."

    # Class-only outputs: promote to canonical diagnosis/fix for stable scoring.
    if response_style == "faulttype_diag_fix" and fault_type != "unknown":
        return f"FaultType: {fault_type}\n{canonical_completion_for_fault(fault_type)}"

    # Fallback: use first sentence as diagnosis and second as fix when possible.
    sentences = [s.strip(" .;") for s in re.split(r"[.!?]+", clean) if s.strip()]
    diagnosis = sentences[0] if sentences else "unknown fault"
    fix = sentences[1] if len(sentences) > 1 else "inspect wiring and component values"
    if response_style == "faulttype_diag_fix":
        if fault_type == "unknown":
            fault_type = classify_fault_text(f"Diagnosis: {diagnosis}. Fix: {fix}.")
        return f"FaultType: {fault_type}\nDiagnosis: {diagnosis}. Fix: {fix}."
    return f"Diagnosis: {diagnosis}. Fix: {fix}."


def classify_fault_text(text: str) -> str:
    """Infer the canonical fault label from generated text."""
    m = re.search(r"faulttype:\s*([a-z_ \-]+)", (text or ""), flags=re.IGNORECASE)
    if m:
        label = normalize_fault_type_label(m.group(1))
        if label != "unknown":
            return label

    t = (text or "").lower()
    if "wrong resistor value" in t:
        return "resistor_wrong_value"
    if "resistor values were swapped" in t or ("swapped" in t and "resistor" in t):
        return "resistor_value_swap"
    if "parameter drift" in t or "parameter change" in t:
        return "param_drift"
    if "missing component" in t:
        return "missing_component"
    if "open connection" in t:
        return "pin_open"
    if "swapped terminals" in t or "nodes were swapped" in t:
        return "swapped_nodes"
    if "short between nodes" in t or "unintended short" in t:
        return "short_between_nodes"
    return "unknown"


FAULT_TYPE_ORDER = [
    "param_drift",
    "missing_component",
    "pin_open",
    "swapped_nodes",
    "short_between_nodes",
    "resistor_value_swap",
    "resistor_wrong_value",
]


def canonical_completion_for_fault(fault_type: str) -> str:
    """Return the canonical diagnosis/fix wording for a known fault class."""
    templates = {
        "param_drift": (
            "Diagnosis: parameter drift in one or more components. "
            "Fix: restore drifted parameter values to their intended targets."
        ),
        "missing_component": (
            "Diagnosis: missing component in the circuit path. "
            "Fix: reinsert the missing component with the intended value/model."
        ),
        "pin_open": (
            "Diagnosis: open connection on a component terminal. "
            "Fix: reconnect the opened pin to its intended node."
        ),
        "swapped_nodes": (
            "Diagnosis: swapped terminals on a component/source. "
            "Fix: swap the two connections back to their intended nodes."
        ),
        "short_between_nodes": (
            "Diagnosis: unintended short between nodes. "
            "Fix: remove the short and restore proper wiring."
        ),
        "resistor_value_swap": (
            "Diagnosis: resistor values were swapped between two resistors. "
            "Fix: restore each resistor to its intended value."
        ),
        "resistor_wrong_value": (
            "Diagnosis: wrong resistor value on one resistor. "
            "Fix: change that resistor back to its intended value."
        ),
    }
    return templates.get(
        fault_type,
        "Diagnosis: unknown fault. Fix: inspect wiring and component values.",
    )


def prerule_fault_type(input_text: str) -> str | None:
    """Apply simple deterministic rules before invoking the LLM scorer."""
    text = (input_text or "").strip()
    if not text:
        return None

    # This signature is unambiguous in the current data: no parsed measurements.
    # It consistently maps to a missing-component class.
    has_measured_none = False
    has_delta_none = False
    sim_success_false = False
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low == "measured: none":
            has_measured_none = True
        elif low == "deltasvsgolden: none":
            has_delta_none = True
        elif low == "simsuccess: false":
            sim_success_false = True

    if has_measured_none and (has_delta_none or sim_success_false):
        return "missing_component"
    return None


def parse_measurement_features(input_text: str) -> dict[str, float]:
    """Parse numeric feature fields back out of an instruct-style input block."""
    text = (input_text or "").strip()
    if not text:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        low_line = stripped.lower()
        m = re.match(r"^\s*([^:]+?)\s*[:=]\s*(.*)$", stripped)
        if not m:
            continue
        raw_label = m.group(1).strip()
        raw_value = m.group(2).strip()
        label = raw_label.lower()
        if label == "lab":
            lab = raw_value.lower()
            if lab:
                # Preserve sign information (e.g. ..._-4 vs ..._4) to avoid collisions.
                lab = lab.replace("-", "__neg__")
                safe_lab = re.sub(r"[^a-z0-9_]+", "_", lab).strip("_")
                if safe_lab:
                    out[f"lab__{safe_lab}"] = 1.0
            continue
        if label == "simsuccess":
            val = raw_value.lower()
            if val in {"true", "1", "yes"}:
                out["sim_success"] = 1.0
            elif val in {"false", "0", "no"}:
                out["sim_success"] = 0.0
            continue
        if label not in {"measured", "deltasvsgolden", "goldenmeasurements"}:
            continue
        prefix, payload = label, raw_value
        payload = payload.strip()
        if payload.lower() == "none":
            if prefix == "measured":
                out["measured_none"] = 1.0
                out["measured_count"] = 0.0
            elif prefix == "deltasvsgolden":
                out["deltas_none"] = 1.0
                out["deltas_count"] = 0.0
            elif prefix == "goldenmeasurements":
                out["golden_count"] = 0.0
            continue
        numeric_count = 0
        for item in payload.split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            key, raw_val = item.split("=", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            try:
                out[key] = float(raw_val)
                numeric_count += 1
            except Exception:
                continue
        if prefix == "measured":
            out["measured_none"] = 0.0
            out["measured_count"] = float(numeric_count)
        elif prefix == "deltasvsgolden":
            out["deltas_none"] = 0.0
            out["deltas_count"] = float(numeric_count)
        elif prefix == "goldenmeasurements":
            out["golden_count"] = float(numeric_count)
    return out


def parse_lab_id(input_text: str) -> str | None:
    """Extract a normalized lab identifier from an instruct-style input block."""
    text = (input_text or "").strip()
    if not text:
        return None
    for line in text.splitlines():
        line_l = line.strip().lower()
        if not line_l.startswith("lab:") and not line_l.startswith("lab ="):
            continue
        m = re.match(r"^\s*lab\s*[:=]\s*(.*)\s*$", line, flags=re.IGNORECASE)
        if not m:
            continue
        lab = m.group(1).strip().lower()
        if not lab:
            return None
        # Preserve sign information (e.g. ..._-4 vs ..._4) to avoid collisions.
        lab = lab.replace("-", "__neg__")
        safe_lab = re.sub(r"[^a-z0-9_]+", "_", lab).strip("_")
        return safe_lab or None
    return None


def build_knn_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a simple dense KNN index from labeled instruct examples."""
    feats: list[dict[str, float]] = []
    classes: list[str] = []
    labs: list[str | None] = []
    key_set: set[str] = set()
    for row in rows:
        input_text = str(row.get("input", ""))
        f = parse_measurement_features(input_text)
        if not f:
            continue
        c = classify_fault_text(str(row.get("output", "")))
        if c == "unknown":
            continue
        feats.append(f)
        classes.append(c)
        labs.append(parse_lab_id(input_text))
        key_set.update(f.keys())
    keys = sorted(key_set)
    key_to_col = {k: i for i, k in enumerate(keys)}
    vectors: list[list[float]] = []
    for f in feats:
        vectors.append([f.get(k, 0.0) for k in keys])
    means: list[float] = []
    stds: list[float] = []
    if vectors and keys:
        n = float(len(vectors))
        for i in range(len(keys)):
            m = 0.0
            for row in vectors:
                m += row[i]
            m /= n
            means.append(m)
        for i in range(len(keys)):
            var = 0.0
            for row in vectors:
                d = row[i] - means[i]
                var += d * d
            if len(vectors) > 1:
                var /= float(len(vectors) - 1)
            s = math.sqrt(var)
            if s < 1e-12:
                s = 1.0
            stds.append(s)
    zvectors: list[list[float]] = []
    if vectors:
        for row in vectors:
            zrow: list[float] = []
            for i, val in enumerate(row):
                zrow.append((val - means[i]) / stds[i])
            zvectors.append(zrow)

    lab_to_indices: dict[str, list[int]] = {}
    for i, lab in enumerate(labs):
        if not lab:
            continue
        lab_to_indices.setdefault(lab, []).append(i)

    return {
        "keys": keys,
        "key_to_col": key_to_col,
        "vectors": vectors,
        "zvectors": zvectors,
        "classes": classes,
        "labs": labs,
        "lab_to_indices": lab_to_indices,
        "means": means,
        "stds": stds,
    }


def knn_class_probs(
    index: dict[str, Any],
    input_text: str,
    k: int,
    weighted_vote: bool,
    standardize: bool,
    eps: float,
) -> dict[str, float]:
    """Estimate class probabilities from nearest neighbors in feature space."""
    keys: list[str] = index.get("keys", [])
    vectors: list[list[float]] = index.get("vectors", [])
    zvectors: list[list[float]] = index.get("zvectors", [])
    key_to_col: dict[str, int] = index.get("key_to_col", {})
    classes: list[str] = index.get("classes", [])
    means: list[float] = index.get("means", [])
    stds: list[float] = index.get("stds", [])
    lab_to_indices: dict[str, list[int]] = index.get("lab_to_indices", {})
    if not keys or not vectors or not classes:
        p = 1.0 / float(max(1, len(FAULT_TYPE_ORDER)))
        return {c: p for c in FAULT_TYPE_ORDER}

    f = parse_measurement_features(input_text)
    q_lab = parse_lab_id(input_text)
    if not key_to_col and keys:
        key_to_col = {k: i for i, k in enumerate(keys)}

    # Use sparse distances: only compare against dimensions present in query.
    q_present_keys = [k for k in f.keys() if k in key_to_col]
    use_sparse = bool(q_present_keys)
    if use_sparse:
        q_cols = [key_to_col[k] for k in q_present_keys]
        if standardize and means and stds and len(means) == len(keys):
            q_vals = [
                (f[k] - means[j]) / stds[j] if stds[j] != 0 else f[k]
                for k, j in zip(q_present_keys, q_cols)
            ]
            ref_vectors = zvectors if zvectors else vectors
        else:
            q_vals = [f[k] for k in q_present_keys]
            ref_vectors = vectors
    else:
        # Fallback to legacy dense representation when no overlapping keys were found.
        q_vals = [f.get(key, 0.0) for key in keys]
        if standardize and means and stds and len(means) == len(q_vals):
            q_vals = [(q_vals[i] - means[i]) / stds[i] for i in range(len(q_vals))]
            ref_vectors = zvectors if zvectors else vectors
        else:
            ref_vectors = vectors

    candidate_indices: list[int] | None = None
    if q_lab and lab_to_indices:
        same_lab = lab_to_indices.get(q_lab)
        if same_lab:
            candidate_indices = same_lab

    if candidate_indices is None:
        candidate_indices = list(range(len(ref_vectors)))

    kk = max(1, min(int(k), len(candidate_indices)))
    all_d: list[tuple[float, int]] = []
    for idx in candidate_indices:
        ref = ref_vectors[idx]
        s = 0.0
        if use_sparse:
            for qi, ci in enumerate(q_cols):
                d = q_vals[qi] - ref[ci]
                s += d * d
        else:
            for i, rv in enumerate(ref):
                d = q_vals[i] - rv
                s += d * d
        all_d.append((s, idx))
    all_d.sort(key=lambda x: x[0])
    top = all_d[:kk]

    votes: dict[str, float] = {}
    total_vote = 0.0
    for dist2, idx in top:
        c = classes[idx]
        if c == "unknown":
            continue
        if weighted_vote:
            w = 1.0 / (dist2 + max(1e-12, float(eps)))
        else:
            w = 1.0
        votes[c] = votes.get(c, 0.0) + w
        total_vote += w

    if total_vote <= 0.0:
        p = 1.0 / float(max(1, len(FAULT_TYPE_ORDER)))
        return {c: p for c in FAULT_TYPE_ORDER}

    probs: dict[str, float] = {}
    for c in FAULT_TYPE_ORDER:
        probs[c] = votes.get(c, 0.0) / total_vote
    return probs


def knn_penalties(probs: dict[str, float], alpha: float) -> dict[str, float]:
    """Convert KNN class probabilities into additive negative-log penalties."""
    out: dict[str, float] = {}
    a = max(0.0, float(alpha))
    for c in FAULT_TYPE_ORDER:
        p = max(1e-8, float(probs.get(c, 0.0)))
        out[c] = a * (-math.log(p))
    return out


def build_class_candidates(response_style: str) -> list[tuple[str, str]]:
    """Construct the full candidate completions used for class scoring."""
    items: list[tuple[str, str]] = []
    for fault_type in FAULT_TYPE_ORDER:
        body = canonical_completion_for_fault(fault_type)
        if response_style == "faulttype_diag_fix":
            text = f"FaultType: {fault_type}\n{body}"
        else:
            text = body
        items.append((fault_type, text))
    return items


def build_faulttype_only_candidates(response_style: str) -> list[tuple[str, str]]:
    """Construct short candidates that contain only the class label."""
    items: list[tuple[str, str]] = []
    for fault_type in FAULT_TYPE_ORDER:
        if response_style == "faulttype_diag_fix":
            text = f"FaultType: {fault_type}\n"
        else:
            text = f"{fault_type}\n"
        items.append((fault_type, text))
    return items


def score_output_candidate(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt_ids: list[int],
    prompt_tti: list[int] | None,
    candidate_text: str,
    needs_token_type_ids: bool,
) -> float:
    """Score one candidate completion by evaluating its language-model loss."""
    suffix = tokenizer.eos_token or ""
    candidate = (candidate_text or "").strip() + suffix
    cand_tok = tokenizer(candidate, add_special_tokens=False)
    cand_ids = cand_tok.get("input_ids", [])
    if not cand_ids:
        cand_ids = [tokenizer.eos_token_id]
    input_ids = prompt_ids + cand_ids
    labels = ([-100] * len(prompt_ids)) + cand_ids

    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }
    if needs_token_type_ids:
        pt = prompt_tti if prompt_tti is not None else [0] * len(prompt_ids)
        ct = cand_tok.get("token_type_ids", [0] * len(cand_ids))
        token_type_ids = pt + ct
        batch["token_type_ids"] = torch.tensor([token_type_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        loss = model(**batch).loss
    return float(loss.item())


def predict_by_class_scoring(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    response_style: str,
    class_penalties: dict[str, float] | None = None,
) -> str:
    """Choose the best full completion by scoring every class-specific candidate."""
    prompt_tok = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_tok.get("input_ids", [])
    prompt_tti = prompt_tok.get("token_type_ids")
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    needs_token_type_ids = "token_type_ids" in forward_params

    best_text = ""
    best_score = float("inf")
    for fault_type, candidate_text in build_class_candidates(response_style):
        s = score_output_candidate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_ids=prompt_ids,
            prompt_tti=prompt_tti,
            candidate_text=candidate_text,
            needs_token_type_ids=needs_token_type_ids,
        )
        if class_penalties:
            s += float(class_penalties.get(fault_type, 0.0))
        if s < best_score:
            best_score = s
            best_text = candidate_text
    return best_text


def predict_by_faulttype_scoring(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    response_style: str,
) -> str:
    """Score only fault-type prefixes, then expand the winner to canonical text."""
    prompt_tok = tokenizer(prompt, add_special_tokens=False)
    prompt_ids = prompt_tok.get("input_ids", [])
    prompt_tti = prompt_tok.get("token_type_ids")
    forward_params = set(inspect.signature(model.forward).parameters.keys())
    needs_token_type_ids = "token_type_ids" in forward_params

    best_fault_type = "unknown"
    best_score = float("inf")
    for fault_type, candidate_text in build_faulttype_only_candidates(response_style):
        s = score_output_candidate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_ids=prompt_ids,
            prompt_tti=prompt_tti,
            candidate_text=candidate_text,
            needs_token_type_ids=needs_token_type_ids,
        )
        if s < best_score:
            best_score = s
            best_fault_type = fault_type

    body = canonical_completion_for_fault(best_fault_type)
    if response_style == "faulttype_diag_fix":
        return f"FaultType: {best_fault_type}\n{body}"
    return body


def predict_by_knn_only(
    input_text: str,
    response_style: str,
    knn_index: dict[str, Any],
    k: int,
    weighted_vote: bool,
    standardize: bool,
    eps: float,
) -> str:
    """Return the highest-probability class from KNN without invoking the LLM."""
    probs = knn_class_probs(
        index=knn_index,
        input_text=input_text,
        k=k,
        weighted_vote=weighted_vote,
        standardize=standardize,
        eps=eps,
    )
    best_fault_type = max(FAULT_TYPE_ORDER, key=lambda c: probs.get(c, 0.0))
    body = canonical_completion_for_fault(best_fault_type)
    if response_style == "faulttype_diag_fix":
        return f"FaultType: {best_fault_type}\n{body}"
    return body


def load_model(
    model_name: str,
    adapter_dir: Path | None,
    device: str,
    dtype: torch.dtype,
):
    """Load either a base model alone or a base model plus LoRA adapter."""
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig.parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir)) if adapter_dir else base
    model.eval()
    model.to(device)
    return model

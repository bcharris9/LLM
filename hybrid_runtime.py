from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib

try:
    from LLM import llm_knn_helpers as helpers
    from LLM.runtime import (
        DebugResult,
        build_circuit_catalog,
        measurement_key_for_node,
        measurement_key_for_vsource_current,
    )
except ModuleNotFoundError as e:
    if e.name != "LLM":
        raise
    import llm_knn_helpers as helpers  # type: ignore[no-redef]
    from runtime import (  # type: ignore[no-redef]
        DebugResult,
        build_circuit_catalog,
        measurement_key_for_node,
        measurement_key_for_vsource_current,
    )


def _format_measurement_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _metric_suffix_priority(key: str, stat_mode: str) -> tuple[int, bool]:
    low = key.lower()
    if low.endswith("_max"):
        suffix_rank = 0
    elif low.endswith("_rms"):
        suffix_rank = 1
    elif low.endswith("_min"):
        suffix_rank = 2
    else:
        suffix_rank = 3
    if stat_mode == "full":
        allowed = True
    elif stat_mode == "max_only":
        allowed = low.endswith("_max") or ("_" not in low)
    else:
        allowed = low.endswith("_max") or low.endswith("_rms") or ("_" not in low)
    return suffix_rank, allowed


def _measurement_group_priority(key: str, prefer_voltage_keys: bool) -> int:
    low = key.lower()
    if low.startswith("v_"):
        return 0 if prefer_voltage_keys else 1
    if low.startswith("i_"):
        return 1 if prefer_voltage_keys else 0
    if low in {"temp", "tnom"}:
        return 2
    if low in {"method", "solver"}:
        return 3
    return 4


def _ordered_measurement_keys(
    measurements: dict[str, Any],
    max_measurements: int,
    voltage_only: bool,
    stat_mode: str,
    prefer_voltage_keys: bool,
) -> list[str]:
    ranked: list[tuple[tuple[int, int, str], str]] = []
    for key in measurements.keys():
        low = str(key).lower()
        if voltage_only and not low.startswith("v_"):
            continue
        suffix_rank, allowed = _metric_suffix_priority(low, stat_mode)
        if not allowed:
            continue
        group_rank = _measurement_group_priority(low, prefer_voltage_keys)
        ranked.append(((group_rank, suffix_rank, low), str(key)))
    ranked.sort(key=lambda x: x[0])
    keys = [k for _, k in ranked]
    return keys[: max(1, max_measurements)] if keys else []


def _compact_measurements(
    measurements: dict[str, Any],
    max_measurements: int,
    voltage_only: bool,
    stat_mode: str,
    prefer_voltage_keys: bool,
) -> str:
    if not measurements:
        return "none"
    keys = _ordered_measurement_keys(
        measurements, max_measurements, voltage_only, stat_mode, prefer_voltage_keys
    )
    if not keys:
        return "none"
    return "; ".join(f"{k}={_format_measurement_value(measurements.get(k))}" for k in keys)


def _compact_deltas(
    measurements: dict[str, Any],
    golden: dict[str, Any],
    max_deltas: int,
    voltage_only: bool,
    stat_mode: str,
    prefer_voltage_keys: bool,
) -> str:
    if not measurements or not golden:
        return "none"
    keys = _ordered_measurement_keys(
        measurements, max_deltas, voltage_only, stat_mode, prefer_voltage_keys
    )
    out: list[str] = []
    for k in keys:
        v = measurements.get(k)
        g = golden.get(k)
        if isinstance(v, (int, float)) and isinstance(g, (int, float)):
            out.append(f"{k}_delta={float(v) - float(g):.6g}")
    return "; ".join(out) if out else "none"


def _parse_diag_fix(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "unknown fault", "inspect wiring and component values"
    low = raw.lower()
    dpos = low.find("diagnosis:")
    fpos = low.find("fix:")
    if dpos >= 0 and fpos >= 0 and fpos > dpos:
        diagnosis = raw[dpos + len("diagnosis:") : fpos].strip(" .;\n")
        fix = raw[fpos + len("fix:") :].strip(" .;\n")
        return diagnosis or "unknown fault", fix or "inspect wiring and component values"
    return "unknown fault", "inspect wiring and component values"


def _resolve_config_path(path_value: object, *, hybrid_assets_dir: Path, api_dir: Path) -> Path:
    """
    Resolve paths stored in hybrid_config across platforms.

    Packaged configs may contain Windows-style separators and older paths rooted at
    `circuit_debug_api/` or `LLM/`. We normalize those and try sensible bases.
    """
    raw = str(path_value or "").strip()
    if not raw:
        return Path(raw)

    # Normalize Windows separators on POSIX and vice versa.
    normalized = raw.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p

    candidates: list[Path] = []

    # Directly relative to current working dir / process.
    candidates.append(p)

    # Relative to the API dir (`LLM/`) and hybrid assets dir.
    candidates.append(api_dir / p)
    candidates.append(hybrid_assets_dir / p)

    # Backward-compatible configs sometimes embed a leading folder name.
    parts = list(p.parts)
    if parts and parts[0] in {"circuit_debug_api", "LLM"}:
        stripped = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        candidates.append(api_dir / stripped)
        candidates.append(hybrid_assets_dir / stripped)

    # Return the first existing path; otherwise return best normalized guess.
    for c in candidates:
        if c.exists():
            return c
    return api_dir / p


class CircuitDebugHybridRuntime:
    """
    LLM + KNN hybrid runtime using the same scoring/KNN functions as pipeline/test_lora_model.py.

    This keeps the external API shape compatible with the existing tabular runtime.
    """

    def __init__(
        self,
        *,
        catalog_path: str | Path,
        hybrid_assets_dir: str | Path,
        auto_build_catalog_from: str | Path | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.hybrid_assets_dir = Path(hybrid_assets_dir)
        self.api_dir = self.hybrid_assets_dir.parent
        self.auto_build_catalog_from = Path(auto_build_catalog_from) if auto_build_catalog_from else None

        self.config_path = self.hybrid_assets_dir / "hybrid_config.json"
        if not self.catalog_path.exists():
            if not self.auto_build_catalog_from:
                raise FileNotFoundError(f"Missing catalog: {self.catalog_path}")
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            catalog = build_circuit_catalog(self.auto_build_catalog_from)
            self.catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing hybrid config: {self.config_path}")

        catalog_doc = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.catalog = catalog_doc["circuits"]
        self.catalog_meta = {k: v for k, v in catalog_doc.items() if k != "circuits"}
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        self._eval_mod: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._knn_index: dict[str, Any] | None = None
        self._knn_index_loaded = False
        # compatibility with existing health endpoint code
        self.family_pair_models: dict[str, Any] = {}
        self.pair_threshold = 0.0

    def list_circuits(self) -> list[str]:
        return sorted(self.catalog.keys())

    def has_circuit(self, circuit_name: str) -> bool:
        return circuit_name in self.catalog

    def circuit_spec(self, circuit_name: str) -> dict[str, Any]:
        return dict(self.catalog[circuit_name])

    def _load_eval_module(self):
        if self._eval_mod is None:
            self._eval_mod = helpers
        return self._eval_mod

    def _ensure_knn_index(self) -> dict[str, Any]:
        if self._knn_index_loaded and self._knn_index is not None:
            return self._knn_index
        mod = self._load_eval_module()
        knn_index_joblib = self.hybrid_assets_dir / "knn_index.joblib"
        if knn_index_joblib.exists():
            self._knn_index = joblib.load(knn_index_joblib)
        else:
            ref_path = _resolve_config_path(
                self.config.get("knn_ref_file"),
                hybrid_assets_dir=self.hybrid_assets_dir,
                api_dir=self.api_dir,
            )
            rows = mod.load_jsonl(ref_path)
            self._knn_index = mod.build_knn_index(rows)
        self._knn_index_loaded = True
        return self._knn_index or {}

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        mod = self._load_eval_module()
        self._device, self._dtype = mod.choose_device()
        model_name = str(self.config["model_name"])
        adapter_dir = _resolve_config_path(
            self.config.get("adapter_dir"),
            hybrid_assets_dir=self.hybrid_assets_dir,
            api_dir=self.api_dir,
        )
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Hybrid adapter dir not found: {adapter_dir}")
        tokenizer = helpers.AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = mod.load_model(model_name, adapter_dir, self._device, self._dtype)
        self._tokenizer = tokenizer
        self._model = model

    def _normalize_measurements_from_request(
        self,
        node_voltages: dict[str, float] | None,
        source_currents: dict[str, float] | None,
        measurement_overrides: dict[str, float] | None,
        temp: float | None,
        tnom: float | None,
        spec: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        measured: dict[str, Any] = {}
        used_v_keys: list[str] = []
        used_i_keys: list[str] = []

        if node_voltages:
            for node_name, value in node_voltages.items():
                key = measurement_key_for_node(str(node_name))
                measured[key] = float(value)
                used_v_keys.append(key)
        if source_currents:
            for source_name, value in source_currents.items():
                key = measurement_key_for_vsource_current(str(source_name))
                measured[key] = float(value)
                used_i_keys.append(key)
        if measurement_overrides:
            for k, v in measurement_overrides.items():
                measured[str(k).lower()] = v

        defaults = spec.get("golden_defaults", {})
        measured.setdefault("temp", float(temp if temp is not None else defaults.get("temp", 27.0)))
        measured.setdefault("tnom", float(tnom if tnom is not None else defaults.get("tnom", 27.0)))
        # Keep these text values for prompt parity (KNN parser ignores nonnumeric values).
        measured.setdefault("method", str(defaults.get("method", "trap")))
        measured.setdefault("solver", str(defaults.get("solver", "Normal")))
        return measured, sorted(set(used_v_keys)), sorted(set(used_i_keys))

    def _build_input_text(
        self,
        *,
        circuit_name: str,
        measured: dict[str, Any],
        golden: dict[str, Any],
    ) -> str:
        cfg = self.config
        voltage_only = bool(cfg.get("voltage_only", False))
        stat_mode = str(cfg.get("measurement_stat_mode", "max_only"))
        prefer_voltage_keys = bool(cfg.get("prefer_voltage_keys", True))
        max_measurements = int(cfg.get("max_measurements", 24))
        max_deltas = int(cfg.get("max_deltas", 24))

        lines: list[str] = []
        if bool(cfg.get("include_lab_id_in_prompt", False)):
            lines.append(f"Lab: {circuit_name}")
        lines.append("SimSuccess: True")
        lines.append(
            "DeltasVsGolden: "
            + _compact_deltas(
                measured,
                golden,
                max_deltas=max_deltas,
                voltage_only=voltage_only,
                stat_mode=stat_mode,
                prefer_voltage_keys=prefer_voltage_keys,
            )
        )
        lines.append(
            "Measured: "
            + _compact_measurements(
                measured,
                max_measurements=max_measurements,
                voltage_only=voltage_only,
                stat_mode=stat_mode,
                prefer_voltage_keys=prefer_voltage_keys,
            )
        )
        lines.append("Task: choose the most likely fault class and provide diagnosis/fix.")
        return "\n".join(lines)

    def _score_class_candidates_with_knn(self, prompt: str, input_text: str) -> tuple[str, float, list[dict[str, float]]]:
        mod = self._load_eval_module()
        self._ensure_model()
        knn_index = self._ensure_knn_index()
        response_style = str(self.config.get("response_style", "faulttype_diag_fix"))
        k = int(self.config.get("knn_k", 1))
        alpha = float(self.config.get("knn_alpha", 1.0))
        weighted_vote = bool(self.config.get("knn_weighted_vote", True))
        standardize = bool(self.config.get("knn_standardize", False))
        eps = float(self.config.get("knn_eps", 1e-9))

        probs = mod.knn_class_probs(
            index=knn_index,
            input_text=input_text,
            k=k,
            weighted_vote=weighted_vote,
            standardize=standardize,
            eps=eps,
        )
        penalties = mod.knn_penalties(probs, alpha)

        prompt_tok = self._tokenizer(prompt, add_special_tokens=False)
        prompt_ids = prompt_tok.get("input_ids", [])
        prompt_tti = prompt_tok.get("token_type_ids")
        forward_params = set(inspect.signature(self._model.forward).parameters.keys())
        needs_tti = "token_type_ids" in forward_params

        scored: list[tuple[str, str, float]] = []
        for fault_type, candidate_text in mod.build_class_candidates(response_style):
            s = mod.score_output_candidate(
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                prompt_ids=prompt_ids,
                prompt_tti=prompt_tti,
                candidate_text=candidate_text,
                needs_token_type_ids=needs_tti,
            )
            s += float(penalties.get(fault_type, 0.0))
            scored.append((fault_type, candidate_text, float(s)))
        scored.sort(key=lambda x: x[2])

        # Convert scores to a softmax over -score (relative only, not calibrated probability).
        min_s = scored[0][2]
        exps = [math.exp(-(s - min_s)) for _, _, s in scored]
        z = sum(exps) or 1.0
        soft = [e / z for e in exps]
        top_candidates = [
            {"fault_type": ft, "confidence": float(p)}
            for (ft, _, _), p in zip(scored[:5], soft[:5])
        ]
        best_fault, best_text, _ = scored[0]
        best_conf = float(soft[0])
        return best_text, best_conf, top_candidates

    def predict_fault(
        self,
        *,
        circuit_name: str,
        node_voltages: dict[str, float] | None = None,
        source_currents: dict[str, float] | None = None,
        measurement_overrides: dict[str, float] | None = None,
        temp: float | None = None,
        tnom: float | None = None,
        strict: bool = False,
    ) -> DebugResult:
        if circuit_name not in self.catalog:
            raise KeyError(f"Unknown circuit: {circuit_name}")
        spec = self.catalog[circuit_name]
        golden = dict(spec.get("golden_measurements_max") or {})

        measured, used_v_keys, used_i_keys = self._normalize_measurements_from_request(
            node_voltages=node_voltages,
            source_currents=source_currents,
            measurement_overrides=measurement_overrides,
            temp=temp,
            tnom=tnom,
            spec=spec,
        )

        required_nodes = [item["node_name"] for item in spec.get("nodes", [])]
        provided_nodes_norm = {str(k).upper() for k in (node_voltages or {}).keys()}
        missing_required_nodes = [n for n in required_nodes if n.upper() not in provided_nodes_norm]
        if strict and missing_required_nodes:
            raise ValueError(f"Missing required nodes for {circuit_name}: {', '.join(missing_required_nodes)}")

        input_text = self._build_input_text(circuit_name=circuit_name, measured=measured, golden=golden)
        mod = self._load_eval_module()
        response_style = str(self.config.get("response_style", "faulttype_diag_fix"))
        instruction = str(
            self.config.get(
                "instruction",
                "Classify the LTspice fault and provide a fix. Return exactly two lines: FaultType then Diagnosis/Fix.",
            )
        )

        pred_text: str
        confidence: float
        top_candidates: list[dict[str, float]]

        if bool(self.config.get("use_prerules", True)):
            pre_class = mod.prerule_fault_type(input_text)
        else:
            pre_class = None

        if pre_class:
            body = mod.canonical_completion_for_fault(pre_class)
            pred_text = f"FaultType: {pre_class}\n{body}" if response_style == "faulttype_diag_fix" else body
            confidence = 1.0
            top_candidates = [{"fault_type": pre_class, "confidence": 1.0}]
        else:
            prompt = mod.build_prompt(instruction, input_text, response_style)
            pred_text, confidence, top_candidates = self._score_class_candidates_with_knn(prompt, input_text)

        pred_text = mod.force_diag_fix_format(pred_text, response_style)
        fault_type = mod.classify_fault_text(pred_text)
        diagnosis, fix = _parse_diag_fix(pred_text)

        return DebugResult(
            circuit_name=circuit_name,
            fault_type=fault_type,
            confidence=confidence,
            diagnosis=diagnosis,
            fix=fix,
            provided_node_count=len(node_voltages or {}),
            required_node_count=len(required_nodes),
            missing_required_nodes=missing_required_nodes,
            used_voltage_measurement_keys=used_v_keys,
            used_current_measurement_keys=used_i_keys,
            top_candidates=top_candidates,
        )

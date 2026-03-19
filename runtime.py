from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PAIR_MC_PO = ("missing_component", "pin_open")

FAULT_TEMPLATES: dict[str, dict[str, str]] = {
    "param_drift": {
        "diagnosis": "parameter drift in one or more components",
        "fix": "restore drifted parameter values to their intended targets",
    },
    "missing_component": {
        "diagnosis": "missing component in the circuit path",
        "fix": "reinsert the missing component with the intended value/model",
    },
    "pin_open": {
        "diagnosis": "open connection on a component terminal",
        "fix": "reconnect the opened pin to its intended node",
    },
    "swapped_nodes": {
        "diagnosis": "swapped terminals on a component/source",
        "fix": "swap the two connections back to their intended nodes",
    },
    "short_between_nodes": {
        "diagnosis": "unintended short between nodes",
        "fix": "remove the short and restore proper wiring",
    },
    "resistor_value_swap": {
        "diagnosis": "resistor values were swapped between two resistors",
        "fix": "restore each resistor to its intended value",
    },
    "resistor_wrong_value": {
        "diagnosis": "wrong resistor value on one resistor",
        "fix": "change that resistor back to its intended value",
    },
    "unknown": {
        "diagnosis": "unknown fault class from provided measurements",
        "fix": "inspect wiring and component values",
    },
}


def safe_measure_name(token: str) -> str:
    # Match pipeline/generate_variants.py safe_measure_name()
    cleaned = re.sub(r"[^A-Za-z0-9_.$]", "_", token)
    if not cleaned:
        return "x"
    if cleaned[0].isdigit():
        return f"n_{cleaned}"
    return cleaned


def measurement_key_for_node(node_name: str) -> str:
    return f"v_{safe_measure_name(node_name)}_max".lower()


def measurement_key_for_vsource_current(source_name: str) -> str:
    return f"i_{safe_measure_name(source_name)}_max".lower()


def family_id(circuit_name: str) -> str:
    parts = circuit_name.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else circuit_name


def prefix_id(circuit_name: str) -> str:
    return circuit_name.split("_")[0] if "_" in circuit_name else circuit_name


def _strip_metric_key(key: str, prefix: str) -> str:
    low = key.lower()
    if not low.startswith(prefix) or not low.endswith("_max"):
        return key
    return low[len(prefix) : -len("_max")]


def best_effort_display_from_voltage_key(key: str) -> str:
    token = _strip_metric_key(key, "v_")
    if token.startswith("_"):
        token = "-" + token[1:]
    return token.upper()


def best_effort_display_from_current_key(key: str) -> str:
    token = _strip_metric_key(key, "i_")
    if token.startswith("_"):
        token = "-" + token[1:]
    return token.upper()


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isfinite(v):
            return v
    return None


def _agg_feature_block(out: dict[str, Any], prefix: str, values: list[float]) -> None:
    if not values:
        out[prefix + "count"] = 0.0
        return

    arr = np.array(values, dtype=float)
    aval = np.abs(arr)
    out[prefix + "count"] = float(arr.size)
    out[prefix + "sum"] = float(arr.sum())
    out[prefix + "mean"] = float(arr.mean())
    out[prefix + "std"] = float(arr.std())
    out[prefix + "abs_sum"] = float(aval.sum())
    out[prefix + "abs_mean"] = float(aval.mean())
    out[prefix + "abs_std"] = float(aval.std())
    out[prefix + "abs_max"] = float(aval.max())
    out[prefix + "abs_min"] = float(aval.min())

    q = np.quantile(aval, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    for name, val in zip(("q25", "q50", "q75", "q90", "q95", "q99"), q):
        out[prefix + name] = float(val)

    s = np.sort(aval)[::-1]
    for i in range(min(8, len(s))):
        out[f"{prefix}top{i + 1}"] = float(s[i])

    for thr in (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0):
        out[f"{prefix}abs_gt_{str(thr).replace('.', 'p')}"] = float((aval > thr).sum())

    out[prefix + "near_zero"] = float((aval < 1e-3).sum())
    out[prefix + "near_5rail"] = float((np.abs(aval - 5.0) < 0.25).sum())
    out[prefix + "near_3v3"] = float((np.abs(aval - 3.3) < 0.25).sum())
    out[prefix + "pos_count"] = float((arr > 1e-4).sum())
    out[prefix + "neg_count"] = float((arr < -1e-4).sum())


def build_feature_dict_from_measurements(
    circuit_name: str,
    measured: dict[str, Any],
    golden: dict[str, Any],
    *,
    sim_success: bool = True,
) -> dict[str, Any]:
    """
    Build the engineered feature dict used by the final tabular xgboost model.

    This mirrors the feature extractor used in the v2+v3+v4+v5 training/eval run.
    """
    f: dict[str, Any] = {
        "lab": circuit_name,
        "family": family_id(circuit_name),
        "prefix": prefix_id(circuit_name),
        "sim_success": 1.0 if sim_success else 0.0,
    }

    d: dict[str, float] = {}
    m: dict[str, float] = {}
    for k, v in (measured or {}).items():
        nv = _numeric(v)
        if nv is not None:
            m[str(k).lower()] = nv
    gnum: dict[str, float] = {}
    for k, v in (golden or {}).items():
        nv = _numeric(v)
        if nv is not None:
            gnum[str(k).lower()] = nv

    for k, mv in m.items():
        if k in gnum:
            d[k] = mv - gnum[k]

    f["n_deltas"] = float(len(d))
    f["n_measured"] = float(len(m))

    common_keys = set(d).intersection(m)
    for k, v in d.items():
        f["d:" + k] = float(v)
    for k, v in m.items():
        f["m:" + k] = float(v)
    for k in common_keys:
        dv = d[k]
        mv = m[k]
        gv = mv - dv  # reconstructs golden numeric value
        f["g:" + k] = float(gv)
        denom = abs(gv) + 1e-6
        f["adivg:" + k] = abs(dv) / denom
        f["sgnflip:" + k] = 1.0 if (mv != 0 and gv != 0 and mv * gv < 0) else 0.0
        f["same_sign:" + k] = 1.0 if (mv == 0 or gv == 0 or mv * gv > 0) else 0.0

    dvals = [v for v in d.values() if math.isfinite(v)]
    mvals = [v for v in m.values() if math.isfinite(v)]
    gvals = [m[k] - d[k] for k in common_keys if math.isfinite(m[k]) and math.isfinite(d[k])]

    _agg_feature_block(f, "d_all_", dvals)
    _agg_feature_block(f, "m_all_", mvals)
    _agg_feature_block(f, "g_all_", gvals)

    for pfx in ("v_", "i_"):
        dsub = [v for k, v in d.items() if k.startswith(pfx)]
        msub = [v for k, v in m.items() if k.startswith(pfx)]
        gsub = [m[k] - d[k] for k in common_keys if k.startswith(pfx)]
        _agg_feature_block(f, f"d_{pfx}", dsub)
        _agg_feature_block(f, f"m_{pfx}", msub)
        _agg_feature_block(f, f"g_{pfx}", gsub)
        if pfx == "v_":
            ratios = [abs(d[k]) / (abs(m[k] - d[k]) + 1e-6) for k in common_keys if k.startswith("v_")]
            _agg_feature_block(f, "rdelta_v_", ratios)

    keys = list(set(list(d.keys()) + list(m.keys())))
    f["key_count_v"] = float(sum(1 for k in keys if k.startswith("v_")))
    f["key_count_i"] = float(sum(1 for k in keys if k.startswith("i_")))
    f["key_count_supply"] = float(sum(1 for k in keys if ("vcc" in k.lower() or "vdd" in k.lower())))
    f["key_count_out"] = float(sum(1 for k in keys if ("out" in k.lower())))
    return f


@dataclass
class DebugResult:
    circuit_name: str
    fault_type: str
    confidence: float
    diagnosis: str
    fix: str
    provided_node_count: int
    required_node_count: int
    missing_required_nodes: list[str]
    used_voltage_measurement_keys: list[str]
    used_current_measurement_keys: list[str]
    top_candidates: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_name": self.circuit_name,
            "fault_type": self.fault_type,
            "confidence": self.confidence,
            "diagnosis": self.diagnosis,
            "fix": self.fix,
            "response_text": (
                f"FaultType: {self.fault_type}\n"
                f"Diagnosis: {self.diagnosis}. Fix: {self.fix}."
            ),
            "provided_node_count": self.provided_node_count,
            "required_node_count": self.required_node_count,
            "missing_required_nodes": self.missing_required_nodes,
            "used_voltage_measurement_keys": self.used_voltage_measurement_keys,
            "used_current_measurement_keys": self.used_current_measurement_keys,
            "top_candidates": self.top_candidates,
        }


class CircuitDebugRuntime:
    def __init__(
        self,
        model_bundle_path: str | Path,
        circuit_catalog_path: str | Path,
        *,
        family_pair_models_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.model_bundle_path = Path(model_bundle_path)
        self.circuit_catalog_path = Path(circuit_catalog_path)
        self.family_pair_models_path = Path(family_pair_models_path) if family_pair_models_path else None
        self.config_path = Path(config_path) if config_path else None

        bundle = joblib.load(self.model_bundle_path)
        self.vectorizer = bundle["vectorizer"]
        self.global_model = bundle["global_model"]
        self.pair_model = bundle["pair_model"]
        self.report = dict(bundle.get("report") or {})

        catalog_doc = json.loads(self.circuit_catalog_path.read_text(encoding="utf-8"))
        self.catalog = catalog_doc["circuits"]
        self.catalog_meta = {k: v for k, v in catalog_doc.items() if k != "circuits"}

        self.config: dict[str, Any] = {}
        if self.config_path and self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.family_pair_models: dict[str, Any] = {}
        if self.family_pair_models_path and self.family_pair_models_path.exists():
            loaded = joblib.load(self.family_pair_models_path)
            if isinstance(loaded, dict):
                self.family_pair_models = loaded

        self.pair_threshold = float(
            self.config.get("pair_threshold", self.report.get("pair_threshold", 0.5))
        )
        raw_classes = list(self.global_model.classes_)
        config_class_names = list(self.config.get("class_names_sorted", []) or [])
        if raw_classes and all(isinstance(x, (int, np.integer)) for x in raw_classes) and config_class_names:
            # XGBoost stores integer-encoded classes in the model; use saved index->label mapping from runtime config.
            self.class_names = [str(config_class_names[int(i)]) for i in raw_classes]
        else:
            self.class_names = [str(x) for x in raw_classes]

    def list_circuits(self) -> list[str]:
        return sorted(self.catalog.keys())

    def has_circuit(self, circuit_name: str) -> bool:
        return circuit_name in self.catalog

    def circuit_spec(self, circuit_name: str) -> dict[str, Any]:
        return dict(self.catalog[circuit_name])

    def _normalize_measurements_from_request(
        self,
        node_voltages: dict[str, float] | None,
        source_currents: dict[str, float] | None,
        measurement_overrides: dict[str, float] | None,
        temp: float | None,
        tnom: float | None,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        measured: dict[str, float] = {}
        used_v_keys: list[str] = []
        used_i_keys: list[str] = []

        if node_voltages:
            for node_name, value in node_voltages.items():
                key = measurement_key_for_node(str(node_name))
                nv = _numeric(value)
                if nv is None:
                    continue
                measured[key] = nv
                used_v_keys.append(key)

        if source_currents:
            for source_name, value in source_currents.items():
                key = measurement_key_for_vsource_current(str(source_name))
                nv = _numeric(value)
                if nv is None:
                    continue
                measured[key] = nv
                used_i_keys.append(key)

        if measurement_overrides:
            for key, value in measurement_overrides.items():
                nv = _numeric(value)
                if nv is None:
                    continue
                measured[str(key).lower()] = nv

        measured["temp"] = float(temp) if temp is not None else 27.0
        measured["tnom"] = float(tnom) if tnom is not None else 27.0
        return measured, sorted(set(used_v_keys)), sorted(set(used_i_keys))

    def predict_fault(
        self,
        *,
        circuit_name: str,
        node_voltages: dict[str, float] | None = None,
        source_currents: dict[str, float] | None = None,
        measurement_overrides: dict[str, float] | None = None,
        temp: float | None = None,
        tnom: float | None = None,
        strict: bool = True,
    ) -> DebugResult:
        if circuit_name not in self.catalog:
            raise KeyError(f"Unknown circuit: {circuit_name}")

        spec = self.catalog[circuit_name]
        golden_max = dict(spec.get("golden_measurements_max") or {})
        measured, used_v_keys, used_i_keys = self._normalize_measurements_from_request(
            node_voltages=node_voltages,
            source_currents=source_currents,
            measurement_overrides=measurement_overrides,
            temp=temp,
            tnom=tnom,
        )

        required_nodes = [item["node_name"] for item in spec.get("nodes", [])]
        provided_nodes_norm = {str(k).upper() for k in (node_voltages or {}).keys()}
        missing_required_nodes = [n for n in required_nodes if n.upper() not in provided_nodes_norm]
        if strict and missing_required_nodes:
            raise ValueError(
                f"Missing required nodes for {circuit_name}: {', '.join(missing_required_nodes)}"
            )

        feats = build_feature_dict_from_measurements(
            circuit_name=circuit_name,
            measured=measured,
            golden=golden_max,
            sim_success=True,
        )
        X = self.vectorizer.transform([feats])
        proba = np.asarray(self.global_model.predict_proba(X))[0]

        idx_sorted = np.argsort(proba)[::-1]
        base_idx = int(idx_sorted[0])
        base_pred = str(self.class_names[base_idx])
        final_pred = base_pred
        final_conf = float(proba[base_idx])

        if base_pred in PAIR_MC_PO:
            pair_prob = float(np.asarray(self.pair_model.predict_proba(X))[0, 1])  # P(pin_open)
            fam = family_id(circuit_name)
            fam_model = self.family_pair_models.get(fam)
            if fam_model is not None:
                pair_prob = float(np.asarray(fam_model.predict_proba(X))[0, 1])
            final_pred = "pin_open" if pair_prob >= self.pair_threshold else "missing_component"
            final_conf = pair_prob if final_pred == "pin_open" else (1.0 - pair_prob)

        templ = FAULT_TEMPLATES.get(final_pred, FAULT_TEMPLATES["unknown"])
        top_candidates = [
            {"fault_type": str(self.class_names[int(i)]), "confidence": float(proba[int(i)])}
            for i in idx_sorted[: min(5, len(idx_sorted))]
        ]

        return DebugResult(
            circuit_name=circuit_name,
            fault_type=final_pred,
            confidence=final_conf,
            diagnosis=templ["diagnosis"],
            fix=templ["fix"],
            provided_node_count=len(node_voltages or {}),
            required_node_count=len(required_nodes),
            missing_required_nodes=missing_required_nodes,
            used_voltage_measurement_keys=used_v_keys,
            used_current_measurement_keys=used_i_keys,
            top_candidates=top_candidates,
        )


def build_circuit_catalog(golden_root: str | Path) -> dict[str, Any]:
    root = Path(golden_root)
    circuits: dict[str, Any] = {}

    for lab_dir in sorted(root.iterdir()):
        if not lab_dir.is_dir():
            continue
        if lab_dir.name.startswith("merged_"):
            continue
        golden_path = lab_dir / "golden" / "golden_measurements.json"
        if not golden_path.exists():
            continue

        measurements = json.loads(golden_path.read_text(encoding="utf-8"))
        golden_max: dict[str, float] = {}
        nodes: list[dict[str, Any]] = []
        source_currents: list[dict[str, Any]] = []

        for key, value in measurements.items():
            low = str(key).lower()
            nv = _numeric(value)
            if low in {"temp", "tnom"} and nv is not None:
                golden_max[low] = nv
            if nv is None:
                continue
            if low.startswith("v_") and low.endswith("_max"):
                golden_max[low] = nv
                nodes.append(
                    {
                        "node_name": best_effort_display_from_voltage_key(low),
                        "measurement_key": low,
                        "golden_value": nv,
                    }
                )
            elif low.startswith("i_") and low.endswith("_max"):
                golden_max[low] = nv
                source_currents.append(
                    {
                        "source_name": best_effort_display_from_current_key(low),
                        "measurement_key": low,
                        "golden_value": nv,
                    }
                )

        circuits[lab_dir.name] = {
            "circuit_name": lab_dir.name,
            "nodes": sorted(nodes, key=lambda x: x["measurement_key"]),
            "source_currents": sorted(source_currents, key=lambda x: x["measurement_key"]),
            "golden_measurements_max": golden_max,
            "golden_defaults": {
                "solver": measurements.get("solver", "Normal"),
                "method": measurements.get("method", "trap"),
                "temp": measurements.get("temp", 27.0),
                "tnom": measurements.get("tnom", 27.0),
            },
            "paths": {
                "golden_measurements": str(golden_path).replace("/", "\\"),
                "lab_dir": str(lab_dir).replace("/", "\\"),
            },
        }

    return {
        "catalog_version": 1,
        "golden_root": str(root).replace("/", "\\"),
        "circuit_count": len(circuits),
        "circuits": circuits,
    }

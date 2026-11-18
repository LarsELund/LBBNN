#!/usr/bin/env python3
"""
inst/scripts/run_native_pyoperon.py
Batch runner for genuine C++ PyOperon (`pyoperon.sklearn.SymbolicRegressor`)
paired with SymPy canonical expansion to extract additive basis terms |A|,
active variable support, and held-out test predictions.
"""

import argparse
import json
import sys
import warnings

import numpy as np
import sympy as sp
from pyoperon.sklearn import SymbolicRegressor

warnings.filterwarnings("ignore")


def analyze_sympy_expr(expr_str: str, col_names: list[str]) -> tuple[int, list[str]]:
    if not expr_str or expr_str.strip() in ("0", "0.0", "1", "nan", "inf"):
        return 0, []
    try:
        cleaned = expr_str.replace("^", "**")
        local_dict = {nm: sp.Symbol(nm, real=True) for nm in col_names}
        expr = sp.sympify(cleaned, locals=local_dict)
        used_vars = [nm for nm in col_names if local_dict[nm] in expr.free_symbols]
        expanded = sp.expand(expr)
        args = sp.Add.make_args(expanded)
        non_const = []
        for term in args:
            if term.free_symbols:
                coeff, rest = term.as_coeff_Mul()
                try:
                    if abs(complex(coeff)) < 1e-10:
                        continue
                except Exception:
                    pass
                non_const.append(rest)
        return max(len(non_const), 1 if used_vars else 0), used_vars
    except Exception:
        used_vars = [nm for nm in col_names if nm in expr_str]
        return max(1, len(used_vars)), used_vars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tasks = payload["tasks"]
    results = []
    for idx, t in enumerate(tasks):
        X_tr = np.asarray(t["X_train"], dtype=float)
        y_tr = np.asarray(t["y_train"], dtype=float)
        X_te = np.asarray(t["X_test"], dtype=float)
        raw_cols = t.get("col_names") or []
        col_names = [str(c) for c in raw_cols] if len(raw_cols) == X_tr.shape[1] else [f"x{i+1}" for i in range(X_tr.shape[1])]
        seed = int(t.get("seed", 1))
        max_len = int(t.get("max_length", 25))
        max_dep = int(t.get("max_depth", 4))
        gens = int(t.get("generations", 80))
        pop = int(t.get("population_size", 300))
        n_threads = int(t.get("n_threads", 8))

        allowed_syms = t.get("allowed_symbols", "add,sub,mul,div,sin,cos,exp,square,constant,variable")
        sym_parts = [s.strip() for s in allowed_syms.split(",") if s.strip()]
        if "variable" not in sym_parts:
            sym_parts.append("variable")
        allowed_syms = ",".join(sym_parts)

        reg = SymbolicRegressor(
            allowed_symbols=allowed_syms,
            optimizer="lm",
            optimizer_iterations=10,
            max_length=max_len,
            max_depth=max_dep,
            generations=gens,
            population_size=pop,
            model_selection_criterion="minimum_description_length",
            n_threads=n_threads,
            random_state=seed,
        )
        reg.fit(X_tr, y_tr)
        expr_str = reg.get_model_string(reg.model_, precision=4, names=col_names)
        pred_te = np.asarray(reg.predict(X_te), dtype=float)
        if not np.all(np.isfinite(pred_te)):
            pred_te = np.where(np.isfinite(pred_te), pred_te, float(np.mean(y_tr)))
        n_terms, used_vars = analyze_sympy_expr(expr_str, col_names)
        results.append({
            "task_id": t.get("task_id", idx),
            "seed": seed,
            "eq": int(t.get("eq", 1)),
            "formula": expr_str,
            "n_terms": int(n_terms),
            "used_vars": used_vars,
            "pred_test": pred_te.tolist(),
        })
        if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
            print(f"[pyoperon] completed {idx + 1}/{len(tasks)}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()

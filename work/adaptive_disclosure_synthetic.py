"""CPU-only frozen synthetic mechanism diagnostic; numerical proposals, not certificates.

Run --freeze before --smoke or --run. No private state enters observer_design or
observer_decode. All prescribed failures and duplicate sampled IDs are retained.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import pathlib
import platform
import time
import numpy as np
import scipy
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=__import__("runtime_paths").ROOT
PROTOCOL = ROOT / "outputs/adaptive_disclosure_synthetic_protocol.json"
RESULT = ROOT / "outputs/adaptive_disclosure_synthetic.json"
ARMS = ("adaptive_anchor_leverage", "static_uniform", "static_bias_anchor_leverage")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def scores(W, bias, h):
    z = W @ h + bias
    return z - logsumexp(z)


def quantize(values, grid):
    # Exact specified numerical API: NumPy nearest-even, zero-phase float64 grid.
    return grid * np.rint(values / grid)


def kl_pair(lp, lq):
    p, q = np.exp(lp), np.exp(lq)
    return {"forward": float(p @ (lp - lq)), "reverse": float(q @ (lq - lp))}


def fisher(W, lp):
    p = np.exp(lp)
    mu = p @ W
    X = W - mu
    return p, mu, X.T @ (p[:, None] * X)


def public_basis(W):
    contrasts = W - W[0]
    rank = W.shape[1]
    _, _, pivots = qr(contrasts[1:].T, pivoting=True, mode="economic")
    ids = (pivots[:rank] + 1).astype(int)
    swaps = []
    for iteration in range(2001):
        B = contrasts[ids]
        alpha = np.linalg.solve(B.T, contrasts.T).T
        y, col = np.unravel_index(np.argmax(np.abs(alpha)), alpha.shape)
        maximum = float(np.max(np.abs(alpha)))
        if maximum <= 1.05:
            break
        if iteration == 2000:
            raise RuntimeError("maxvol proposal reached frozen 2000-swap cap")
        swaps.append({"old_id": int(ids[col]), "new_id": int(y), "basis_slot": int(col),
                      "coefficient": float(alpha[y, col])})
        ids[col] = y
    return B, ids, {"reference_id": 0, "basis_ids": ids.tolist(),
                    "maximum_coefficient_numerical": maximum, "swap_ledger": swaps,
                    "minimum_singular_value": float(np.linalg.svd(B, compute_uv=False)[-1]),
                    "scope": "float64 QR/maxvol proposal; no exact-rational certification"}


def freeze_protocol():
    heads, cases = [], []
    for rank in (2, 4, 8):
        for family in ("gaussian", "anisotropic_bias"):
            for seed in (101, 202):
                rng = np.random.default_rng(seed + 1000 * rank)
                W = rng.normal(size=(64, rank)) / np.sqrt(rank)
                bias = rng.normal(0, .2, size=64)
                if family == "anisotropic_bias":
                    W = W * np.geomspace(.35, 2.5, rank)
                    bias = rng.normal(0, 1.0, size=64)
                hid = f"r{rank}_{family}_seed{seed}"
                heads.append({"id": hid, "rank": rank, "V": 64, "family": family,
                              "seed": seed, "W": W.tolist(), "bias": bias.tolist()})
                for norm in (0.0, 1.0, 4.0):
                    for state_seed in (11, 22):
                        srng = np.random.default_rng(state_seed + 100 * rank)
                        h = srng.normal(size=rank)
                        h = norm * h / np.linalg.norm(h)
                        for g1, g2 in ((.02, .002), (.2, .02)):
                            idx = len(cases)
                            cases.append({"case_id": idx, "head_id": hid, "state_norm": norm,
                                          "state_seed": state_seed, "h": h.tolist(),
                                          "g1": g1, "g2": g2, "draw_seed": 900000 + idx})
    ledger = {
        "version": 1, "frozen_before_metrics": True, "script_sha256": digest(pathlib.Path(__file__)),
        "experiment_scope": "CPU-only float64 synthetic mechanism diagnostic; no trained models",
        "not_claimed": ["algorithm novelty", "rigorous numerical certificates", "equal total encoding bits",
                        "real-model held-out performance", "optimal adaptive disclosure"],
        "arithmetic": {"numpy": np.__version__, "scipy": scipy.__version__,
                       "quantizer": "zero-phase nearest-even np.rint(values/grid)*grid; float64",
                       "device": "CPU", "BLAS_threads": 1},
        "protocol": {
            "first_stage": "reference ID0, pivoted-QR contrast basis, maxvol swaps until maxabs<=1.05",
            "stage1_swap_cap": 2000, "tau_for_bound": 1.1,
            "stage2_draw_count": "4*(r+1) WITH replacement for every arm; all draws charged",
            "arms": list(ARMS),
            "adaptive_proposal": "p0_i*(1+(Wi-mu0)^T F0^-1 (Wi-mu0))/(r+1)",
            "adaptive_raw_draw_weights": "p0_i/(m*q_i)",
            "static_uniform": "uniform proposal; each draw raw weight1/m",
            "static_bias_anchor_leverage": "same augmented leverage at public h=0; raw pstatic_i/(m*qstatic_i)",
            "draws": "common fixed seed per case across arms; shared uniforms with arm-specific categorical CDF",
            "weight_aggregation": "sum repeated draw weights by public ID, then normalize",
            "covariance_check": "numerical eigenvalue of F0-whitened selected covariance; c>1e-12",
            "conditioning_cap": 1e12,
            "decoder": "weighted LS with free normalizer intercept, then identical stage1 boxD metric projection",
            "projection": "z=B(h-h0), box[-g1,g1]; SciPy L-BFGS-B with analytic Jacobian",
            "solver_options": {"ftol": 1e-15, "gtol": 1e-13, "maxiter": 2000, "maxls": 100},
            "failure_policy": "retain all arms/cases; singular designs and failed solves have no diagnostic certificate",
            "score_cost": "each arm charges r+1 coarse scores PLUS4(r+1) fine draws; unique IDs reported separately",
            "truth_access": "h used only by simulator and post-decode metrics, never observer design or decoding"
        },
        "heads": heads, "cases": cases
    }
    write_json(PROTOCOL, ledger)
    print(json.dumps({"frozen": str(PROTOCOL), "heads": len(heads), "cases": len(cases),
                      "protocol_sha256": digest(PROTOCOL)}))


def observer_design(W, bias, h0, F0, kind, draw_seed, count):
    # No truth h, true scores, true probabilities or truth Fisher are arguments.
    anchor_lp = scores(W, bias, h0)
    if kind == "adaptive_anchor_leverage":
        target, mu, F = fisher(W, anchor_lp)
    elif kind == "static_bias_anchor_leverage":
        target, mu, F = fisher(W, scores(W, bias, np.zeros(W.shape[1])))
    else:
        target = np.full(W.shape[0], 1 / W.shape[0])
        mu, F = None, None
    if kind != "static_uniform":
        X = W - mu
        lev = np.einsum("ij,ji->i", X, np.linalg.solve(F, X.T))
        proposal = target * (1 + lev) / (W.shape[1] + 1)
        proposal = proposal / proposal.sum()
    else:
        proposal = target.copy()
    rng = np.random.default_rng(draw_seed)
    uniforms = rng.random(count)
    draws = np.searchsorted(np.cumsum(proposal), uniforms, side="right")
    draws = np.minimum(draws, W.shape[0] - 1)
    ids, multiplicity = np.unique(draws, return_counts=True)
    raw = multiplicity * target[ids] / (count * proposal[ids])
    omega = raw / raw.sum()
    WS = W[ids]
    muS = omega @ WS
    X = WS - muS
    G = X.T @ (omega[:, None] * X)
    ef, Uf = np.linalg.eigh(F0)
    if float(ef[0]) <= 0:
        raise RuntimeError("nonpositive anchor Fisher in native arithmetic")
    invroot = (Uf * (1 / np.sqrt(ef))) @ Uf.T
    whitened = invroot @ G @ invroot
    whitened = (whitened + whitened.T) / 2
    eigen = np.linalg.eigvalsh(whitened)
    return ids, omega, G, {
        "kind": kind, "seed": draw_seed, "draw_count": count, "draw_ids": draws.tolist(),
        "selected_unique_ids": ids.tolist(), "multiplicity": multiplicity.tolist(),
        "raw_aggregate_weights": raw.tolist(), "normalized_weights": omega.tolist(),
        "proposal_probabilities": proposal.tolist(), "target_probabilities": target.tolist(),
        "selected_covariance": G.tolist(), "whitened_eigenvalues": eigen.tolist(),
        "c_numerical_proposal": float(eigen[0]), "G_condition": float(np.linalg.cond(G)),
        "F0_condition": float(np.linalg.cond(F0)),
        "certificate_scope": "native float64 covariance proposal; NOT a rigorous Loewner certificate"
    }


def observer_decode(W, bias, h0, B, g1, ids, omega, G, received):
    # Only public head, observed messages and public proposed design enter here.
    WS = W[ids]
    X = WS - omega @ WS
    response = received - bias[ids]
    hLS = np.linalg.solve(G, X.T @ (omega * response))
    intercept = float(omega @ (response - WS @ hLS))
    residual = response - WS @ hLS - intercept
    Binv = np.linalg.inv(B)
    zLS = B @ (hLS - h0)
    H = Binv.T @ G @ Binv
    H = (H + H.T) / 2
    scale = float(np.linalg.norm(H, 2))
    if scale <= 0:
        raise RuntimeError("projection metric nonpositive")
    Hscaled = H / scale
    def objective(z):
        v = z - zLS
        return .5 * float(v @ Hscaled @ v), Hscaled @ v
    opt = minimize(objective, np.clip(zLS, -g1, g1), jac=True, method="L-BFGS-B",
                   bounds=[(-g1, g1)] * len(h0),
                   options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 2000, "maxls": 100})
    z = opt.x
    hproj = h0 + Binv @ z
    grad = Hscaled @ (z - zLS)
    pgrad = z - np.clip(z - grad, -g1, g1)
    return hproj, hLS, {
        "hLS": hLS.tolist(), "h_projected": hproj.tolist(), "intercept": intercept,
        "weighted_LS_residual_variance": float(omega @ residual ** 2),
        "zLS": zLS.tolist(), "z_projected": z.tolist(), "solver_success": bool(opt.success),
        "solver_status": int(opt.status), "solver_message": str(opt.message),
        "iterations": int(opt.nit), "projection_objective": float(opt.fun * scale),
        "projection_metric_scale": scale, "projected_gradient_inf_scaled": float(np.max(np.abs(pgrad))),
        "projected_gradient_inf_raw": float(np.max(np.abs(pgrad)) * scale),
        "box_violation": float(max(0, np.max(np.abs(z)) - g1)),
        "solver_scope": "numerical L-BFGS-B output, not a verified exact metric projection"
    }


def run_cases(smoke=False):
    ledger = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if ledger["script_sha256"] != digest(pathlib.Path(__file__)):
        raise RuntimeError("source changed since freeze; refreeze and preserve any earlier pilot outputs")
    heads = {head["id"]: head for head in ledger["heads"]}
    cases = ledger["cases"][:1] if smoke else ledger["cases"]
    all_rows, bases = [], {}
    start = time.perf_counter()
    for case in cases:
        begin = time.perf_counter()
        head = heads[case["head_id"]]
        W, bias, trueh = np.array(head["W"]), np.array(head["bias"]), np.array(case["h"])
        r, g1, g2 = head["rank"], case["g1"], case["g2"]
        row = {**case, "rank": r, "V": len(W), "family": head["family"], "arms": []}
        try:
            if case["head_id"] not in bases:
                bases[case["head_id"]] = public_basis(W)
            B, basisids, basisinfo = bases[case["head_id"]]
            row["stage1_basis"] = basisinfo
            if basisinfo["maximum_coefficient_numerical"] > 1.1:
                raise RuntimeError("frozen tau1.1 proposal failed")
            firstids = np.r_[0, basisids]
            true_lp = scores(W, bias, trueh)
            first_received = quantize(true_lp[firstids], g1)
            h0 = np.linalg.solve(B, first_received[1:] - first_received[0] - bias[basisids] + bias[0])
            anchor_lp = scores(W, bias, h0)
            p0, mu0, F0 = fisher(W, anchor_lp)
            row["stage1"] = {
                "selected_ids": firstids.tolist(), "received_scores": first_received.tolist(),
                "h0_from_received_only": h0.tolist(), "anchor_probabilities": p0.tolist(),
                "anchor_Fisher": F0.tolist(), "coarse_decoder_KL": kl_pair(true_lp, anchor_lp),
                "coarse_bound_proposal": r*r*1.1*1.1*g1*g1/2,
                "truth_box_coordinate": (B @ (trueh-h0)).tolist(),
                "truth_feasibility_numerical": bool(np.max(np.abs(B @ (trueh-h0))) <= g1 + 1e-12),
                "score_error_max": float(np.max(np.abs(first_received - true_lp[firstids]))),
                "truth_scope": "post-hoc simulator check; not used by observer"
            }
            for arm in ARMS:
                armstart = time.perf_counter()
                out = {"arm": arm, "status": "failed", "coarse_score_count": r+1,
                       "fine_draw_count_charged": 4*(r+1), "total_score_count_charged": 5*(r+1)}
                try:
                    ids, omega, G, design = observer_design(W,bias,h0,F0,arm,case["draw_seed"],4*(r+1))
                    out["design"] = design
                    out["fine_unique_count"] = len(ids)
                    out["total_distinct_token_ids"] = len(np.unique(np.r_[firstids,ids]))
                    if design["c_numerical_proposal"] <= 1e-12 or design["G_condition"] > 1e12:
                        raise RuntimeError("frozen numerical covariance threshold/conditioning failed")
                    received = quantize(true_lp[ids],g2)
                    out["fine_received_scores"] = received.tolist()
                    hp, hLS, decoding = observer_decode(W,bias,h0,B,g1,ids,omega,G,received)
                    out["decoding"] = decoding
                    result_kl = kl_pair(true_lp,scores(W,bias,hp))
                    c = design["c_numerical_proposal"]
                    L = float(np.exp(2*r*1.1*g1))
                    bound = L*g2*g2/(8*c)
                    noise = received-true_lp[ids]
                    dls, dp = hLS-trueh, hp-trueh
                    out["metrics"] = {
                        "KL": result_kl, "L_global_proposal": L, "bound_proposal": bound,
                        "bound_satisfied_numerically": max(result_kl.values()) <= bound*(1+1e-8)+1e-13,
                        "fine_noise_variance": float(omega@(noise-omega@noise)**2),
                        "hLS_truth_G_error": float(dls@G@dls),
                        "projected_truth_G_error": float(dp@G@dp),
                        "noise_variance_limit": g2*g2/4,
                        "projection_contraction_numerical": float(dp@G@dp) <= float(dls@G@dls)+1e-13,
                        "truth_checks_scope": "post-hoc numerical checks, never inputs to observer",
                        "scope": "float64 diagnostic; no interval/Loewner/projection certificate"
                    }
                    if not decoding["solver_success"]:
                        raise RuntimeError("projection solver reported failure; output retained diagnostically")
                    out["status"] = "completed_diagnostic"
                except Exception as error:
                    out["failure"] = f"{type(error).__name__}: {error}"
                out["seconds"] = time.perf_counter()-armstart
                row["arms"].append(out)
            row["status"] = "completed_case" if all(a["status"]=="completed_diagnostic" for a in row["arms"]) else "case_with_retained_failure"
        except Exception as error:
            row["status"] = "stage1_failure"
            row["failure"] = f"{type(error).__name__}: {error}"
        row["seconds"] = time.perf_counter()-begin
        all_rows.append(row)
        if len(all_rows)%24==0:
            print(json.dumps({"cases_finished":len(all_rows),"cases_total":len(cases)}),flush=True)
    arm_summary = {}
    for name in ARMS:
        arms = [arm for row in all_rows for arm in row["arms"] if arm["arm"]==name]
        valid = [arm for arm in arms if arm["status"]=="completed_diagnostic"]
        arm_summary[name] = {
            "prescribed_case_denominator":len(cases), "attempted":len(arms),"completed":len(valid),
            "retained_failures":len(cases)-len(valid),
            "median_forward_among_completed":float(np.median([a["metrics"]["KL"]["forward"] for a in valid])) if valid else None,
            "median_reverse_among_completed":float(np.median([a["metrics"]["KL"]["reverse"] for a in valid])) if valid else None,
            "median_c_among_completed":float(np.median([a["design"]["c_numerical_proposal"] for a in valid])) if valid else None,
            "numeric_bound_violations":sum(not a["metrics"]["bound_satisfied_numerically"] for a in valid),
            "projection_contraction_violations":sum(not a["metrics"]["projection_contraction_numerical"] for a in valid),
            "medians_scope":"completed diagnostics only; failures explicitly counted, no zero-imputation of decoder errors"
        }
    payload = {
        "metadata": {"status":"SMOKE diagnostic excluded from formal synthetic denominator" if smoke else "frozen CPU synthetic mechanism diagnostic",
                     "script_sha256":digest(pathlib.Path(__file__)),"protocol_sha256":digest(PROTOCOL),
                     "protocol_path":str(PROTOCOL),"numpy":np.__version__,"scipy":scipy.__version__,
                     "python":platform.python_version(),"device":"CPU","case_count":len(cases),
                     "scope":"numerical covariance/projection proposals only; not rigorous certificates or pretrained performance",
                     "matched_cost":"all three arms reuse identical coarse stage1 and D; second draw counts and fine grids matched",
                     "not_claimed":"same total bits, optimality, algorithm novelty, held-out real LM performance",
                     "elapsed_seconds":time.perf_counter()-start},
        "summary":arm_summary,"rows":all_rows
    }
    path = ROOT/"work/adaptive_disclosure_synthetic_smoke.json" if smoke else RESULT
    write_json(path,payload)
    print(json.dumps({"saved":str(path),"cases":len(cases),"summary":arm_summary}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("--freeze","--smoke","--run"))
    # Use plain subcommands too, because argparse treats --freeze as an option.
    args = None
    import sys
    if len(sys.argv)!=2 or sys.argv[1] not in ("--freeze","--smoke","--run"):
        raise SystemExit("usage: adaptive_disclosure_synthetic.py --freeze|--smoke|--run")
    if sys.argv[1]=="--freeze":
        freeze_protocol()
    else:
        run_cases(smoke=sys.argv[1]=="--smoke")

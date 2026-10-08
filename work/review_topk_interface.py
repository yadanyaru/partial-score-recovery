"""Frozen strict top-k score interface diagnostic on the new LAMBADA corpus.

Only the teacher's top-k token IDs and rounded calibrated scores enter either
receiver. Decoder matrices depend on the public head and returned IDs. Teacher
probabilities enter metrics only. No returned tail mass or normalizer is used.
"""
import os
for name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "4"
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as sla
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT=__import__("runtime_paths").ROOT
PROTOCOL = ROOT / "outputs/REVIEW_TOPK_PROTOCOL.json"
STATES = ROOT / "outputs/review_native_states"
SLUGS = ["EleutherAI__pythia-70m", "openai-community__gpt2", "Qwen__Qwen2.5-0.5B"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def freeze():
    models = []
    for slug in SLUGS:
        path = STATES / (slug + ".json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["metadata"]["completed"] and len(data["rows"]) == 128
        models.append(dict(slug=slug, state_sha256=sha(path), **data["metadata"]))
    spec = dict(source_sha256=sha(__file__), corpus_sha256=sha(ROOT / "work/review_lambada_corpus.json"),
                models=models, paragraphs=list(range(128)), budgets=[32, 128, "r+101"],
                meshes=[0.01, 0.5, 2.0], zero_phase=True,
                provider="Only exact teacher top-k IDs and calibrated log-probabilities rounded by d*floor(lp/d+0.5)",
                tie_break="Stable descending exact log-probability; vocabulary ID ascending for exact ties",
                teacher="Native frozen FP32 state and native public affine head; float64 score evaluation",
                decoder_ls="A=[W_selected,-1], y=z-bias_selected; reconstruct h from intercept-aware LS and public full softmax",
                decoder_undercomplete="For k<r+1: v=A.T@(A@A.T+lambda*I)^(-1)y, lambda=1e-6*trace(A@A.T)/k; fixed public ridge scaling",
                decoder_overcomplete="For k>=r+1: unregularized Cholesky of A.T@A; failures retained as failed cells with no fallback",
                decoder_tail="For returned scores z: selected masses exp(z-d/2), residual mass uniformly over omitted tokens; normalize for floating arithmetic",
                decoder_access="Public W,bias,V and returned packet only; neither teacher hidden state, probabilities, normalizer nor gold IDs are receiver inputs",
                diagnostics="Cholesky diagonal ratio for every LS system; exact SVD rank and 2-norm condition for every16th paragraph at each budget",
                metrics=["forward_KL", "reverse_KL", "max_KL", "absolute_error_teacher_top20_complement_mass", "gold_first_token_NLL", "gold_first_token_accuracy", "teacher_argmax_agreement"],
                metric_scope="Initial answer-prefix distribution; first gold answer token, not full word accuracy or answer continuation likelihood",
                planned_configuration_cells=128*3*3*3, receivers_per_cell=2,
                fixed_cpu_threads=4, frozen_before_metrics=True,
                smoke="Optional first2 passages; excluded separate output directory, no protocol tuning")
    if PROTOCOL.exists():
        assert json.loads(PROTOCOL.read_text()) == spec, "Frozen protocol differs"
    else:
        save(PROTOCOL, spec)
    print(str(PROTOCOL), flush=True)


def metrics(lp, p, lq, top20, gold, teacher_argmax):
    q = np.exp(lq)
    forward = float(p @ (lp-lq))
    reverse = float(q @ (lq-lp))
    predicted = int(np.argmax(lq))
    return dict(forward_KL=forward, reverse_KL=reverse, max_KL=max(forward, reverse),
                absolute_error_teacher_top20_complement_mass=float(abs(q[top20].sum()-p[top20].sum())),
                gold_first_token_NLL=float(-lq[gold]), gold_first_token_accuracy=int(predicted==gold),
                teacher_argmax_agreement=int(predicted==teacher_argmax), predicted_top1_id=predicted)


def run(slug, smoke=False):
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert spec["source_sha256"] == sha(__file__)
    model_spec = next(item for item in spec["models"] if item["slug"]==slug)
    state_path = STATES / (slug+".json")
    assert sha(state_path)==model_spec["state_sha256"]
    data=json.loads(state_path.read_text(encoding="utf-8"))
    head_path=Path(model_spec["head_cache"])
    assert sha(head_path)==model_spec["head_cache_sha256"]
    started=time.perf_counter()
    with np.load(head_path) as head:
        W=np.asarray(head["W"],dtype=np.float64)
        bias=np.asarray(head["bias"],dtype=np.float64)
    V,r=W.shape
    rows=[]; diagnostics=[]; failures=[]
    samples=data["rows"][:2] if smoke else data["rows"]
    head_loaded=time.perf_counter()-started
    for j,item in enumerate(samples):
        h=np.asarray(item["hidden_state"],dtype=np.float64)
        gold=int(item["gold_ids"][0])
        logits=W@h+bias
        lp=logits-logsumexp(logits)
        p=np.exp(lp)
        maxk=r+101
        candidate=np.argpartition(-lp,maxk-1)[:maxk]
        # Explicit stable ID ordering among equal teacher scores.
        order=np.lexsort((candidate,-lp[candidate]))
        top=candidate[order]
        top20=top[:20]
        teacher_argmax=int(top[0])
        teacher=dict(gold_first_token_NLL=float(-lp[gold]), gold_first_token_accuracy=int(gold==teacher_argmax),
                     top1_id=teacher_argmax, top20_complement_mass=float(1-p[top20].sum()))
        solutions=[]; planned=[]
        for k in [32,128,maxk]:
            ids=top[:k]
            A=np.column_stack((W[ids],-np.ones(k)))
            under=k<r+1
            gram=A@A.T if under else A.T@A
            ridge=1e-6*float(np.trace(gram))/k if under else 0.
            diag=dict(paragraph=item["paragraph"],source_row=item["source_row"],k=k,r=r,V=V,
                      K_over_V=k/V,system_size=len(gram),undercomplete=under,ridge_lambda=ridge)
            if j%16==0:
                sv=sla.svdvals(A,check_finite=False)
                tol=max(A.shape)*np.finfo(float).eps*float(sv[0])
                diag.update(svd_sampled=True, numerical_rank=int((sv>tol).sum()),
                            expected_full_rank=min(A.shape), singular_value_tolerance=tol,
                            maximum_singular_value=float(sv[0]), minimum_singular_value=float(sv[-1]),
                            condition_number_2=float(sv[0]/sv[-1]) if sv[-1]>0 else None)
            else:
                diag["svd_sampled"]=False
            system=gram+ridge*np.eye(len(gram)) if under else gram
            try:
                chol=sla.cho_factor(system,lower=True,check_finite=False)
                dd=np.diag(chol[0])
                diag.update(cholesky_success=True,cholesky_diagonal_ratio=float(dd.max()/dd.min()))
            except np.linalg.LinAlgError as e:
                chol=None
                diag.update(cholesky_success=False,failure=str(e))
                failures.append(dict(paragraph=item["paragraph"],k=k,error=str(e)))
            diagnostics.append(diag)
            Z=np.column_stack([d*np.floor(lp[ids]/d+.5) for d in spec["meshes"]])
            if chol is not None:
                rhs=Z-bias[ids,None]
                solved=sla.cho_solve(chol,rhs if under else A.T@rhs,check_finite=False)
                decoded=A.T@solved if under else solved
                for t,d in enumerate(spec["meshes"]):
                    solutions.append(decoded[:-1,t]); planned.append((k,d))
            for t,d in enumerate(spec["meshes"]):
                lower=np.exp(Z[:,t]-d/2)
                residual=float(1-lower.sum())
                assert residual>0, (slug,j,k,d,residual)
                lq=np.full(V,np.log(residual/(V-k)))
                lq[ids]=np.log(lower)
                lq-=logsumexp(lq)
                row=dict(paragraph=item["paragraph"],source_row=item["source_row"],k=k,K_over_V=k/V,
                         mesh=d,g=2*d,gold_first_token_id=gold,gold_word_token_count=len(item["gold_ids"]),
                         teacher=teacher,teacher_mass_of_returned_ids=float(p[ids].sum()),
                         selected_score_max_absolute_error=float(np.abs(Z[:,t]-lp[ids]).max()),
                         receivers={"uniform_tail":metrics(lp,p,lq,top20,gold,teacher_argmax)})
                if chol is None:row["receivers"]["head_ls"]=dict(failed=True,error=diag["failure"])
                rows.append(row)
        if solutions:
            H=np.column_stack(solutions)
            pred=W@H+bias[:,None]
            pred-=logsumexp(pred,axis=0)[None,:]
            for column,(k,d) in enumerate(planned):
                row=next(x for x in rows[-9:] if x["k"]==k and x["mesh"]==d)
                row["receivers"]["head_ls"]=metrics(lp,p,pred[:,column],top20,gold,teacher_argmax)
        if (j+1)%8==0 or smoke:
            print(json.dumps(dict(model=slug,passages=j+1,elapsed=time.perf_counter()-started,configurations=len(rows))),flush=True)
    groups=[]
    for k in [32,128,r+101]:
        for d in spec["meshes"]:
            subset=[x for x in rows if x["k"]==k and x["mesh"]==d]
            for receiver in ["head_ls","uniform_tail"]:
                vals=[x["receivers"][receiver] for x in subset if not x["receivers"][receiver].get("failed",False)]
                names=spec["metrics"]
                groups.append(dict(k=k,mesh=d,receiver=receiver,n_success=len(vals),n_failed=len(subset)-len(vals),
                                   mean={name:float(np.mean([v[name] for v in vals])) for name in names} if vals else None,
                                   median={name:float(np.median([v[name] for v in vals])) for name in names} if vals else None))
    result=dict(metadata=dict(model=model_spec["model"],slug=slug,r=r,V=V,protocol_sha256=sha(PROTOCOL),
                              source_sha256=sha(__file__),state_sha256=sha(state_path),head_cache_sha256=sha(head_path),
                              completed=not smoke,smoke_excluded=smoke,paragraphs=len(samples),configuration_cells=len(rows),
                              receivers_per_cell=2,cpu_threads=4,head_load_seconds=head_loaded,elapsed_seconds=time.perf_counter()-started),
                failures=failures,groups=groups,diagnostics=diagnostics,rows=rows)
    dest=ROOT/"outputs"/("review_topk_smoke" if smoke else "review_topk")/(slug+".json")
    save(dest,result)
    print(json.dumps(result["metadata"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--model",choices=SLUGS)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.freeze:freeze()
        else:
            for slug in [args.model] if args.model else SLUGS:run(slug,args.smoke)

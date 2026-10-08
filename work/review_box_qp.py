"""Dual active-set solution of a positive-definite metric box projection."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve

def project_box(target,covariance,bound,tolerance=1e-9,maxiter=10000):
    """min .5(z-target)^T covariance^-1(z-target), |z|<=bound.

    Lawson-Hanson active-set steps solve the equivalent nonnegative quadratic
    dual; covariance is computed from the original primal regression metric.
    """
    t=np.asarray(target,dtype=float)/bound
    C=np.asarray(covariance,dtype=float)
    scale=np.sqrt(np.diag(C)); active=[]; lam=np.empty(0); z=t.copy()
    outer=inner=0
    for outer in range(maxiter):
        signed=np.r_[z,-z]; violation=(signed-1)/np.r_[scale,scale]
        choice=int(np.argmax(violation))
        if float(violation[choice])<=tolerance:
            break
        if choice in active:
            # Numerical solve residual bounds the remaining violation.
            break
        active.append(choice); lam=np.r_[lam,0.]
        for inner in range(maxiter):
            ids=np.asarray(active)%len(t); signs=np.where(np.asarray(active)<len(t),1.,-1.)
            S=signs/scale[ids]
            K=(S[:,None]*C[np.ix_(ids,ids)])*S[None,:]
            rhs=(signs*t[ids]-1)/scale[ids]
            candidate=cho_solve(cho_factor((K+K.T)/2,lower=True,check_finite=False),rhs,check_finite=False)
            if np.all(candidate>0):
                lam=candidate; break
            bad=candidate<=0
            ratios=lam[bad]/(lam[bad]-candidate[bad])
            step=float(ratios.min())
            lam=lam+step*(candidate-lam)
            keep=lam>1e-13
            active=[a for a,k in zip(active,keep) if k];lam=lam[keep]
        if len(active):
            ids=np.asarray(active)%len(t); signs=np.where(np.asarray(active)<len(t),1.,-1.)
            z=t-C[:,ids]@(signs*lam/scale[ids])
        else:z=t.copy()
    violation=max(0,float(np.max(np.abs(z)))-1)
    dualmin=float(lam.min()) if len(lam) else 0.
    success=violation<=max(1e-8,tolerance*float(scale.max())) and dualmin>=-1e-10
    clipped=np.clip(z,-1,1); delta=clipped-t
    gradient=cho_solve(cho_factor((C+C.T)/2,lower=True,check_finite=False),delta,check_finite=False)
    alpha=np.zeros(len(t))
    if len(active):
        ids=np.asarray(active)%len(t); signs=np.where(np.asarray(active)<len(t),1.,-1.)
        alpha[ids]=signs*lam/scale[ids]
        rhs=(signs*t[ids]-1)/scale[ids]
        dual=float(lam@rhs-.5*alpha@C@alpha)
        complementarity=float(np.max(np.abs(lam*(signs*clipped[ids]-1)/scale[ids])))
    else:dual=0.;complementarity=0.
    primal=float(.5*delta@gradient)
    stationarity=float(np.max(np.abs(gradient+alpha)))
    return clipped*bound,dict(active=True,success=success,iterations=outer,
          active_constraints=len(active),box_violation=violation*bound,
          normalized_dual_minimum=dualmin,message='dual active-set metric projection',
          stationarity_infinity_norm=stationarity,
          relative_stationarity=stationarity/(1+float(np.max(np.abs(alpha)))),
          primal_objective=primal*bound**2,dual_objective=dual*bound**2,
          absolute_duality_gap=abs(primal-dual)*bound**2,
          relative_duality_gap=abs(primal-dual)/max(abs(primal),abs(dual),1e-30),
          complementarity_infinity_norm=complementarity,
          maximum_scaled_constraint_violation=float(np.max((np.r_[z,-z]-1)/np.r_[scale,scale])))

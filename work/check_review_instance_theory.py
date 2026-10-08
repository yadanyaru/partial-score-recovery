"""Numerical falsification checks for the new finite-error statements.

These are mathematical examples, not additional pretrained-model experiments.
"""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.special import softmax, logsumexp

ROOT=__import__("runtime_paths").ROOT
rng = np.random.default_rng(2026100819)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def fisher(W, p):
    mu = p @ W
    return (W - mu).T @ (p[:, None] * (W - mu))


def js(p, q):
    mid = (p + q) / 2
    return (kl(p, mid) + kl(q, mid)) / 2


def osc(x):
    return float(np.ptp(x))


report = {'seed': 2026100819, 'scope': 'finite mathematical examples; no pretrained performance claim'}

# Exhaustive Bayes comparison over every static two-coordinate multiset.
examples = []
for m in [3, 4, 8, 16]:
    n = int(np.ceil(np.log(100 * m)))
    t = np.exp(-(n + .5))
    probs = []
    for a in range(m):
        for sign in [-1, 1]:
            p = np.full(m + 2, t)
            p[0] = np.exp(-(a + 3))
            p[a + 1] *= np.exp(sign / 4)
            p[-1] = 1 - p[:-1].sum()
            probs.append(p)
    probs = np.array(probs)
    # Build logs from their analytic definitions to resolve exact lattice ties.
    logs = np.log(probs)
    logs[:, 1:m + 1] = -(n + .5)
    for a in range(m):
        logs[2*a, a+1] -= .25
        logs[2*a+1, a+1] += .25
    codes = np.floor(logs + .5).astype(int)
    assert np.all(probs[:, -1] > .94)
    assert np.all(codes[:, -1] == 0)
    for a in range(m):
        changed = np.flatnonzero(codes[2*a] != codes[2*a+1])
        assert changed.tolist() == [a+1]
        assert len(set((int(codes[i, 0]), int(codes[i, a+1])) for i in [2*a, 2*a+1])) == 2
    # The posterior arithmetic mixture minimizes forward KL exactly.
    best = np.inf
    best_list = None
    count = 0
    for S in itertools.combinations_with_replacement(range(m + 2), 2):
        groups = {}
        for i, code in enumerate(codes[:, S]):
            groups.setdefault(tuple(code), []).append(i)
        risk = 0.
        for group in groups.values():
            q = probs[group].mean(axis=0)
            risk += sum(kl(probs[i], q) for i in group) / (2*m)
        if risk < best:
            best, best_list = risk, S
        count += 1
    J = np.array([js(probs[2*a], probs[2*a+1]) for a in range(m)])
    lower = (m-2) / m * J.min()
    assert best > 0 and best >= lower - 1e-14
    # A public affine-softmax factorization, reduced to faithful contrasts.
    b = logs[0]
    X = (logs[1:] - b).T
    X = X - X[-1:]
    u, sv, vh = np.linalg.svd(X, full_matrices=False)
    rank = int(np.sum(sv > 1e-10))
    W = u[:, :rank] * sv[:rank]
    states = np.vstack([np.zeros(rank), vh[:rank].T])
    reconstruction = softmax(states @ W.T + b, axis=1)
    residual = float(np.max(np.abs(reconstruction - probs)))
    assert residual < 1e-12
    examples.append(dict(classes=m, vocabulary=m+2, predictive_rank=rank,
                         static_lists=count, optimal_static_forward_KL=best,
                         routing_lower_bound=lower, adaptive_both_KL=0.,
                         best_static_list=list(best_list), head_residual=residual))
report['routing_examples'] = examples

# Pair separation, local KL constants, and secant-Jacobian comparison.
W = rng.normal(size=(7, 2))
b = rng.normal(scale=.3, size=7)
h0 = np.array([.17, -.21])
p0 = softmax(W @ h0 + b)
F0 = fisher(W, p0)
eig, U = np.linalg.eigh(F0)
root_inv = U @ np.diag(eig**-.5) @ U.T
S = [0, 1, 2, 3]
J0 = W[S] - p0 @ W
A = J0 @ root_inv
vertices = []
for rows in itertools.combinations(range(len(S)), 2):
    if abs(np.linalg.det(A[list(rows)])) < 1e-12:
        continue
    for signs in itertools.product([-1., 1.], repeat=2):
        v = np.linalg.solve(A[list(rows)], signs)
        if np.max(np.abs(A @ v)) <= 1 + 1e-10:
            vertices.append(v)
lambda_s = 1 / np.sqrt(max(v @ v for v in vertices))
R = .025
cross = (W[:, None] - W[None, :]).reshape(-1, 2)
b0 = R * np.max(np.linalg.norm(cross @ root_inv, axis=1))
rho = np.exp(b0) * R / lambda_s
assert rho < 1
max_secant_ratio = 0.
min_JS_ratio = np.inf
max_KL_ratio = 0.
min_pair_lemma_ratio = np.inf
for _ in range(1500):
    points = []
    for j in range(2):
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        points.append(h0 + root_inv @ (direction * R * np.sqrt(rng.random())))
    u, v = points
    d = u - v
    norm2 = float(d @ F0 @ d)
    lu = W @ u + b - logsumexp(W @ u + b)
    lv = W @ v + b - logsumexp(W @ v + b)
    rem = np.max(np.abs(lu[S] - lv[S] - J0 @ d))
    assert rem <= np.exp(b0) * R * np.sqrt(norm2) + 1e-13
    max_secant_ratio = max(max_secant_ratio, rem / max(np.max(np.abs(J0 @ d)), 1e-30))
    p, q = np.exp(lu), np.exp(lv)
    js_ratio = js(p, q) / norm2
    min_JS_ratio = min(min_JS_ratio, js_ratio)
    loss_ratio = max(kl(p, q), kl(q, p)) / norm2
    max_KL_ratio = max(max_KL_ratio, loss_ratio)
    assert js_ratio >= np.exp(-4*b0)/8 - 1e-8
    assert loss_ratio <= np.exp(b0)/2 + 1e-8
    center, step = (u+v)/2, d/2
    t0 = osc(W @ step)
    f0 = float(step @ fisher(W, softmax(W @ center+b)) @ step)
    ratio = js(p, q) / f0
    min_pair_lemma_ratio = min(min_pair_lemma_ratio, ratio / (.5*np.exp(-3*t0)))
    assert .5*np.exp(-3*t0) - 1e-8 <= ratio <= .5*np.exp(t0) + 1e-8
report['local_checks'] = dict(pairs=1500, lambda_S=lambda_s, radius=R,
    b0=float(b0), rho=float(rho), maximum_secant_relative_error=max_secant_ratio,
    minimum_JS_over_squared_Fisher_distance=min_JS_ratio,
    maximum_both_KL_over_squared_Fisher_distance=max_KL_ratio,
    minimum_pair_lower_bound_ratio=min_pair_lemma_ratio)

# Subspace recovery with actual biased and random bounded score errors.
W = rng.normal(size=(11, 5))
b = rng.normal(size=11)
U = np.linalg.qr(rng.normal(size=(5, 2)))[0]
S = [0, 1, 2]
C = W[S[1:]] - W[S[0]]
L = np.linalg.inv(C @ U)
cross = (W[:, None] - W[None, :]).reshape(-1, 5)
AU = float(np.max(np.sum(np.abs(cross @ U @ L), axis=1)))
AX = float(np.max(np.linalg.norm(cross @ (U @ L @ C - np.eye(5)), axis=1)))
max_fraction = 0.
for _ in range(1200):
    z = rng.normal(size=2)
    xi = rng.normal(size=5) * rng.choice([0., .001, .02, .1])
    xi -= U @ (U.T @ xi)
    h = U @ z + xi
    g = float(rng.choice([.001, .02, .2]))
    ell = W @ h + b - logsumexp(W @ h + b)
    e = rng.uniform(-g/2, g/2, size=3)
    contrast = ell[S[1:]] + e[1:] - ell[S[0]] - e[0] - (b[S[1:]] - b[S[0]])
    hat = U @ L @ contrast
    p, q = softmax(W @ h+b), softmax(W @ hat+b)
    risk = max(kl(p, q), kl(q, p))
    bound = (g*AU+AX*np.linalg.norm(xi))**2/8
    assert risk <= bound + 1e-12
    max_fraction = max(max_fraction, risk / max(bound, 1e-30))
report['subspace_checks'] = dict(cases=1200, ambient_rank=5, subspace_rank=2,
                                 scores=3, maximum_risk_bound_fraction=max_fraction)

# Exact conditional candidate risks and moments with genuinely different decoders.
W = rng.normal(size=(8, 2))
b = rng.normal(scale=.3, size=8)
h = rng.normal(scale=.4, size=2)
p = softmax(W @ h + b)
ell = np.log(p)
d = .01
candidates = []
for S in [[0, 1, 2], [2, 3, 4], [0, 3, 6, 7]]:
    augmented = np.column_stack([W[S], -np.ones(len(S))])
    D = np.linalg.pinv(augmented)[:2]
    frac = ell[S]/d - np.floor(ell[S]/d)
    baseline = d*np.floor(ell[S]/d) - ell[S]
    for law in ['nearest', 'independent', 'shared']:
        outcomes = []
        if law == 'nearest':
            outcomes = [(1., d*np.floor(ell[S]/d+.5)-ell[S])]
        elif law == 'independent':
            for bits in itertools.product([0, 1], repeat=len(S)):
                bits = np.array(bits)
                mass = np.prod(np.where(bits, frac, 1-frac))
                outcomes.append((mass, baseline+d*bits))
        else:
            cuts = np.unique(np.r_[0., frac, 1.])
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                bits = ((lo+hi)/2 < frac).astype(float)
                outcomes.append((hi-lo, baseline+d*bits))
        moment = sum(mass*np.outer(e, e) for mass, e in outcomes)
        Q = float(np.trace(D.T @ fisher(W, p) @ D @ moment)/2)
        risk = sum(mass*max(kl(p, softmax(W @ (h+D@e)+b)),
                            kl(softmax(W @ (h+D@e)+b), p)) for mass, e in outcomes)
        pair_scores = np.empty((8, 8))
        for y in range(8):
            for yp in range(8):
                a = D.T @ (W[y]-W[yp])
                pair_scores[y, yp] = a @ moment @ a / 4
        Qpair = float(p @ pair_scores @ p)
        assert abs(Qpair-Q) < 1e-14
        kappa = np.max(np.sum(np.abs((W[:,None]-W[None,:]) @ D), axis=2))
        beta = d*kappa
        assert np.exp(-2*beta)*Q <= risk+1e-14 <= np.exp(2*beta)*Q+1e-13
        candidates.append(dict(S=S, law=law, Q=Q, risk=risk,
                               kappa=float(kappa), pair_scores=pair_scores))
beta = d*max(c['kappa'] for c in candidates)
Bstar = beta**2/4
for N in [16, 64, 256]:
    allowance = 2*np.exp(2*beta)*Bstar*np.sqrt(np.log(2*len(candidates)/.05)/(2*N))
    bound = np.exp(4*beta)*min(c['risk'] for c in candidates)+allowance
    violations = 0
    selected_risks = []
    for _ in range(250):
        Y = rng.choice(8, size=N, p=p)
        Yp = rng.choice(8, size=N, p=p)
        values = [c['pair_scores'][Y, Yp].mean() for c in candidates]
        risk = candidates[int(np.argmin(values))]['risk']
        selected_risks.append(risk)
        violations += risk > bound
    report.setdefault('joint_sampling_checks', []).append(dict(N=N, replications=250,
        candidates=len(candidates), high_probability_bound=float(bound),
        observed_violations=int(violations), mean_selected_risk=float(np.mean(selected_risks))))
report['joint_exact_checks'] = dict(candidates=len(candidates), different_coordinate_lists=3,
    beta=float(beta), minimum_exact_risk=min(c['risk'] for c in candidates),
    identity_max_error=max(abs(float(p@c['pair_scores']@p)-c['Q']) for c in candidates))

# A finite common-geometry prior with state-independent conditional moments.
Fref = fisher(W, p)
ev, basis = np.linalg.eigh(Fref)
invroot = basis @ np.diag(ev**-.5) @ basis.T
library = []
for S in [[0, 1, 2], [2, 3, 4], [0, 3, 6, 7]]:
    D = np.linalg.pinv(np.column_stack([W[S], -np.ones(len(S))]))[:2]
    noise = rng.uniform(-d, d, len(S))
    library.append((D, noise))
risk_matrix = []
epsilon = 0.
beta_common = 0.
for _ in range(100):
    state = h + rng.normal(scale=.005, size=2)
    teacher = softmax(W @ state + b)
    rel = invroot @ fisher(W, teacher) @ invroot
    epsilon = max(epsilon, float(np.max(np.abs(np.linalg.eigvalsh(rel)-1))))
    risks = []
    for D, noise in library:
        beta_common = max(beta_common, osc(W @ D @ noise))
        risks.append(sum(max(kl(teacher, softmax(W@(state+sign*D@noise)+b)),
                             kl(softmax(W@(state+sign*D@noise)+b), teacher))
                         for sign in [-1, 1])/2)
    risk_matrix.append(risks)
risk_matrix = np.array(risk_matrix)
fixed_risk = float(risk_matrix.mean(axis=0).min())
full_state_routing_risk = float(risk_matrix.min(axis=1).mean())
factor = float(np.exp(4*beta_common)*(1+epsilon)/(1-epsilon))
assert epsilon < 1 and fixed_risk <= factor*full_state_routing_risk+1e-14
report['common_geometry_checks'] = dict(prior_states=100, designs=3,
    epsilon=epsilon, zeta=0., beta=beta_common, near_equivalence_factor=factor,
    fixed_design_risk=fixed_risk, full_state_routing_risk=full_state_routing_risk,
    note='The full-state routing benchmark is stronger than any finite pilot; this is a mathematical check, not a realizable-query comparison.')
path = ROOT/'outputs/REVIEW_INSTANCE_THEORY_CHECKS.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))

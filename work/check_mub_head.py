"""Numerical diagnostic of classical prime-dimensional MUB realification.

This checks the proposed head geometry, not a certified minimax bound.
No language-model data or outcomes enter this deterministic construction.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

root=__import__("runtime_paths").ROOT
rows = []
for prime in [7, 11, 17, 31, 67, 127, 257]:
    x = np.arange(prime)[:, None]
    b = np.arange(prime)[None, :]
    complex_bases = [np.eye(prime, dtype=complex)]
    complex_bases += [np.exp(2j*np.pi*((a*x*x+b*x) % prime)/prime)/np.sqrt(prime)
                      for a in range(5)]
    real_bases = []
    for basis in complex_bases:
        real_bases.append(np.column_stack([
            np.concatenate([basis.real, basis.imag], axis=0),
            np.concatenate([-basis.imag, basis.real], axis=0)]))
    head = np.concatenate([basis.T for basis in real_bases], axis=0)
    gram = head @ head.T
    np.fill_diagonal(gram, 0)
    coherence = float(np.max(np.abs(gram)))
    orthogonality = max(float(np.max(np.abs(basis.T@basis-np.eye(2*prime))))
                        for basis in real_bases)
    # The standard real basis plus the uniform chirp column affinely spans R^r.
    # Their affine determinant is 1 - sum(uniform); report this explicit gap.
    uniform = head[2*prime]
    affine_gap = abs(float(np.sum(uniform))-1)
    expected = 1/np.sqrt(prime)
    assert coherence <= expected + 1e-12
    assert orthogonality <= 1e-12
    assert affine_gap >= (np.sqrt(prime)-1)-1e-12
    rank = 2*prime
    vocabulary = 6*rank
    budget = 5*(rank+1)
    omission = max(0, (vocabulary-budget)*(vocabulary-budget-1))/(vocabulary*(vocabulary-1))
    if rank >= 30:
        assert omission >= .01
    rows.append(dict(prime=prime, rank=rank, vocabulary=vocabulary,
                     budget=budget, coherence=coherence,
                     coherence_bound=expected, orthogonality_error=orthogonality,
                     affine_gap=affine_gap, omitted_pair_probability=omission))
report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='deterministic float64 geometry diagnostic; no theorem certificate',
              cases=rows)
(root/'outputs'/'mub_head_geometry_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

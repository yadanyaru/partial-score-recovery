## Quick start: no GPU or network needed after installing dependencies

Use Python 3.11 or newer. Commands work in PowerShell, Bash and other standard shells.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in PowerShell, then run:

```sh
python -m pip install -r requirements-cpu.txt
python review.py verify
python -m unittest discover -s tests -v
python review.py smoke
python review.py analyze
```

`verify` checks reference checksums, source syntax, private-path/credential patterns,
and file sizes. `smoke` runs deterministic CPU mathematical checks. `analyze`
recomputes group statistics, paired passage intervals and comparison summaries from
the included outcomes. Outputs go to
`runs/reference/outputs/`; CPU checks use `runs/cpu-smoke/outputs/`.

An analysis is a recalculation from stored outcomes. Use a new `--workspace NAME` to repeat it after modifying generated summaries.
All original reference files remain unchanged under `reference/`.

## Recompute model predictions and experiments

Install a CUDA-compatible PyTorch build for your system, followed by the remaining
dependencies. `requirements-gpu.txt` includes PyTorch but does not select a CUDA
driver. The original observed versions are in `config/original_environment.json`.

```sh
python -m pip install -r requirements-gpu.txt
python review.py download-models
python review.py prepare-data --workspace experiment
python review.py run --stage states --workspace experiment
python review.py run --stage precision --workspace experiment
python review.py run --stage topk --workspace experiment
python review.py run --stage adaptive --workspace experiment
```

The download command fetches the three exact model revisions listed in
`work/frozen_models.py`. No Hugging Face token is required for these public models.
Data preparation fetches public WikiText-2 and LAMBADA files, selects the original
source rows, and verifies every selected text and serialized corpus against
`config/data_selections.json`. No narrative corpus text or model weights are bundled.

The `states`, `precision` and `adaptive` stages require CUDA. `topk` uses cached
native states and CPU linear algebra. The original GPU was an RTX 3060 Laptop GPU;
runtime and peak allocation are hardware-specific. Results can vary slightly with
GPU kernels, BLAS and library versions. The original state arrays, random packets
and reference losses are provided for comparison.

```sh
python review.py run --stage low-disclosure --workspace witnesses
python review.py run --stage baseline --workspace controls
python review.py run --stage response --workspace controls
python review.py run --stage shared --workspace controls
python review.py run --stage confirmation --workspace controls
python review.py run --stage sampling --workspace controls
```

`low-disclosure` is a CPU numerical witness computation. For the response, shared
and confirmation stages, the previously selected public head-only IDs are reused
from included baseline designs if no new baseline is present. `sampling` requires
the confirmation stage in the same workspace. Each stage preserves its protocol
and existing output; use a fresh workspace for an independent rerun.

## Repository layout

```text
review.py                 documented entry point
work/                     portable experimental implementations
config/                   model/environment, sample selections, checksums
reference/outputs/        gzip-compressed anonymized reference outcomes
tests/                    small deterministic implementation checks
tools/                    release validation
docs/                     experiment map, anonymization and dependency notes
runs/                     generated locally; ignored by Git
cache/                    downloaded models/data/heads; ignored by Git
```

Set `PREDREC_CACHE` to place model/data caches on another disk; `HF_HOME` can
override the Hugging Face cache separately. For example, in PowerShell:

```powershell
$env:PREDREC_CACHE = "./external-cache"
```

Full stage outputs use independent `runs/NAME` workspaces. Source files are copied
there so protocol hashes refer to the actual executed implementation. Historical
source/protocol digests in reference files are remapped to exported anonymized
files; statistical values, selected IDs, seeds and experimental settings are
unchanged. See `config/export_notes.json`.


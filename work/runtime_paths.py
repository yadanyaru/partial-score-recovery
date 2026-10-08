"""Runtime paths shared by the portable experiment entry points."""
from pathlib import Path
import os

REPOSITORY=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('PREDREC_ROOT',str(REPOSITORY))).resolve()
CACHE=Path(os.environ.get('PREDREC_CACHE',str(REPOSITORY/'cache'))).resolve()
HF_CACHE=Path(os.environ.get('HF_HOME',str(CACHE/'hf'))).resolve()
HEAD_CACHE=CACHE/'heads'


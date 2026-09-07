import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# ml/ modules import each other by bare name, and helpers.py reads env.env from the project root
os.chdir(ROOT)
for p in (str(ROOT), str(ROOT / 'ml')):
    if p not in sys.path:
        sys.path.append(p)

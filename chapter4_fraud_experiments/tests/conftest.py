import sys
from pathlib import Path

# make the package fraud/ importable when the tests are run from any folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

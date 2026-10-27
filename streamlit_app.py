"""Run the current checkout; requirements still install its dependencies."""
from pathlib import Path
import sys

# Prefer GitHub's current source to a previously installed copy of the package.
source = str(Path(__file__).resolve().parent / 'src')
if source not in sys.path:
    sys.path.insert(0, source)

from oled_opv_lab.app import main
main()

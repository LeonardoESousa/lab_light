"""Launch the installed package from any working directory."""
import sys
from pathlib import Path


def main():
    from streamlit.web import cli
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(cli.main())


if __name__ == "__main__":
    main()

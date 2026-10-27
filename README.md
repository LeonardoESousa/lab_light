# Molecular Light Lab

Requires **Python 3.10 or newer** and an internet connection for installation.

Download or clone the repository, then open a terminal in the folder containing `pyproject.toml` and `launch.py`.

## Quick start

```sh
python launch.py
```

This installs the package in its own environment and opens the app in your browser. Use the same command for subsequent launches and after updating the source.

## Install manually

If you already have a Python environment:

```sh
python -m pip install .
oled-opv-lab
```

Alternatively, launch with:

```sh
python -m oled_opv_lab.cli
```

After downloading newer source code, reinstall and run:

```sh
python -m pip install --upgrade .
oled-opv-lab
```

Press **Ctrl+C** in the terminal to stop the app.

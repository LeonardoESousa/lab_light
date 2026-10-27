"""Create a private environment automatically, install the local wheel, and launch.

Requires a system Python >=3.10, plus internet on first run for dependencies.
Run: python launch.py
"""
from pathlib import Path
import os
import hashlib
import subprocess
import sys
import venv


def main():
    if sys.version_info < (3,10):
        raise SystemExit('Please install Python 3.10 or newer, or use the hosted app.')
    root=Path(__file__).resolve().parent
    environment=root/'.lab-env'
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    # A source patch must not keep launching the unchanged bundled wheel.
    if (root/'pyproject.toml').exists() and (root/'src'/'oled_opv_lab').is_dir():
        install_target=root
        digest=hashlib.sha256()
        for source in [(root/'pyproject.toml'), *sorted((root/'src').rglob('*.py'))]:
            digest.update(source.relative_to(root).as_posix().encode())
            digest.update(source.read_bytes())
        identity='source:'+digest.hexdigest()
    else:
        wheels=sorted((root/'dist').glob('oled_opv_lab-*.whl'))
        if not wheels:
            raise SystemExit('No source project or wheel found beside launch.py.')
        install_target=wheels[-1]
        identity=f'{install_target.name}:{install_target.stat().st_mtime_ns}'
    marker=environment/'installed-wheel.txt'
    if not python.exists():
        print('Creating the application environment…',flush=True)
        venv.create(environment,with_pip=True)
    if not marker.exists() or marker.read_text()!=identity:
        print('Installing the application and dependencies (first launch requires internet)…',flush=True)
        subprocess.run([str(python),'-m','pip','install','--no-user',str(install_target)],check=True)
        marker.write_text(identity)
    subprocess.run([str(python),'-m','oled_opv_lab.cli',*sys.argv[1:]],check=True)


if __name__=='__main__':
    main()

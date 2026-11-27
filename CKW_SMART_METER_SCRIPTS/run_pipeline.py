"""Ejecuta el flujo de datos: parquet, timestamps, ISO, orden y LOCF."""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Sin nombre de carpeta fijo: --data-dir indica dónde están los días.
DATA = Path(".")
SCRIPTS = {
    "parquet": BASE / "parquet_to_csv.py",
    "timestamps": BASE / "fix_timestamps.py",
    "iso": BASE / "add_iso_columns.py",
    "order": BASE / "order_by_user.py",
    "locf": BASE / "apply_locf.py",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DATA))
    p.add_argument("--years", nargs="*")
    p.add_argument("--ordered-dir")
    p.add_argument("--locf-dir")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--step-minutes", type=int, default=15)
    p.add_argument("--progress-every", type=int)
    p.add_argument("--limit-files", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip", nargs="*", default=[], choices=SCRIPTS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def year_dirs(data_dir, years):
    """Carpetas que contienen CSV de día, con cualquier nombre."""
    excluded = {
        "ordered_files", "locf_files", "fixed_timestamps", "summary", "__pycache__"
    }
    day_parents = {
        p.parent for p in data_dir.rglob("*.csv")
        if p.stem.isdigit()
        and p.is_file()
        and not any(part in excluded for part in p.parts)
    }
    if years:
        dirs = [
            d for d in day_parents
            if any(year == d.name or year in d.name for year in years)
        ]
        if not dirs:
            dirs = [data_dir / name for name in years]
    else:
        dirs = day_parents
    return sorted(d for d in dirs if d.is_dir())


def run(name, command, skip, dry_run):
    if name in skip:
        print(f"\n=== {name}: omitido ===", flush=True)
        return True
    print(f"\n=== {name} ===\n{shlex.join(command)}", flush=True)
    if dry_run:
        return True
    started = time.monotonic()
    result = subprocess.run(command, check=False)
    if result.returncode:
        print(f"ERROR: {name} devolvió {result.returncode}.", file=sys.stderr)
        return False
    print(f"Completado en {time.monotonic() - started:.0f}s.", flush=True)
    return True


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    ordered_dir = Path(args.ordered_dir).expanduser() if args.ordered_dir else data_dir / "ordered_files"
    locf_dir = Path(args.locf_dir).expanduser() if args.locf_dir else data_dir / "locf_files"
    dirs = year_dirs(data_dir, args.years)

    if not data_dir.is_dir() or not dirs:
        print(f"Error: no hay datos en {data_dir}.", file=sys.stderr)
        return 1
    for script in SCRIPTS.values():
        if not script.is_file():
            print(f"Error: falta {script}.", file=sys.stderr)
            return 1

    skip = set(args.skip)
    python = sys.executable
    started = time.monotonic()
    print(f"Datos: {data_dir}\nOrdenados: {ordered_dir}\nLOCF: {locf_dir}", flush=True)

    parquet_dirs = [d for d in dirs if any(d.rglob("*.parquet"))]
    if "parquet" not in skip and not parquet_dirs:
        print("\n=== parquet: no hay .parquet, se omite ===", flush=True)
        skip.add("parquet")
    for d in parquet_dirs:
        if not run("parquet", [python, str(SCRIPTS["parquet"]), str(d)], skip, args.dry_run):
            return 1

    for d in dirs:
        command = [
            python, str(SCRIPTS["timestamps"]), str(d),
            "--in-place", "-w", str(args.workers),
            "--step-minutes", str(args.step_minutes),
        ]
        if args.limit_files:
            command += ["--limit-files", str(args.limit_files)]
        if not run("timestamps", command, skip, args.dry_run):
            return 1

    for d in dirs:
        if not run("iso", [python, str(SCRIPTS["iso"]), str(d), "-w", str(args.workers)], skip, args.dry_run):
            return 1

    command = [
        python, str(SCRIPTS["order"]),
        "--input-dir", str(data_dir),
        "--output-dir", str(ordered_dir),
    ]
    if args.years:
        command += ["--years", *args.years]
    if args.limit_files:
        command += ["--limit-files", str(args.limit_files)]
    if args.progress_every:
        command += ["--progress-every", str(args.progress_every)]
    if args.force:
        command.append("--force")
    if not run("order", command, skip, args.dry_run):
        return 1

    command = [
        python, str(SCRIPTS["locf"]), str(ordered_dir),
        "-o", str(locf_dir), "-w", str(args.workers),
    ]
    if args.limit_files:
        command += ["--limit-files", str(args.limit_files)]
    if args.force:
        command.append("--force")
    if not run("locf", command, skip, args.dry_run):
        return 1

    print(f"\nFlujo completo en {time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

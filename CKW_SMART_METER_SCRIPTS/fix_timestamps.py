"""
Repara timestamps ausentes/inválidos y marca las filas reparadas con imp=1.

Uso:
    python3 fix_timestamps.py <ruta_csv_o_directorio> [opciones]

Opciones:
    -o, --output-dir DIR      Salida (por defecto: fixed_timestamps). Ignorado con --in-place.
    --in-place                Sobrescribe los CSV de entrada.
    -w, --workers N           Procesos en paralelo (por defecto: 8).
    --limit-files N           Procesa solo los primeros N archivos.
    --step-minutes N          Paso para extrapolar vecinos, en minutos (por defecto: 15).
    -f, --force               Reprocesa aunque la salida ya exista.

Ejemplos:
    python3 fix_timestamps.py CKW_SMART_METER/ordered_files -o CKW_SMART_METER/fixed_timestamps
    python3 fix_timestamps.py CKW_SMART_METER/ordered_files --in-place
    python3 fix_timestamps.py un_archivo.csv -o /tmp/salida --force
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from apply_locf import group_positions, natural_sort_key, write_csv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path("fixed_timestamps")

TIMESTAMP_COL = "timestamp"
FLAG_COL = "imp"
DEFAULT_STEP_MINUTES = 15
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def fix_timestamps(df: pd.DataFrame, step_minutes: int = DEFAULT_STEP_MINUTES):
    """Devuelve el DataFrame corregido, timestamps reparados y sin reparar."""
    if TIMESTAMP_COL not in df.columns:
        raise ValueError("Falta la columna 'timestamp'")

    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    invalid = parsed.isna().to_numpy()
    times = parsed.to_numpy(dtype="datetime64[ns]")
    step = pd.Timedelta(minutes=step_minutes)
    values = df[TIMESTAMP_COL].to_numpy(dtype=object).copy()
    fixed = np.zeros(len(df), dtype=bool)

    for positions in group_positions(df):
        missing = positions[invalid[positions]]
        if missing.size == 0:
            continue
        valid = positions[~invalid[positions]]
        if valid.size == 0:
            continue

        for row in missing:
            before = valid[valid < row]
            after = valid[valid > row]
            if before.size and after.size:
                start, end = pd.Timestamp(times[before[-1]]), pd.Timestamp(times[after[0]])
                if end > start:
                    fraction = (row - before[-1]) / (after[0] - before[-1])
                    new_time = start + (end - start) * fraction
                else:  # p. ej. hora repetida por cambio de horario
                    new_time = start + step * (row - before[-1])
            elif before.size:
                new_time = pd.Timestamp(times[before[-1]]) + step * (row - before[-1])
            else:
                new_time = pd.Timestamp(times[after[0]]) - step * (after[0] - row)

            values[row] = new_time.strftime(TS_FORMAT)
            fixed[row] = True

    if FLAG_COL in df.columns:
        imp = pd.to_numeric(df[FLAG_COL], errors="coerce").fillna(0).to_numpy(dtype="int8").copy()
    else:
        imp = np.zeros(len(df), dtype="int8")
    imp[fixed] = 1

    # df no se reutiliza fuera de aquí: se modifica en sitio.
    df[TIMESTAMP_COL] = values
    df[FLAG_COL] = imp
    return df, int(fixed.sum()), int((invalid & ~fixed).sum())


def process_file(
    csv_path: Path, input_root: Path, output_dir: Path, force: bool, step_minutes: int
) -> tuple[bool, str, int, int]:
    """Lee un CSV, repara sus timestamps y lo escribe en output_dir."""
    if output_dir:
        relative = csv_path.relative_to(input_root) if input_root.is_dir() else csv_path.name
        out_path = output_dir / relative
    else:
        out_path = csv_path

    if output_dir and out_path.exists() and not force:
        return True, "Ya procesado (omitido)", 0, 0

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        fixed_df, fixed, unresolved = fix_timestamps(df, step_minutes)
        write_csv(fixed_df, temp_path)
        temp_path.replace(out_path)

        detail = f"{len(fixed_df):,} filas, {fixed:,} timestamps reparados"
        if unresolved:
            detail += f", {unresolved:,} sin vecinos (sin reparar)"
        return True, f"OK ({detail})", fixed, unresolved
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return False, f"ERROR: {exc}", 0, 0


def _worker(args: tuple[Path, Path, Path, bool, int]) -> tuple[bool, str, int, int]:
    csv_path, input_root, output_dir, force, step_minutes = args
    return process_file(csv_path, input_root, output_dir, force, step_minutes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repara timestamps ausentes y marca las filas con imp=1"
    )
    parser.add_argument(
        "path",
        help="CSV o directorio a procesar",
    )
    parser.add_argument(
        "-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directorio de salida (por defecto: {DEFAULT_OUTPUT_DIR}). Ignorado con --in-place",
    )
    parser.add_argument("--in-place", action="store_true", help="Sobrescribe los CSV de entrada")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Procesos en paralelo (por defecto: 8)")
    parser.add_argument("--limit-files", type=int, default=None, help="Procesa solo los primeros N archivos")
    parser.add_argument(
        "--step-minutes", type=int, default=DEFAULT_STEP_MINUTES,
        help=f"Paso para extrapolar vecinos, en minutos (por defecto: {DEFAULT_STEP_MINUTES})",
    )
    parser.add_argument("-f", "--force", action="store_true", help="Reprocesa aunque la salida ya exista")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: la ruta '{target}' no existe.")
        return 1
    if args.workers < 1 or args.step_minutes < 1:
        print("Error: --workers y --step-minutes deben ser mayores que 0.")
        return 1

    output_dir = None if args.in_place else Path(args.output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        files = [target]
        input_root = target.parent
    else:
        input_root = target
        files = sorted(
            (p for p in target.rglob("*.csv")
             if not p.name.endswith((".tmp", ".bak"))
             and (output_dir is None or output_dir not in p.parents)),
            key=natural_sort_key,
        )
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        print(f"No se encontraron archivos .csv en '{target}'.")
        return 0

    destination = "en sitio" if output_dir is None else str(output_dir)
    print(f"Archivos a procesar: {len(files):,}. Destino: {destination}", flush=True)
    tasks = [(p, input_root, output_dir, args.force, args.step_minutes) for p in files]
    done = errors = total_fixed = total_unresolved = 0

    if args.workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(_worker, tasks)
            for index, (path, (ok, msg, fixed, unresolved)) in enumerate(zip(files, results), 1):
                print(f"[{index}/{len(files)}] {path.name} ... {msg}", flush=True)
                done += ok
                errors += not ok
                total_fixed += fixed
                total_unresolved += unresolved
    else:
        for index, task in enumerate(tasks, 1):
            ok, msg, fixed, unresolved = process_file(*task)
            print(f"[{index}/{len(files)}] {task[0].name} ... {msg}", flush=True)
            done += ok
            errors += not ok
            total_fixed += fixed
            total_unresolved += unresolved

    print(
        f"\nFinalizado: {done:,} procesados/omitidos con éxito, {errors:,} errores. "
        f"Timestamps reparados: {total_fixed:,}. Sin reparar: {total_unresolved:,}."
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

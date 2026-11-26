"""
Rellena 'value_kwh' con LAST OBSERVATION CARRIED FORWARD (LOCF) usando 'timestamp'.

En que se basa el metodo:
    LOCF es un metodo de imputacion de series temporales en el que cada valor
    ausente se sustituye por la ultima observacion real anterior en el tiempo
    (R imputeTS::na_locf, DescTools::LOCF, pypots imputation/locf).
    Cuando el hueco esta al principio de la serie no hay observacion anterior:
    se aplica NOCB (Next Observation Carried Backward), es decir, el primer
    valor real se lleva hacia atras. Asi se rellena tambien el principio.

    El orden se deduce de 'timestamp' con orden estable (mergesort), por lo que
    las filas con el mismo timestamp conservan su orden original del CSV. Si
    existe la columna 'id', se aplica por usuario para no mezclar series. Las
    filas con timestamp no válido se dejan intactas y se cuentan al final.

    Se añade la columna 'inputed':
        0 = el valor ya estaba escrito (real)
        1 = el valor se ha puesto con LOCF/NOCB

Uso:
    python3 apply_locf.py [ruta_archivo_o_directorio] [opciones]

Ejemplos:
    python3 apply_locf.py --limit-files 2 -o /tmp/locf_test
    python3 apply_locf.py CKW_SMART_METER/ordered_files -w 8
    python3 apply_locf.py un_archivo.csv -o /tmp/salida --force

Nota: --force sobre un CSV ya imputado no recupera los huecos originales;
recalcula la marca usando los valores que ya tiene. Para volver a imputar, usa
siempre los CSV originales como entrada.
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "CKW_SMART_METER" / "ordered_files"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "CKW_SMART_METER" / "locf_files"

TIMESTAMP_COL = "timestamp"
VALUE_COL = "value_kwh"
ID_COL = "id"
FLAG_COL = "inputed"


def natural_sort_key(path: Path) -> list[int | str]:
    """Ordena números de forma natural: 1, 2, 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rellena value_kwh con LOCF y añade la columna inputed"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"CSV o directorio a procesar. Por defecto {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directorio de destino. Por defecto {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=8,
        help="Número de procesos en paralelo (por defecto: 8)",
    )
    parser.add_argument(
        "--limit-files", type=int, default=None,
        help="Procesar solo los primeros N archivos. Para pruebas",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Rehacer archivos de salida existentes",
    )
    return parser.parse_args(argv)


def group_positions(df: pd.DataFrame) -> list[np.ndarray]:
    """Posiciones de filas de cada serie (por id si existe, si no todo el CSV)."""
    if ID_COL not in df.columns:
        return [np.arange(len(df))]
    return [np.asarray(positions) for positions in df.groupby(ID_COL, sort=False, dropna=False).indices.values()]


def apply_locf(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Devuelve DataFrame, imputadas, sin imputar y timestamps no válidos."""
    required = {TIMESTAMP_COL, VALUE_COL}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(missing_cols))}")

    timestamps = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    valid_timestamps = timestamps.notna().to_numpy()
    invalid = int((~valid_timestamps).sum())

    # Los valores se rellenan como texto: no hay que convertir a float ni
    # reformatear los números al volver a escribir el CSV.
    values = df[VALUE_COL].to_numpy(dtype=object).copy()
    values[pd.isna(values)] = ""
    imputed_mask = np.zeros(len(df), dtype=bool)
    unfilled = 0
    times = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")

    for positions in group_positions(df):
        # Las filas sin timestamp válido se dejan intactas.
        positions = positions[valid_timestamps[positions]]
        if positions.size == 0:
            continue

        # Orden estable por timestamp: los duplicados mantienen su orden.
        order = np.argsort(times[positions], kind="stable")
        positions = positions[order]
        series = values[positions]
        has_value = series != ""
        if not has_value.any():
            unfilled += positions.size
            continue

        # ffill con accumulate, y los huecos iniciales con el primer valor.
        index = np.where(has_value, np.arange(series.size), 0)
        np.maximum.accumulate(index, out=index)
        filled = series[index]
        first = int(np.flatnonzero(has_value)[0])
        if first:
            filled[:first] = series[first]

        values[positions] = filled
        imputed_mask[positions] = ~has_value

    # df no se reutiliza fuera de aquí: se modifica en sitio.
    result = df
    result[VALUE_COL] = values
    result[FLAG_COL] = imputed_mask.astype("int8")
    return result, int(imputed_mask.sum()), unfilled, invalid


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Escribe el CSV con join directo y csv.writer solo si hace falta."""
    header = list(df.columns)
    separators = len(header) - 1
    with open(path, "w", newline="", encoding="utf-8") as fp:
        fp.write(",".join(header) + "\n")
        for row in df.to_numpy(dtype=object):
            line = ",".join(map(str, row))
            if (
                line.count(",") == separators
                and '"' not in line and "\n" not in line and "\r" not in line
            ):
                fp.write(line)
                fp.write("\n")
            else:
                writer = csv.writer(fp, lineterminator="\n")
                writer.writerow(row)


def process_file(
    csv_path: Path,
    input_root: Path,
    output_dir: Path,
    force: bool,
) -> tuple[bool, str, int, int, int]:
    """Lee un CSV, aplica LOCF y lo escribe en output_dir."""
    if output_dir:
        relative = csv_path.relative_to(input_root) if input_root.is_dir() else csv_path.name
        out_path = output_dir / relative
    else:
        out_path = csv_path

    if out_path.exists() and not force:
        return True, "Ya procesado (omitido)", 0, 0, 0

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

        if FLAG_COL in df.columns:
            if not force:
                return True, "Ya tiene 'inputed' (omitido)", 0, 0, 0
            df = df.drop(columns=[FLAG_COL])

        filled, imputed, unfilled, invalid = apply_locf(df)
        write_csv(filled, temp_path)
        temp_path.replace(out_path)

        detail = f"{len(filled):,} filas, {imputed:,} imputadas"
        if unfilled:
            detail += f", {unfilled:,} sin imputar (serie sin valores reales)"
        if invalid:
            detail += f", {invalid:,} timestamps no válidos (sin imputar)"
        return True, f"OK ({detail})", imputed, unfilled, invalid
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return False, f"ERROR: {exc}", 0, 0, 0


def _worker(args: tuple[Path, Path, Path, bool]) -> tuple[bool, str, int, int, int]:
    csv_path, input_root, output_dir, force = args
    return process_file(csv_path, input_root, output_dir, force)


def main() -> int:
    args = parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: la ruta '{target}' no existe.")
        return 1
    if args.workers < 1:
        print("Error: --workers debe ser mayor que 0.")
        return 1
    if args.limit_files is not None and args.limit_files < 1:
        print("Error: --limit-files debe ser mayor que 0.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        files = [target]
        input_root = target.parent
    else:
        input_root = target
        files = sorted(
            (
                path for path in target.rglob("*.csv")
                if not path.name.endswith((".tmp", ".bak"))
                and output_dir not in path.parents
            ),
            key=natural_sort_key,
        )

    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        print(f"No se encontraron archivos .csv en '{target}'.")
        return 0

    print(f"Archivos a procesar: {len(files):,}. Destino: {output_dir}", flush=True)

    tasks = [(path, input_root, output_dir, args.force) for path in files]
    done = errors = total_imputed = total_unfilled = total_invalid = 0

    if args.workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(_worker, tasks)
            for index, (path, (success, msg, imputed, unfilled, invalid)) in enumerate(
                zip(files, results), 1
            ):
                print(f"[{index}/{len(files)}] {path.name} ... {msg}", flush=True)
                if success:
                    done += 1
                else:
                    errors += 1
                total_imputed += imputed
                total_unfilled += unfilled
                total_invalid += invalid
    else:
        for index, path in enumerate(files, 1):
            success, msg, imputed, unfilled, invalid = process_file(
                path, input_root, output_dir, args.force
            )
            print(f"[{index}/{len(files)}] {path.name} ... {msg}", flush=True)
            if success:
                done += 1
            else:
                errors += 1
            total_imputed += imputed
            total_unfilled += unfilled
            total_invalid += invalid

    print(
        f"\nFinalizado: {done:,} procesados/omitidos con éxito, {errors:,} errores. "
        f"Valores imputados: {total_imputed:,}. Sin imputar: {total_unfilled:,}. "
        f"Timestamps no válidos: {total_invalid:,}."
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

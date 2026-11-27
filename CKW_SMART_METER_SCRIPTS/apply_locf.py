"""
Rellena 'value_kwh' usando 'timestamp'.

Metodo: para cada hueco se toma la observacion real mas cercana anterior y la
mas cercana posterior, y se escribe su media. Si la vecina inmediata no tiene
valor se sigue buscando hacia atras/hacia delante la siguiente observacion real
(el "segundo anterior", etc.). Si solo existe un lado, se usa ese unico valor;
si no existe ninguno, se deja vacio. Los huecos consecutivos entre dos valores
reales reciben la misma media.

El orden se deduce de 'timestamp' con orden estable (mergesort), por lo que
las filas con el mismo timestamp conservan su orden original del CSV. Si
existe la columna 'id', se aplica por usuario para no mezclar series. Las
filas con timestamp no válido se dejan intactas y se cuentan al final.

Se añade la columna 'imp' (de imputation):
    0 = el valor ya estaba escrito (real)
    1 = el valor se ha imputado (o ya venía marcado por otra imputación)

Uso:
    python3 apply_locf.py [ruta_archivo_o_directorio] [opciones]

Ejemplos:
    python3 apply_locf.py --limit-files 2 -o /tmp/locf_test
    python3 apply_locf.py ordered_files -w 8
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
DEFAULT_INPUT_DIR = Path("ordered_files")
DEFAULT_OUTPUT_DIR = Path("locf_files")

TIMESTAMP_COL = "timestamp"
VALUE_COL = "value_kwh"
ID_COL = "id"
FLAG_COL = "imp"


def to_float(value: object) -> float:
    """Convierte a float; devuelve nan si el texto no es un número."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def format_imputed(value: float) -> str:
    """Escribe la media sin ceros sobrantes ni artefactos de coma flotante."""
    return format(value, ".12g")


def natural_sort_key(path: Path) -> list[int | str]:
    """Ordena números de forma natural: 1, 2, 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rellena value_kwh con la media de los vecinos y añade la columna imp"
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


def apply_locf(
    df: pd.DataFrame, previous_flags: np.ndarray | None = None
) -> tuple[pd.DataFrame, int, int, int]:
    """Devuelve DataFrame, imputadas, sin imputar y timestamps no válidos."""
    required = {TIMESTAMP_COL, VALUE_COL}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(missing_cols))}")

    timestamps = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    valid_timestamps = timestamps.notna().to_numpy()
    invalid = int((~valid_timestamps).sum())

    # Los valores reales se conservan como texto: no hay que reformatearlos al
    # volver a escribir el CSV. Solo se convierten a float los vecinos que se
    # usan para calcular una media.
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

        # Vecinos reales más cercanos a cada hueco (hacia atrás y adelante):
        # si el inmediato está vacío se sigue con el siguiente real.
        real = np.flatnonzero(has_value)
        missing = np.flatnonzero(~has_value)
        numbers = np.fromiter(
            (to_float(v) for v in series), dtype=float, count=series.size
        )
        before = np.searchsorted(real, missing, side="left") - 1
        after = np.searchsorted(real, missing, side="right")
        left = np.where(before >= 0, numbers[real[np.maximum(before, 0)]], np.nan)
        right = np.where(
            after < real.size, numbers[real[np.minimum(after, real.size - 1)]], np.nan
        )

        estimate = np.full(missing.size, np.nan, dtype=float)
        both = np.isfinite(left) & np.isfinite(right)
        only_left = np.isfinite(left) & ~np.isfinite(right)
        only_right = ~np.isfinite(left) & np.isfinite(right)
        estimate[both] = (left[both] + right[both]) / 2.0
        estimate[only_left] = left[only_left]
        estimate[only_right] = right[only_right]

        filled_text = np.array(
            [format_imputed(x) if np.isfinite(x) else "" for x in estimate],
            dtype=object,
        )
        has_estimate = filled_text != ""
        series[missing] = filled_text
        imputed_mask[positions[missing]] = has_estimate
        unfilled += int((~has_estimate).sum())

        values[positions] = series

    # Conserva las marcas de imputaciones anteriores (p. ej. timestamps ya
    # reparados por fix_timestamps.py).
    if previous_flags is not None:
        imputed_mask |= previous_flags

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
    """Lee un CSV, imputa los huecos y lo escribe en output_dir."""
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

        previous_flags = None
        if FLAG_COL in df.columns:
            previous_flags = (
                pd.to_numeric(df[FLAG_COL], errors="coerce").fillna(0).to_numpy() != 0
            )
            df = df.drop(columns=[FLAG_COL])

        filled, imputed, unfilled, invalid = apply_locf(df, previous_flags)
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

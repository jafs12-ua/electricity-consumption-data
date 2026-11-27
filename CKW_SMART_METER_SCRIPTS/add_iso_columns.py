"""
Añade columnas ISO a archivos CSV.

Columnas añadidas:
  - weekday: Día de la semana en formato ISO
  - iso_week: Número de semana ISO
  - iso_year: Año ISO exacto
  - hora: Hora del día (0 a 23)
  - minutos: Minuto de la hora (0 a 59)

Nota: los timestamps del dataset están en hora local de Suiza (Europe/Zurich,
sin sufijo UTC), por lo que 'hora' se obtiene directamente, sin convertir zonas.
El cambio de horario se aprecia en los propios datos: el 2021-03-28 faltan las
franjas 02:00-02:45 y el 2021-10-31 están repetidas.

Uso:
    python3 add_iso_columns.py [ruta_archivo_o_directorio] [opciones]
"""

import sys
import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(path))]

"""Añade weekday (lunes = 1 - domingo = 7), iso_week (1-53), iso_year, hora (0-23) y minutos (0-59)"""
def add_temporal_columns(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    
    if timestamp_col not in df.columns:
        raise ValueError(f"No se encontró la columna '{timestamp_col}'")

    # Convertir a datetime; valores no válidos
    dt = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Saltar filas con timestamp no válido
    valid_mask = dt.notna()
    if not valid_mask.all():
        df = df[valid_mask].copy()
        dt = dt[valid_mask]

    iso = dt.dt.isocalendar()

    df["weekday"] = iso["day"].astype("int64")
    df["iso_week"] = iso["week"].astype("int64")
    df["iso_year"] = iso["year"].astype("int64")
    df["hora"] = dt.dt.hour.astype("int64")
    df["minutos"] = dt.dt.minute.astype("int64")
    return df

"""Lee el CSV, añade las columnas y guarda el resultado."""
def process_file(csv_path: Path, output_dir: Path | None = None, force: bool = False) -> tuple[bool, str]:
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / csv_path.name
    else:
        out_path = csv_path

    try:
        # Comprobar cabecera rápidamente sin cargar todo el archivo
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as fp:
            header = fp.readline()

        if "timestamp" not in header:
            return False, "Omitido (sin columna 'timestamp')"

        # Si ya contiene las columnas, se salta
        if "iso_year" in header and not force and output_dir is None:
            return True, "Ya procesado (omitido)"

        # Procesamiento por bloques
        chunk_size = 250000
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        first_chunk = True
        total_rows = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunk = add_temporal_columns(chunk)
            if chunk.empty:
                continue
            total_rows += len(chunk)
            chunk.to_csv(
                tmp_out,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )
            first_chunk = False

        if tmp_out.exists():
            tmp_out.replace(out_path)
            return True, f"OK ({total_rows:,} filas)"
        return False, "Omitido (sin filas válidas)"
    except Exception as e:
        if "tmp_out" in locals() and tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        return False, f"ERROR: {e}"


def _worker(args: tuple[Path, Path | None, bool]) -> tuple[bool, str]:
    csv_path, output_dir, force = args
    return process_file(csv_path, output_dir, force)


def main():
    parser = argparse.ArgumentParser(
        description="Añade columnas temporales e ISO a archivos CSV"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Ruta al archivo .csv o directorio a procesar"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directorio de destino"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Número de procesos en paralelo (por defecto: 4)."
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Forzar el reprocesamiento aunque el archivo ya tenga las columnas añadidas."
    )

    args = parser.parse_args()

    target_path = Path(args.path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not target_path.exists():
        print(f"Error: La ruta '{target_path}' no existe.")
        sys.exit(1)

    # Procesar un único archivo CSV
    if target_path.is_file():
        print(f"Procesando: {target_path.name} ... ", end="", flush=True)
        success, msg = process_file(target_path, output_dir, force=args.force)
        print(msg)
        sys.exit(0 if success else 1)

    # Buscar y ordenar archivos con orden numérico natural
    csv_files = sorted(
        [
            f for f in target_path.rglob("*.csv")
            if not f.name.endswith(".tmp") and not f.name.endswith(".bak")
        ],
        key=natural_sort_key
    )

    if not csv_files:
        print(f"No se encontraron archivos .csv en '{target_path}'.")
        sys.exit(0)

    print(f"Se encontraron {len(csv_files)} archivo(s) CSV para procesar.\n")

    converted = 0
    errors = 0

    if args.workers > 1 and len(csv_files) > 1:
        tasks = [(f, output_dir, args.force) for f in csv_files]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx, (f, (success, msg)) in enumerate(zip(csv_files, executor.map(_worker, tasks)), 1):
                print(f"[{idx}/{len(csv_files)}] Procesando: {f.name} ... {msg}", flush=True)
                if success:
                    converted += 1
                else:
                    errors += 1
    else:
        for idx, f in enumerate(csv_files, 1):
            print(f"[{idx}/{len(csv_files)}] Procesando: {f.name} ... ", end="", flush=True)
            success, msg = process_file(f, output_dir, force=args.force)
            print(msg, flush=True)
            if success:
                converted += 1
            else:
                errors += 1

    print(f"\nFinalizado: {converted} procesados con éxito, {errors} errores/omitidos.")


if __name__ == "__main__":
    main()

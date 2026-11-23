"""
Añade columnas ISO a los CSV de Elec-tariffs.ie:

    Columnas añadidas:
  - weekday: Día de la semana en formato ISO
  - iso_week: Número de semana ISO
  - iso_year: Año ISO exacto
  - hora: Hora del día (0 a 23)
  - minutos: Minuto de la hora (0 a 59)


Uso:
    python3 add_iso_columns.py [ruta] [-o directorio] [-f](para volver a procesar un archivo ya procesado)
"""

import argparse
from pathlib import Path

import pandas as pd

TIME_COL = "Read Date and End Time"
TIME_FORMATS = ("%d-%m-%Y %H:%M", "%d-%m-%y %H:%M")
NEW_COLUMNS = ["weekday", "iso_week", "iso_year", "hora", "minutos"]
CHUNK_SIZE = 250_000

"""
    Acepta fechas con guiones o barras y con año de 2 o 4 dígitos.
    Las filas con fecha no válida se descartan.
"""
def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    # Los CSV mezclan 'dd/mm/YYYY', 'dd-mm-YYYY' y años de dos dígitos
    dates = df[TIME_COL].str.replace("/", "-", regex=False)
    dt = pd.to_datetime(dates, format=TIME_FORMATS[0], errors="coerce")
    two_digit_year = dt.isna() & dates.notna()
    if two_digit_year.any():
        dt.loc[two_digit_year] = pd.to_datetime(
            dates[two_digit_year], format=TIME_FORMATS[1], errors="coerce"
        )
    valid = dt.notna()
    df = df[valid].copy()
    dt = dt[valid]

    iso = dt.dt.isocalendar()
    df["weekday"] = iso["day"].astype("int64")
    df["iso_week"] = iso["week"].astype("int64")
    df["iso_year"] = iso["year"].astype("int64")
    df["hora"] = dt.dt.hour.astype("int64")
    df["minutos"] = dt.dt.minute.astype("int64")
    return df


"""
    Añade las columnas temporales a un CSV, leyéndolo por bloques.

    Si no se indica output_dir, se modifica el archivo original usando un
    temporal para no dejarlo a medias si falla. Los archivos que ya tienen
    las columnas se omiten, salvo que se pase force.
"""
def process_file(csv_path: Path, output_dir: Path | None, force: bool) -> tuple[bool, str]:
    
    out_path = output_dir / csv_path.name if output_dir else csv_path

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        header = f.readline()

    if TIME_COL not in header:
        return False, "Omitido (sin columna de fecha)"

    if not force and output_dir is None and all(col in header for col in NEW_COLUMNS):
        return True, "Ya procesado (omitido)"

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    first_chunk = True
    total_rows = 0

    try:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            chunk = add_temporal_columns(chunk)
            if chunk.empty:
                continue
            total_rows += len(chunk)
            chunk.to_csv(
                tmp_out,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False

        if not tmp_out.exists():
            return False, "Omitido (sin filas válidas)"

        tmp_out.replace(out_path)
        return True, f"OK ({total_rows:,} filas)"
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        return False, f"ERROR: {e}"


def main() -> None:

    default_path = Path(__file__).resolve().parent.parent / "Elec-tariffs.ie"

    parser = argparse.ArgumentParser(description="Añade columnas ISO y temporales a los CSV")
    parser.add_argument("path", nargs="?", type=Path, default=default_path,
                        help="archivo .csv o carpeta a procesar (por defecto: ../Elec-tariffs.ie)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="directorio de salida (por defecto: modificar en sitio)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="reprocesar archivos que ya tienen las columnas")
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"No existe la ruta '{args.path}'")

    if args.path.is_file():
        files = [args.path]
    else:
        # Se ignoran los archivos '._*' que crea macOS en __MACOSX
        files = sorted(f for f in args.path.rglob("*.csv") if not f.name.startswith("._"))

    if not files:
        print(f"No se encontraron archivos .csv en '{args.path}'.")
        return

    print(f"Se encontraron {len(files)} archivo(s) CSV.\n")
    ok = errors = 0
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name} ... ", end="", flush=True)
        success, msg = process_file(path, args.output_dir, args.force)
        print(msg, flush=True)
        ok += success
        errors += not success

    print(f"\nFinalizado: {ok} procesados con éxito, {errors} omitidos/con errores.")


if __name__ == "__main__":
    main()

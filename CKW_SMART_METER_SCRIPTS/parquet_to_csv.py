"""
Convierte archivos .parquet a .csv

Uso:
    python3 parquet_to_csv.py [ruta_archivo_o_directorio] [opciones]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

def convert_file(parquet_path: Path, output_dir: Path | None = None, delete_source: bool = False):
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{parquet_path.stem}.csv"
    else:
        csv_path = parquet_path.with_suffix(".csv")

    print(f"Convirtiendo: {parquet_path} -> {csv_path} ...", end=" ", flush=True)
    try:
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        print(f"OK ({len(df):,} filas)")
        if delete_source:
            parquet_path.unlink()
            print(f"  [Eliminado original: {parquet_path}]")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convierte archivos .parquet a .csv")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Ruta al archivo .parquet o directorio a procesar"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directorio de destino para guardar los CSVs."
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Eliminar el archivo .parquet original tras convertirlo con éxito."
    )

    args = parser.parse_args()
    target_path = Path(args.path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not target_path.exists():
        print(f"Error: La ruta '{target_path}' no existe.")
        sys.exit(1)

    if target_path.is_file():
        if target_path.suffix.lower() != ".parquet":
            print(f"Aviso: '{target_path}' no tiene extensión .parquet, pero se intentará procesar.")
        success = convert_file(target_path, output_dir, args.delete_source)
        sys.exit(0 if success else 1)

    # Si es un directorio, buscar todos los archivos .parquet recursivamente
    parquet_files = sorted(list(target_path.rglob("*.parquet")))
    if not parquet_files:
        print(f"No se encontraron archivos .parquet en '{target_path}'.")
        sys.exit(0)

    print(f"Se encontraron {len(parquet_files)} archivo(s) .parquet para convertir.\n")
    converted = 0
    errors = 0

    for idx, p_file in enumerate(parquet_files, 1):
        print(f"[{idx}/{len(parquet_files)}] ", end="")
        if convert_file(p_file, output_dir, args.delete_source):
            converted += 1
        else:
            errors += 1

    print(f"\nFinalizado: {converted} convertidos con éxito, {errors} errores.")

if __name__ == "__main__":
    main()

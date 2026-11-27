"""
Agrupa los CSV de Elec-tariffs.ie en un CSV por usuario.

Genera ordered_files/<id>.csv con todas las filas de ese MPRN en todos los
CSV de entrada, ordenadas por 'Read Date and End Time' (más antiguo primero).
Los ficheros se recorren en streaming y se procesa un usuario a la vez, así
que no hace falta cargar los ~8 GB del dataset en memoria.

Uso:
    python3 order_by_user.py [opciones]

Ejemplos:
    python3 order_by_user.py --limit-files 3 --output-dir /tmp/order_elec_test
    python3 order_by_user.py --user-id 21
    python3 order_by_user.py --force
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd


# Sin nombres de carpeta fijos: --input-dir decide dónde están los CSV.
DEFAULT_INPUT_DIR = Path(".")
ID_COL = "MPRN_Anon"
TIME_COL = "Read Date and End Time"
DATE_FORMATS = ("%d-%m-%Y %H:%M", "%d-%m-%y %H:%M")
EXCLUDED_DIRS = {"ordered_files", "locf_files", "summary", "__MACOSX", "__pycache__"}
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
STALE_TMP = re.compile(r"\.[A-Za-z0-9_.-]+\.\d+\.tmp")


def natural_key(value: str) -> list[int | str]:
    """Ordena números de forma natural: 2, 10, 100."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrupa los CSV de Elec-tariffs.ie en un CSV por usuario"
    )
    parser.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR),
        help="Carpeta con los CSV. Por defecto la carpeta actual (.)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Carpeta de salida. Por defecto <input-dir>/ordered_files"
    )
    parser.add_argument(
        "--id-column", default=ID_COL,
        help=f"Columna con el identificador de usuario. Por defecto '{ID_COL}'"
    )
    parser.add_argument(
        "--time-column", default=TIME_COL,
        help=f"Columna de fecha a ordenar. Por defecto '{TIME_COL}'"
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=250_000,
        help="Filas por bloque al indexar usuarios. Por defecto 250000"
    )
    parser.add_argument(
        "--limit-files", type=int, default=None,
        help="Procesar solo los primeros N ficheros. Para pruebas"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rehacer ficheros de usuario existentes"
    )
    parser.add_argument(
        "--user-id", default=None,
        help="Regenerar solo el fichero de este usuario (por ejemplo 21)"
    )
    parser.add_argument(
        "--progress-every", type=int, default=None,
        help="Mostrar el progreso cada N usuarios (por defecto: cada 10%%)"
    )
    return parser.parse_args(argv)


def find_csv_files(
    input_dir: Path, output_dir: Path, limit: int | None = None,
) -> list[Path]:
    """Devuelve los CSV reales, ordenados por nombre, sin los de macOS/salida."""
    output_dir = output_dir.resolve()
    files = []
    for path in input_dir.rglob("*.csv"):
        if path.name.startswith(("._", ".")):
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if output_dir in path.resolve().parents:
            continue
        files.append(path)
    files.sort(key=lambda p: natural_key(p.name))
    return files[:limit] if limit is not None else files


def read_header(path: Path) -> list[str]:
    """Lee la cabecera de un CSV sin cargar el fichero."""
    with open(path, newline="", encoding="utf-8", errors="ignore") as fp:
        return next(csv.reader(fp), []) or []


def index_users(
    files: list[Path], id_col: str, chunk_rows: int, only_user: str | None,
) -> tuple[dict[str, list[Path]], set[str], int]:
    """Asocia cada usuario con los ficheros en los que aparece."""
    users: dict[str, list[Path]] = {}
    invalid: set[str] = set()
    failed = 0
    started = time.monotonic()
    last_step = 0

    for index, path in enumerate(files, 1):
        try:
            ids_in_file: set[str] = set()
            chunks = pd.read_csv(
                path,
                usecols=[id_col],
                dtype=str,
                keep_default_na=False,
                chunksize=chunk_rows,
            )
            for chunk in chunks:
                ids_in_file.update(chunk[id_col].unique().tolist())
        except Exception as exc:
            failed += 1
            print(f"Aviso: no se pudo indexar '{path}': {exc}", flush=True)
            continue

        for user_id in ids_in_file:
            if only_user is not None and user_id != only_user:
                continue
            if not SAFE_ID.fullmatch(user_id):
                invalid.add(user_id)
                continue
            users.setdefault(user_id, []).append(path)

        step = index * 10 // len(files)
        if step > last_step or index == len(files):
            last_step = step
            print(
                f"[índice {index}/{len(files)}] {len(users):,} usuarios "
                f"encontrados | {time.monotonic() - started:.0f}s",
                flush=True,
            )

    return users, invalid, failed


def sort_by_date(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, int]:
    """Ordena por fecha (formato dd-mm-YYYY / dd-mm-YY), antiguo primero."""
    dates = df[time_col].astype("string").str.strip()
    dates = dates.str.replace("/", "-", regex=False)
    parsed = pd.to_datetime(dates, format=DATE_FORMATS[0], errors="coerce")
    parsed = parsed.fillna(
        pd.to_datetime(dates, format=DATE_FORMATS[1], errors="coerce")
    )
    # Por si algún fichero usa otro formato: último intento permissivo.
    parsed = parsed.fillna(pd.to_datetime(dates, errors="coerce", dayfirst=True))
    invalid_dates = int(parsed.isna().sum() - dates.eq("").sum())
    df = df.assign(_sort_date=parsed).sort_values(
        "_sort_date", kind="mergesort", na_position="last"
    )
    return df.drop(columns="_sort_date"), max(invalid_dates, 0)


def write_user(
    user_id: str,
    paths: list[Path],
    header: list[str],
    id_col: str,
    time_col: str,
    output_dir: Path,
    pid: int,
) -> tuple[int, int]:
    """Escribe el CSV temporal de un usuario y devuelve (filas, fechas inválidas)."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = df[df[id_col] == user_id].reindex(columns=header)
        if not df.empty:
            frames.append(df)

    if not frames:
        return 0, 0

    df = pd.concat(frames, ignore_index=True)
    df, invalid_dates = sort_by_date(df, time_col)
    temp = output_dir / f".{user_id}.{pid}.tmp"
    try:
        df.to_csv(temp, index=False, lineterminator="\n")
        temp.replace(output_dir / f"{user_id}.csv")
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return len(df), invalid_dates


def prepare_output(output_dir: Path, force: bool, user_id: str | None) -> set[str]:
    """Prepara la carpeta de salida y devuelve los usuarios a omitir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(".*.tmp"):
        if STALE_TMP.fullmatch(path.name):
            path.unlink(missing_ok=True)

    skip_users = set() if force else {p.stem for p in output_dir.glob("*.csv")}
    if user_id:
        skip_users.discard(user_id)
    if skip_users and not user_id:
        word = "fichero" if len(skip_users) == 1 else "ficheros"
        print(
            f"Aviso: {len(skip_users):,} {word} de usuario ya existen y se omitirán. "
            "Usa --force para rehacerlos.",
            flush=True,
        )
    return skip_users


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir else input_dir / "ordered_files"
    )
    if not input_dir.is_dir():
        print(f"Error: la carpeta de entrada '{input_dir}' no existe.")
        return 1
    if args.chunk_rows < 1:
        print("Error: --chunk-rows debe ser mayor que 0.")
        return 1
    if args.limit_files is not None and args.limit_files < 1:
        print("Error: --limit-files debe ser mayor que 0.")
        return 1
    if args.progress_every is not None and args.progress_every < 1:
        print("Error: --progress-every debe ser mayor que 0.")
        return 1
    if args.user_id is not None and not SAFE_ID.fullmatch(args.user_id):
        print(f"Error: el id '{args.user_id}' no es válido.")
        return 1

    files = find_csv_files(input_dir, output_dir, args.limit_files)
    if not files:
        print(f"No se encontraron ficheros .csv en '{input_dir}'.")
        return 1

    header = read_header(files[0])
    for column in (args.id_column, args.time_column):
        if column not in header:
            print(f"Error: la columna '{column}' no está en '{files[0]}'.")
            return 1

    print(
        f"Ficheros CSV a procesar: {len(files):,} desde '{input_dir}'.",
        flush=True,
    )
    skip_users = prepare_output(output_dir, args.force, args.user_id)

    users, invalid, failed = index_users(
        files, args.id_column, args.chunk_rows, args.user_id
    )
    if args.user_id and args.user_id not in users:
        print(f"Error: el usuario {args.user_id} no aparece en los CSV de entrada.")
        return 1

    pending = [u for u in users if u not in skip_users]
    pending.sort(key=natural_key)
    if not pending:
        print("No hay usuarios nuevos que procesar.")
        return 0

    step_every = args.progress_every or max(1, len(pending) * 10 // 100)
    print(
        f"Usuarios a escribir: {len(pending):,} de {len(users):,} encontrados.",
        flush=True,
    )

    pid = os.getpid()
    total_rows = 0
    total_bad_dates = 0
    started = time.monotonic()

    for index, user_id in enumerate(pending, 1):
        rows, bad_dates = write_user(
            user_id, users[user_id], header, args.id_column, args.time_column,
            output_dir, pid,
        )
        total_rows += rows
        total_bad_dates += bad_dates
        if index % step_every == 0 or index == len(pending):
            print(
                f"[{index}/{len(pending)}] {user_id}.csv | {rows:,} filas | "
                f"{total_rows:,} filas acumuladas | {time.monotonic() - started:.0f}s",
                flush=True,
            )

    word = "fichero" if len(pending) == 1 else "ficheros"
    print(
        f"\nFinalizado: {len(pending):,} {word} de usuario en '{output_dir}', "
        f"{total_rows:,} filas de {len(files):,} CSV. "
        f"{len(skip_users):,} usuarios existentes omitidos, "
        f"{failed} ficheros con error al indexar, "
        f"{len(invalid)} ids no válidos, "
        f"{total_bad_dates:,} fechas no válidas (colocadas al final), "
        f"{time.monotonic() - started:.0f}s."
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

"""
Agrupa los ficheros diarios de CKW_SMART_METER en un CSV por usuario.

Genera ordered_files/<id>.csv con todas las filas de cada id.
Los días se procesan en orden y los ids ya vienen agrupados, así que
cada fichero queda cronológico sin necesidad de ordenar.

Uso:
    python3 order_by_user.py [opciones]

Ejemplos:
    python3 order_by_user.py --limit-files 2 --output-dir /tmp/order_test
    python3 order_by_user.py --years 2021 2022
    python3 order_by_user.py --force
    python3 order_by_user.py --user-id 768c2f018fbbe3910adb700128f8e506
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "CKW_SMART_METER"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "ordered_files"
EXCLUDED_DIRS = {"ordered_files", "summary", "__pycache__"}
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
STALE_TMP = re.compile(r"\.[A-Za-z0-9_.-]+\.\d+\.tmp")


def natural_sort_key(path: Path) -> list[int | str]:
    """Ordena números de forma natural: 1, 2, 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrupa los ficheros diarios por usuario"
    )
    parser.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR),
        help=f"Carpeta de entrada. Por defecto {DEFAULT_INPUT_DIR}"
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Carpeta de salida. Por defecto {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--years", nargs="*", default=None,
        help="Años a procesar, por ejemplo 2021 2022. Por defecto todos"
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=500_000,
        help="Filas por bloque. Por defecto 500000"
    )
    parser.add_argument(
        "--limit-files", type=int, default=None,
        help="Procesar solo los primeros N días. Para pruebas"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rehacer ficheros de usuario existentes"
    )
    parser.add_argument(
        "--user-id", default=None,
        help="Regenerar solo el fichero de este id"
    )
    parser.add_argument(
        "--progress-every", type=int, default=1,
        help="Mostrar el progreso cada N días. Por defecto 1"
    )
    return parser.parse_args(argv)


def find_day_files(input_dir: Path, years: list[str] | None) -> list[Path]:
    """Devuelve los ficheros diarios ordenados por año y día."""
    if years:
        directories = [input_dir / name for name in years]
    else:
        directories = [
            path for path in input_dir.iterdir()
            if path.is_dir() and path.name not in EXCLUDED_DIRS
        ]

    files: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            print(f"Aviso: el directorio '{directory}' no existe, se omite.")
            continue
        files.extend(
            path for path in directory.glob("*.csv")
            if path.stem.isdigit() and path.is_file()
        )

    return sorted(files, key=natural_sort_key)


def consecutive_user_groups(chunk: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
    """Devuelve los bloques de filas de cada id."""
    ids = chunk["id"].to_numpy(dtype=object)

    # Los ids ya están juntos: solo cortamos donde cambia el id.
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(ids)]
    rows = chunk.to_numpy(dtype=object)

    for start, end in zip(starts, ends):
        yield str(ids[start]), rows[start:end]


class UserWriters:
    """Temporales de usuario abiertos durante toda la ejecución."""

    def __init__(self, output_dir: Path, pid: int, skip_users: set[str]):
        self.output_dir = output_dir
        self.pid = pid
        self.skip_users = skip_users
        self.header: list[str] | None = None
        self.handles: dict[str, object] = {}
        self.csv_writers: dict[str, csv.writer] = {}
        self.written_users: set[str] = set()
        self.skipped_users: set[str] = set()
        self.invalid_users: set[str] = set()

    def _temp_path(self, user_id: str) -> Path:
        return self.output_dir / f".{user_id}.{self.pid}.tmp"

    def write_group(self, user_id: str, rows: np.ndarray) -> None:
        if not SAFE_ID.fullmatch(user_id):
            self.invalid_users.add(str(user_id))
            return
        if user_id in self.skip_users:
            self.skipped_users.add(user_id)
            return

        fp = self.handles.get(user_id)
        if fp is None:
            # Un fichero abierto por usuario, en vez de uno por día.
            fp = open(self._temp_path(user_id), "w", newline="", encoding="utf-8")
            self.handles[user_id] = fp
            fp.write(",".join(self.header) + "\n")

        separators = len(self.header) - 1
        for row in rows:
            line = ",".join(row)
            # Join directo solo si no hay comas, comillas ni saltos de línea.
            if (
                line.count(",") == separators
                and '"' not in line and "\n" not in line and "\r" not in line
            ):
                fp.write(line)
                fp.write("\n")
            else:
                writer = self.csv_writers.get(user_id)
                if writer is None:
                    writer = csv.writer(fp, lineterminator="\n")
                    self.csv_writers[user_id] = writer
                writer.writerow(row)
        self.written_users.add(user_id)

    def close_handles(self) -> None:
        for fp in self.handles.values():
            fp.close()

    def commit(self) -> None:
        """Cierra y renombra los temporales al terminar."""
        self.close_handles()
        for user_id in self.written_users:
            self._temp_path(user_id).replace(self.output_dir / f"{user_id}.csv")

    def abort(self) -> None:
        """Borra los temporales si falla la ejecución."""
        self.close_handles()
        for user_id in self.written_users:
            self._temp_path(user_id).unlink(missing_ok=True)


def process_day(path: Path, writers: UserWriters, chunk_rows: int) -> int:
    """Lee un fichero diario por bloques y devuelve las filas leídas."""
    rows_read = 0
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_rows,
        dtype=str,
        keep_default_na=False,
        memory_map=True,
    ):
        rows_read += len(chunk)
        if writers.header is None:
            writers.header = list(chunk.columns)
            if "id" not in writers.header:
                raise ValueError(f"La columna 'id' no está en '{path}'")
        for user_id, rows in consecutive_user_groups(chunk):
            writers.write_group(user_id, rows)
    return rows_read


def prepare_output(
    output_dir: Path, force: bool, regenerate_user: str | None = None,
) -> tuple[Path, set[str]]:
    """Prepara la salida y devuelve el pid y los usuarios a omitir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Borra temporales viejos. No se admiten dos instancias a la vez.
    stale = [p for p in output_dir.glob(".*.tmp") if STALE_TMP.fullmatch(p.name)]
    for path in stale:
        path.unlink(missing_ok=True)

    pid = os.getpid()
    # Los ficheros ya presentes se dan por terminados, salvo con --force.
    skip_users = set() if force else {p.stem for p in output_dir.glob("*.csv")}
    if regenerate_user:
        skip_users.discard(regenerate_user)
    if skip_users and not regenerate_user:
        word = "fichero" if len(skip_users) == 1 else "ficheros"
        print(
            f"Aviso: {len(skip_users):,} {word} de usuario ya existen y se omitirán. "
            "Usa --force para rehacerlos.",
            flush=True,
        )
    return pid, skip_users


def progress_line(
    index: int,
    total: int,
    path: Path,
    year_progress: Counter,
    year_totals: Counter,
    total_rows: int,
    writers: UserWriters,
    elapsed: float,
) -> str:
    """Línea de progreso con el avance por año."""
    detail = " | ".join(
        f"{name}: {year_progress[name]}/{count}"
        for name, count in year_totals.items()
    )
    return (
        f"[{index}/{total}] {path.parent.name}/{path.name} | {detail} | "
        f"{total_rows:,} filas leídas | {len(writers.written_users):,} usuarios | "
        f"{elapsed:.0f}s"
    )


def read_user_rows(path: Path, user_id: str) -> list[str]:
    """Filas de un usuario en un día, sin leer el CSV completo."""
    parquet = path.with_suffix(".parquet")
    if parquet.is_file():
        table = pq.read_table(
            parquet,
            columns=["id", "timestamp", "value_kwh"],
            filters=[("id", "=", user_id)],
        )
        df = table.to_pandas()
        if df.empty:
            return []
        dt = pd.to_datetime(df["timestamp"])
        iso = dt.dt.isocalendar()
        df["weekday"] = iso["day"].astype(int)
        df["iso_week"] = iso["week"].astype(int)
        df["iso_year"] = iso["year"].astype(int)
        df["hora"] = dt.dt.hour.astype(int)
        df["minutos"] = dt.dt.minute.astype(int)
        columns = [
            "id", "timestamp", "value_kwh",
            "weekday", "iso_week", "iso_year", "hora", "minutos",
        ]
        return df[columns].to_csv(
            index=False, header=False, lineterminator="\n"
        ).splitlines()

    # Sin parquet: busca solo las líneas del id en el CSV.
    prefix = f"{user_id},"
    result = subprocess.run(
        ["grep", "-F", prefix, str(path)],
        capture_output=True, text=True, check=False,
    )
    return [line for line in result.stdout.splitlines() if line.startswith(prefix)]


def regenerate_user(
    files: list[Path],
    output_dir: Path,
    user_id: str,
    year_totals: Counter,
    pid: int,
    progress_every: int,
) -> int:
    """Rehace solo el fichero de un usuario a partir de los ficheros diarios."""
    temp = output_dir / f".{user_id}.{pid}.tmp"
    header = files[0].open(encoding="utf-8").readline().rstrip("\n")
    year_progress = Counter()
    rows = 0
    started = time.monotonic()

    try:
        with open(temp, "w", newline="", encoding="utf-8") as fp:
            for index, path in enumerate(files, 1):
                lines = read_user_rows(path, user_id)
                if lines:
                    if not rows:
                        fp.write(header + "\n")
                    fp.write("".join(line + "\n" for line in lines))
                    rows += len(lines)
                year_progress[path.parent.name] += 1
                if index % progress_every == 0 or index == len(files):
                    detail = " | ".join(
                        f"{name}: {year_progress[name]}/{count}"
                        for name, count in year_totals.items()
                    )
                    print(
                        f"[{index}/{len(files)}] {path.parent.name}/{path.name} | "
                        f"{detail} | {rows:,} filas del usuario | "
                        f"{time.monotonic() - started:.0f}s",
                        flush=True,
                    )

        if not rows:
            temp.unlink(missing_ok=True)
            print(f"Error: el usuario {user_id} no aparece en los ficheros diarios.")
            return 1
        temp.replace(output_dir / f"{user_id}.csv")
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    print(
        f"\nFinalizado: {user_id}.csv en '{output_dir}', {rows:,} filas de "
        f"{len(files)} ficheros diarios, {time.monotonic() - started:.0f}s."
    )
    return 0


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"Error: la carpeta de entrada '{input_dir}' no existe.")
        return 1
    if args.chunk_rows < 1:
        print("Error: --chunk-rows debe ser mayor que 0.")
        return 1
    if args.progress_every < 1:
        print("Error: --progress-every debe ser mayor que 0.")
        return 1
    if args.user_id is not None and not SAFE_ID.fullmatch(args.user_id):
        print(f"Error: el id '{args.user_id}' no es válido.")
        return 1

    files = find_day_files(input_dir, args.years)
    if not files:
        print(f"No se encontraron ficheros .csv diarios en '{input_dir}'.")
        return 1
    if args.limit_files is not None:
        files = files[:args.limit_files]
    if not files:
        print("No hay ficheros que procesar.")
        return 1

    year_totals = Counter(path.parent.name for path in files)
    year_progress = Counter()
    years_detail = ", ".join(
        f"{year}: {count}" for year, count in year_totals.items()
    )
    year_word = "año" if len(year_totals) == 1 else "años"
    print(
        f"Ficheros diarios a procesar: {len(files)} en {len(year_totals)} {year_word}. "
        f"Detalle: {years_detail}.",
        flush=True,
    )

    pid, skip_users = prepare_output(output_dir, args.force, args.user_id)
    if args.user_id:
        print(f"Regenerando solo el usuario {args.user_id}.", flush=True)
        return regenerate_user(
            files, output_dir, args.user_id, year_totals, pid, args.progress_every
        )
    writers = UserWriters(output_dir, pid, skip_users)
    total_rows = 0
    started = time.monotonic()

    try:
        # Días en orden: cada usuario queda en orden cronológico.
        for index, path in enumerate(files, 1):
            total_rows += process_day(path, writers, args.chunk_rows)
            year_progress[path.parent.name] += 1
            if index % args.progress_every == 0 or index == len(files):
                print(
                    progress_line(
                        index, len(files), path, year_progress, year_totals,
                        total_rows, writers, time.monotonic() - started,
                    ),
                    flush=True,
                )
        # Renombra solo al final para no dejar ficheros a medias.
        writers.commit()
    except BaseException:
        writers.abort()
        raise

    detail = ", ".join(
        f"{name}: {year_progress[name]}/{count}"
        for name, count in year_totals.items()
    )
    elapsed = time.monotonic() - started
    print(
        f"\nFinalizado: {len(writers.written_users):,} ficheros de usuario en '{output_dir}', "
        f"{total_rows:,} filas de {len(files)} ficheros diarios. Por años: {detail}, "
        f"{len(writers.skipped_users):,} usuarios ya existentes omitidos, "
        f"{len(writers.invalid_users):,} ids no válidos, {elapsed:.0f}s."
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())

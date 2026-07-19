import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DATABASE_PATH


def _table_name(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"Invalid SQLite table name: {value}")
    return value


def connect_database(path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            source TEXT,
            recorded_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT,
            dataset_fingerprint TEXT,
            trained_at TEXT,
            metrics_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


@contextmanager
def database_connection(path: str | Path = DATABASE_PATH):
    connection = connect_database(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _sqlite_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype == "object":
            output[column] = output[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
    return output


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    stable = _sqlite_frame(frame).fillna("").astype(str)
    stable = stable.sort_values(list(stable.columns)).reset_index(drop=True)
    values = pd.util.hash_pandas_object(stable, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def sync_dataframe(
    table: str,
    frame: pd.DataFrame,
    source: str = "",
    path: str | Path = DATABASE_PATH,
) -> None:
    table = _table_name(table)
    output = _sqlite_frame(frame)
    if len(output.columns) == 0:
        output = pd.DataFrame({"_empty": pd.Series(dtype=str)})
    fingerprint = dataframe_fingerprint(output)
    with database_connection(path) as connection:
        output.to_sql(table, connection, if_exists="replace", index=False)
        connection.execute(
            """
            INSERT INTO dataset_versions (table_name, row_count, fingerprint, source, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                table,
                len(output),
                fingerprint,
                source,
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )


def sync_match_data(matches: pd.DataFrame, source: str = "vlr_matches.csv") -> None:
    sync_dataframe("player_map_stats", matches, source=source)
    if matches.empty:
        sync_dataframe("maps", pd.DataFrame(), source=source)
        sync_dataframe("matches", pd.DataFrame(), source=source)
        return

    match_columns = [
        column
        for column in [
            "match_id",
            "match_url",
            "match_date",
            "season_year",
            "event_id",
            "event_name",
            "event_series",
            "event_stage",
            "event_tier",
            "event_region",
            "is_lan",
            "patch",
            "patch_source",
            "map_veto",
            "match_importance",
            "competition_tier",
            "competition_strength_weight",
            "team_tier_at_match",
            "opponent_tier_at_match",
            "team_promoted_next_season",
            "opponent_promoted_next_season",
            "data_source",
            "team",
            "team_id",
            "team_region",
            "opponent",
            "opponent_id",
            "opponent_region",
            "winner",
            "winner_id",
            "team_score",
            "opp_score",
        ]
        if column in matches.columns
    ]
    map_columns = [
        column
        for column in [
            "match_id",
            "map_id",
            "map_number",
            "map_name",
            "map_pick_team",
            "map_pick_type",
            "map_veto_order",
            "map_data_source",
            "team",
            "opponent",
            "map_team_score",
            "map_opp_score",
        ]
        if column in matches.columns
    ]
    match_rows = matches[match_columns].drop_duplicates(
        subset=[column for column in ["match_id", "team"] if column in match_columns],
        keep="last",
    )
    map_rows = matches[map_columns].drop_duplicates(
        subset=[column for column in ["match_id", "map_id", "team"] if column in map_columns],
        keep="last",
    )
    sync_dataframe("matches", match_rows, source=source)
    sync_dataframe("maps", map_rows, source=source)


def load_table(table: str, path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    table = _table_name(table)
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        with database_connection(path) as connection:
            return pd.read_sql_query(f"SELECT * FROM {table}", connection)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return pd.DataFrame()


def record_model_run(metadata: dict, path: str | Path = DATABASE_PATH) -> None:
    with database_connection(path) as connection:
        connection.execute(
            """
            INSERT INTO model_runs (
                model_version, dataset_fingerprint, trained_at, metrics_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                metadata.get("model_version"),
                metadata.get("dataset_fingerprint"),
                metadata.get("trained_at"),
                json.dumps(metadata, sort_keys=True, default=str),
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )

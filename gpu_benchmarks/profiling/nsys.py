import sqlite3

import pandas as pd


def table_names(connection: sqlite3.Connection) -> set[str]:
    """Return the tables available in an Nsight Systems SQLite export."""
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", connection)
    return set(tables["name"].astype(str))


def load_nvtx_ranges(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load complete NVTX ranges across the schemas emitted by supported NSYS versions."""
    available = table_names(connection)
    for table in ("NVTX_EVENTS", "NVTX_RANGES"):
        if table not in available:
            continue
        try:
            ranges = pd.read_sql_query(
                f"""
                SELECT start, end, text AS name
                FROM {table}
                WHERE end IS NOT NULL AND text IS NOT NULL AND text != ''
                ORDER BY start
                """,
                connection,
            )
        except (pd.errors.DatabaseError, sqlite3.DatabaseError):
            continue
        if not ranges.empty:
            return ranges
    return pd.DataFrame(columns=["start", "end", "name"])


def filter_by_nvtx(
    frame: pd.DataFrame,
    ranges: pd.DataFrame,
    include_pattern: str,
    *,
    timestamp_column: str,
    exclude_pattern: str | None = None,
) -> pd.DataFrame:
    """Select samples whose timestamps fall inside matching NVTX ranges."""
    included = ranges[ranges["name"].str.contains(include_pattern, regex=True, na=False)]
    if included.empty:
        return frame.copy()

    mask = pd.Series(False, index=frame.index)
    for row in included.itertuples(index=False):
        mask |= (frame[timestamp_column] >= row.start) & (frame[timestamp_column] <= row.end)

    if exclude_pattern:
        excluded = ranges[ranges["name"].str.contains(exclude_pattern, regex=True, na=False)]
        for row in excluded.itertuples(index=False):
            mask &= ~((frame[timestamp_column] >= row.start) & (frame[timestamp_column] <= row.end))
    return frame[mask].copy()


def classify_by_range(
    frame: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    suffix: str = ":case",
    timestamp_column: str,
    label_segment: int = 2,
) -> pd.Series:
    """Label samples using a colon-delimited segment from enclosing NVTX ranges."""
    matching = ranges[ranges["name"].str.endswith(suffix, na=False)]
    labels = pd.Series(index=frame.index, dtype="object")
    for row in matching.itertuples(index=False):
        mask = (frame[timestamp_column] >= row.start) & (frame[timestamp_column] <= row.end)
        segments = str(row.name).split(":")
        if label_segment < len(segments):
            labels.loc[mask] = segments[label_segment]
    return labels

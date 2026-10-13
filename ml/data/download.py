"""Stage data.download: fetch and validate the 5-core ratings CSV (CLAUDE.md s7.1, D19).

The only raw file the pipeline keeps: 0.14 GB, read by every filter, stored under ``data/`` so it
survives Colab restarts. It is always fetched whole, even in ``--sample`` runs, because it is the
item list everything else is filtered against. The large sources are streamed by the stages that
read them (``url:`` inputs) and never stored.
"""

from collections import Counter
from datetime import UTC, datetime

from ml.common.errors import DataError
from ml.common.stage import StageContext, stage
from ml.data.stream import StreamSettings, StreamStats, iter_lines

RATINGS_HEADER = ["user_id", "parent_asin", "rating", "timestamp"]
TS_MS_RANGE = (8e11, 1.8e12)  # CLAUDE.md s6.1: timestamps are ms (1995-2027); seconds fail this
VALID_STARS = {"1.0", "2.0", "3.0", "4.0", "5.0"}


@stage(
    "data.download",
    inputs=["url:ratings_5core"],
    outputs=["data:ratings_5core.csv.gz"],
    config_keys=["data.stream"],
)
def download_ratings(ctx: StageContext) -> None:
    """Copy the 5-core CSV byte for byte while checking its header, stars and ms timestamps."""
    url = ctx.source("url:ratings_5core", full=True)
    stats = StreamStats(url)
    users: set[str] = set()
    items: set[str] = set()
    stars: Counter[str] = Counter()
    ts_min, ts_max = float("inf"), float("-inf")
    with open(ctx.out("data:ratings_5core.csv.gz"), "wb") as out:
        lines = iter_lines(
            url, StreamSettings.from_config(ctx.param("data.stream")), stats=stats, tee=out
        )
        header = next(lines, "").split(",")
        if header != RATINGS_HEADER:
            raise DataError(f"5-core header is {header}, expected {RATINGS_HEADER}")
        for n, line in enumerate(lines, start=2):
            fields = line.split(",")
            if len(fields) != 4 or fields[2] not in VALID_STARS:
                raise DataError(f"5-core line {n} is malformed: {line[:120]!r}")
            ts = int(fields[3])
            if not TS_MS_RANGE[0] < ts < TS_MS_RANGE[1]:
                raise DataError(f"5-core line {n}: timestamp {ts} is not in milliseconds")
            users.add(fields[0])
            items.add(fields[1])
            stars[fields[2]] += 1
            ts_min, ts_max = min(ts_min, ts), max(ts_max, ts)
    ctx.count("ratings", stats.lines - 1)
    ctx.count("users", len(users))
    ctx.count("items", len(items))
    ctx.notes["source"] = stats.as_note()
    ctx.notes["stars"] = dict(sorted(stars.items()))
    ctx.notes["time_range"] = [_iso(ts_min), _iso(ts_max)] if items else []


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")

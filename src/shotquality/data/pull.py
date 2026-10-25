"""Download shot-level data, with a cache and a fallback.

Preferred source is ``nba_api``'s ``ShotChartDetail`` endpoint (one file per
team per season, with a pause between calls). From this environment
stats.nba.com timed out, so the default cache is the public SportsDataverse
mirror of the same NBA stats feed:

    https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nba_stats_shots

Those parquet files are regular-season and postseason shots. We keep the
regular season (``season_type_id == 2``, game id prefix ``002``).
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

from shotquality.config import DATA_RAW, MIRROR_END_YEARS, MIRROR_URL
from shotquality.data.clean import parse_clock, season_label_from_game_id

logger = logging.getLogger(__name__)

# Columns every downstream step can count on, no matter which source loaded.
COMMON_COLUMNS = [
    "game_id",
    "season",
    "period",
    "seconds_left",
    "team_id",
    "team_tricode",
    "player_id",
    "player_name_raw",
    "action_raw",
    "shot_made",
    "shot_value",
    "shot_distance",
    "loc_x",
    "loc_y",
    "data_source",
]


def mirror_path(end_year: int, raw_dir: Path | None = None) -> Path:
    return (raw_dir or DATA_RAW) / f"shots_{end_year}.parquet"


def mirror_files_ready(raw_dir: Path | None = None) -> bool:
    return all(mirror_path(year, raw_dir).exists() and mirror_path(year, raw_dir).stat().st_size > 0 for year in MIRROR_END_YEARS)


def _download(url: str, dest: Path, retries: int = 3) -> None:
    """Download ``url`` to ``dest``. A failed attempt never leaves a half file in place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "shotquality-portfolio"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"Download was empty: {url}")
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors vary; retry then raise the last one
            last_error = exc
            logger.warning("Download attempt %s failed for %s: %s", attempt, url, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}") from last_error


def download_mirror(raw_dir: Path | None = None, refresh: bool = False) -> list[Path]:
    """Cache the three SportsDataverse shot files under ``data/raw``."""
    raw_dir = raw_dir or DATA_RAW
    paths: list[Path] = []
    for end_year in MIRROR_END_YEARS:
        dest = mirror_path(end_year, raw_dir)
        if dest.exists() and dest.stat().st_size > 0 and not refresh:
            logger.info("Using cached %s", dest.name)
        else:
            url = MIRROR_URL.format(end_year=end_year)
            logger.info("Downloading %s", url)
            _download(url, dest)
        paths.append(dest)
    return paths


def probe_nba_api(timeout: int = 20) -> bool:
    """Return True if ShotChartDetail answers. One cheap call, not the full pull."""
    try:
        from nba_api.stats.endpoints import shotchartdetail

        shotchartdetail.ShotChartDetail(
            team_id=1610612738,  # Boston, arbitrary probe target
            player_id=0,
            season_nullable="2024-25",
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
            timeout=timeout,
        )
        return True
    except Exception as exc:
        logger.warning("stats.nba.com did not answer (%s: %s). Using the mirror.", type(exc).__name__, exc)
        return False


def pull_shotchart_api(
    seasons: tuple[str, ...] = ("2023-24", "2024-25", "2025-26"),
    raw_dir: Path | None = None,
    sleep_seconds: float = 0.8,
    timeout: int = 60,
) -> pd.DataFrame:
    """Pull regular-season shot charts for every team, caching each response as CSV.

    ``player_id=0`` asks the endpoint for every player on that team. Skipping a
    team that is already cached makes a rerun safe if a later call times out.
    """
    from nba_api.stats.endpoints import shotchartdetail
    from nba_api.stats.static import teams

    raw_dir = raw_dir or DATA_RAW
    cache_dir = raw_dir / "nba_api"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        for team in teams.get_teams():
            cache = cache_dir / f"shotchart_{season}_{team['id']}.csv"
            if cache.exists() and cache.stat().st_size > 0:
                frames.append(pd.read_csv(cache, dtype={"GAME_ID": str}))
                continue
            detail = shotchartdetail.ShotChartDetail(
                team_id=team["id"],
                player_id=0,
                season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",
                timeout=timeout,
            )
            frame = detail.get_data_frames()[0]
            frame.to_csv(cache, index=False)
            frames.append(frame)
            time.sleep(sleep_seconds)
    if not frames:
        raise RuntimeError("nba_api returned no shot charts.")
    raw = pd.concat(frames, ignore_index=True)
    return shotchart_detail_to_common(raw)


def shotchart_detail_to_common(frame: pd.DataFrame) -> pd.DataFrame:
    """Map a ShotChartDetail table onto the same columns the mirror uses.

    The endpoint's ``ACTION_TYPE`` is the shot description (layup, dunk, ...).
    ``EVENT_TYPE`` and ``SHOT_MADE_FLAG`` are the outcome and are not copied
    into a feature column. ``SHOT_MADE_FLAG`` becomes the label ``shot_made``.
    """
    required = {
        "GAME_ID",
        "TEAM_ID",
        "PLAYER_ID",
        "PLAYER_NAME",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "ACTION_TYPE",
        "SHOT_TYPE",
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "SHOT_MADE_FLAG",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ShotChartDetail frame is missing columns: {sorted(missing)}")

    game_id = frame["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    shot_type = frame["SHOT_TYPE"].astype(str)
    seconds_left = frame["MINUTES_REMAINING"].astype(float) * 60.0 + frame["SECONDS_REMAINING"].astype(float)
    team_name = frame["TEAM_NAME"].astype(str) if "TEAM_NAME" in frame.columns else ""
    out = pd.DataFrame(
        {
            "game_id": game_id,
            "season": game_id.map(season_label_from_game_id),
            "period": frame["PERIOD"].astype(int),
            "seconds_left": seconds_left,
            "team_id": frame["TEAM_ID"].astype(int),
            "team_tricode": team_name,
            "player_id": frame["PLAYER_ID"].astype(int),
            "player_name_raw": frame["PLAYER_NAME"].astype(str),
            "action_raw": frame["ACTION_TYPE"].astype(str),
            "shot_made": frame["SHOT_MADE_FLAG"].astype(int),
            "shot_value": shot_type.str.startswith("3").map({True: 3, False: 2}).astype(int),
            "shot_distance": frame["SHOT_DISTANCE"].astype(float),
            "loc_x": frame["LOC_X"].astype(float),
            "loc_y": frame["LOC_Y"].astype(float),
            "data_source": "nba_api",
        }
    )
    return out.loc[:, COMMON_COLUMNS]


def mirror_to_common(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season shots and rename the mirror's columns.

    The mirror's ``action_type`` column is "Made Shot" / "Missed Shot". That is
    the outcome, so we drop it. The shot description lives in ``sub_type``.
    ``description`` also contains the word MISS, so it never becomes a feature.
    The score columns are only filled in on makes and include free throws that
    happened around the shot, so they are not a clean pre-shot score. Left out
    on purpose.
    """
    regular = frame.loc[frame["season_type_id"].astype(str) == "2"].copy()
    regular = regular.loc[regular["game_id"].astype(str).str.startswith("002")].copy()
    if regular.empty:
        raise ValueError("Mirror file had no regular-season shots.")

    game_id = regular["game_id"].astype(str).str.zfill(10)
    season = game_id.map(season_label_from_game_id)
    # The file's `season` field is the ending year (2024 means 2023-24).
    # Check it against the game id so a mislabeled file fails loudly.
    file_end = regular["season"].astype(int)
    id_end = season.str.slice(0, 4).astype(int) + 1
    if not file_end.reset_index(drop=True).equals(id_end.reset_index(drop=True)):
        raise ValueError("Mirror season field does not match the game id. Refusing to guess.")

    out = pd.DataFrame(
        {
            "game_id": game_id.to_numpy(),
            "season": season.to_numpy(),
            "period": regular["period"].astype(int).to_numpy(),
            "seconds_left": regular["clock"].map(parse_clock).to_numpy(),
            "team_id": regular["team_id"].astype(int).to_numpy(),
            "team_tricode": regular["team_tricode"].astype(str).to_numpy(),
            "player_id": regular["person_id"].astype(int).to_numpy(),
            "player_name_raw": regular["player_name"].astype(str).to_numpy(),
            "action_raw": regular["sub_type"].astype(str).to_numpy(),
            "shot_made": (regular["shot_result"].astype(str) == "Made").astype(int).to_numpy(),
            "shot_value": regular["shot_value"].astype(int).to_numpy(),
            "shot_distance": regular["shot_distance"].astype(float).to_numpy(),
            "loc_x": regular["x_legacy"].astype(float).to_numpy(),
            "loc_y": regular["y_legacy"].astype(float).to_numpy(),
            "data_source": "sportsdataverse",
        }
    )
    return out.loc[:, COMMON_COLUMNS]


def load_mirror_frame(raw_dir: Path | None = None) -> pd.DataFrame:
    paths = [mirror_path(year, raw_dir) for year in MIRROR_END_YEARS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing mirror files: {missing}. Run scripts/pull_shots.py.")
    frames = [mirror_to_common(pd.read_parquet(path)) for path in paths]
    return pd.concat(frames, ignore_index=True)


def ensure_raw_files(source: str = "auto", refresh: bool = False, raw_dir: Path | None = None) -> str:
    """Make sure raw files exist. Return which source will be modeled.

    ``auto`` uses the cache if it is already there. Otherwise it tries
    stats.nba.com once, and on failure downloads the mirror.
    """
    raw_dir = raw_dir or DATA_RAW
    if source not in {"auto", "api", "mirror"}:
        raise ValueError("source must be auto, api, or mirror")

    if source == "mirror":
        download_mirror(raw_dir, refresh=refresh)
        return "sportsdataverse"

    if source == "auto" and mirror_files_ready(raw_dir) and not refresh:
        return "sportsdataverse"

    # A full API pull is 90 calls. Probe once before starting it.
    if source in {"api", "auto"} and probe_nba_api():
        pull_shotchart_api(raw_dir=raw_dir)
        return "nba_api"
    if source == "api":
        raise RuntimeError("stats.nba.com did not answer. Re-run with --source mirror.")

    download_mirror(raw_dir, refresh=refresh)
    return "sportsdataverse"


def load_common_frame(source: str = "auto", raw_dir: Path | None = None) -> pd.DataFrame:
    """Return every regular-season shot in the common column layout.

    If an API cache exists and ``source='api'``, that cache is used. The
    shipped project uses the mirror, which is what ``auto`` loads when the
    parquet files are present.
    """
    raw_dir = raw_dir or DATA_RAW
    if source == "api":
        cache = raw_dir / "nba_api"
        files = sorted(cache.glob("shotchart_*.csv")) if cache.exists() else []
        if not files:
            raise FileNotFoundError("No nba_api cache. Run scripts/pull_shots.py --source api.")
        frames = [shotchart_detail_to_common(pd.read_csv(path, dtype={"GAME_ID": str})) for path in files]
        return pd.concat(frames, ignore_index=True)
    if not mirror_files_ready(raw_dir):
        ensure_raw_files(source="mirror", raw_dir=raw_dir)
    return load_mirror_frame(raw_dir)

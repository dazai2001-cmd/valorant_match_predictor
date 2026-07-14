import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

BASE_URL = "https://www.vlr.gg"
VLRGGAPI_BASE_URL = os.getenv(
    "VLRGGAPI_BASE_URL",
    "http://127.0.0.1:3001",
).rstrip("/")

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL + "/",
}

TEAM_PAGES = {
    "G2 Esports": "https://www.vlr.gg/team/matches/11058/g2-esports/",
    "Xi Lai Gaming": "https://www.vlr.gg/team/matches/13581/xi-lai-gaming/",
    "Fnatic": "https://www.vlr.gg/team/matches/2593/fnatic/",
    "Rex Regum Qeon": "https://www.vlr.gg/team/matches/878/rex-regum-qeon/",
    "Gen.G": "https://www.vlr.gg/team/matches/17/gen-g/",
    "Sentinels": "https://www.vlr.gg/team/matches/2/sentinels/",
    "Wolves Esports": "https://www.vlr.gg/team/matches/13790/wolves-esports/",
    "Paper Rex": "https://www.vlr.gg/team/matches/624/paper-rex/",
}

MATCHES_CSV = DATA_DIR / "vlr_matches.csv"
MATCH_COVERAGE_CSV = DATA_DIR / "vlr_match_coverage.csv"
UPCOMING_MATCHES_CSV = DATA_DIR / "vlr_upcoming_matches.csv"
NEWS_CSV = DATA_DIR / "vlr_news.csv"
NEWS_EVENTS_CSV = DATA_DIR / "vlr_news_events.csv"
ROSTERS_CSV = DATA_DIR / "vlr_rosters.csv"
TEAMS_CSV = DATA_DIR / "vlr_teams.csv"
PREDICTIONS_CSV = DATA_DIR / "predicted_player_ratings.csv"
MATCH_PREDICTION_CSV = DATA_DIR / "match_prediction.csv"
DATABASE_PATH = DATA_DIR / "valorant_predictor.sqlite3"
VCT_EVENTS_CSV = DATA_DIR / "vct_events.csv"
JOB_STATUS_PATH = DATA_DIR / "job_status.json"

PLAYER_MODEL_PATH = MODELS_DIR / "player_rating_model.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_win_model.pkl"
MAP_MODEL_PATH = MODELS_DIR / "map_win_model.pkl"
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"
MODEL_SELECTION_PATH = MODELS_DIR / "model_selection.json"

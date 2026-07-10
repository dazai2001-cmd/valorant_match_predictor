from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

BASE_URL = "https://www.vlr.gg"

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
NEWS_CSV = DATA_DIR / "vlr_news.csv"
ROSTERS_CSV = DATA_DIR / "vlr_rosters.csv"
TEAMS_CSV = DATA_DIR / "vlr_teams.csv"
PREDICTIONS_CSV = DATA_DIR / "predicted_player_ratings.csv"
MATCH_PREDICTION_CSV = DATA_DIR / "match_prediction.csv"

PLAYER_MODEL_PATH = MODELS_DIR / "player_rating_model.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_win_model.pkl"
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"

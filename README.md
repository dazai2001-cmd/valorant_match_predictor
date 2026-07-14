# Valorant Match Predictor

This project predicts a rating range for each player, map-win probabilities, and a calibrated series-win probability using:

1. Recent and longer-term player performance from VLR match stat pages.
2. Opponent-adjusted team Elo, recent team form, head-to-head history, map-pool form, and map-veto context.
3. Current roster continuity and lineup availability.
4. Structured VLR news events such as injuries, benchings, returns, roster moves, and substitutes.

The result keeps evidence quality separate from win probability. A prediction can favor one team while still reporting weak data, uncertain lineups, or wide player-rating intervals.

## Setup

Install the packages used by the scraper:

```powershell
pip install -r requirements.txt
```

Start the pinned local `vlrggapi` helper used for official VCT event discovery:

```powershell
docker compose up -d
docker compose ps
```

The helper listens only on `http://127.0.0.1:3001`. Docker downloads the
immutable upstream commit listed in `THIRD_PARTY.md`; its source is not copied
into this repository. The direct VLR parser remains responsible for player-map
statistics because those tables are not consistently populated by the API.

The code is organized as an importable package under `src/valorant_predictor`. Use the scripts in `scripts/` from the project root for normal workflows.

## Project Layout

```text
src/valorant_predictor/   Core scraper, features, training, and prediction code
web/                      Flask app, templates, and static assets
scripts/                  Command entry points
data/                     CSV inputs and prediction outputs
artifacts/models/         Trained model files and training metrics
archive/old_scrapers/     Early scratch scrapers and experiments
infra/vlrggapi/            Reproducible wrapper for the pinned helper service
```

## Collect Data

Seed the VCT Tier 1 registry first. This stores the 48 Tier 1 teams and resolves their VLR team URLs when VLR search is reachable:

```powershell
python scripts\seed_teams.py
```

Scrape every match currently exposed on each active Tier 1 team's VLR match page:

```powershell
python scripts\scrape.py --matches --season-year 2026
```

Import the curated 2025 VCT season by official event ID:

```powershell
python scripts\import_vct.py --season-year 2025
```

This imports VCT Americas, EMEA, Pacific, China, Masters, and Champions. Only
Ascension matches involving a team promoted into the next Tier 1 season are
retained, with a lower competition weight. Event metadata comes from
`vlrggapi`; unseen player/map rows still come from the direct VLR match parser.

There is no default per-team cap. Match rows are saved to `data/vlr_matches.csv`. Each row represents one player's stats on one map; VLR's duplicate `?game=` links and the `All Maps` aggregate table are excluded. Coverage is saved to `data/vlr_match_coverage.csv` and counts unique match IDs for the selected season. The 20-match value is a data-quality benchmark only; it never excludes a team or prevents training.

CSV files remain the portable import/export layer. Scrapes and predictions are also synchronized to `data/valorant_predictor.sqlite3`, which keeps normalized match, map, player-map, roster, news-event, prediction, dataset-version, and model-run tables.

Updates merge into the existing CSV. Complete matches are reused, unseen matches are downloaded, and newly downloaded versions replace legacy rows for the same match ID. Use `--replace-history` only when you intentionally want a fresh file, or `--refresh-existing` to re-download complete matches.

The direct VLR parser records the veto note, map picker, decider, and veto order when VLR exposes them. The bundled self-hosted [vlrggapi](https://github.com/axsddlr/vlrggapi) service can enrich the same fields when the upstream endpoint exposes them:

```powershell
$env:VLRGGAPI_BASE_URL = "http://127.0.0.1:3001"
python scripts\scrape.py --matches --refresh-existing
```

The scraper checks the helper once per run and falls back to direct VLR parsing when it is unavailable. Existing historical rows are kept and marked `legacy`/`unknown` until refreshed; the app never invents old vetoes.

Files produced by the older scraper are marked as legacy because they do not contain map IDs. The first repaired update re-downloads those recent matches; only map-complete matches count toward the coverage target.

Scrape recent VLR news:

```powershell
python scripts\scrape.py --news --news-pages 3
```

Scrape current VLR rosters:

```powershell
python scripts\scrape.py --rosters
```

Scrape the current Tier 1 schedule from VLR:

```powershell
python scripts\scrape.py --upcoming
```

You can collect matches, news, rosters, and the schedule together:

```powershell
python scripts\scrape.py --matches --news --rosters --upcoming --news-pages 3
```

## Predict Player Ratings

Example:

```powershell
python scripts\predict_player_ratings.py --team1 "Gen.G" --team2 "Sentinels" --refresh-news --refresh-rosters
```

The command writes `data/predicted_player_ratings.csv`.

## Predict A Match Winner

Example:

```powershell
python scripts\predict_match.py --team1 "Gen.G" --team2 "Sentinels" --best-of 3 --refresh-news --refresh-rosters
```

Use trained models after running `scripts\train.py`:

```powershell
python scripts\predict_match.py --team1 "Gen.G" --team2 "Sentinels" --best-of 3 --use-trained-models
```

When the veto is known, pass the maps in play order. This removes map-selection uncertainty and is more accurate than asking the model to estimate the veto:

```powershell
python scripts\predict_match.py --team1 "BBL Esports" --team2 "NRG" --best-of 3 --maps Haven Breeze Lotus --use-trained-models
```

The command writes two files:

- `data/predicted_player_ratings.csv`
- `data/match_prediction.csv`

It prints:

- Predicted winner
- Win probability for each team
- Team strength table
- Key reasons for the pick
- Predicted rating for current active roster players
- `used_in_lineup` for the five players used in team strength
- Roster uncertainty for new or unknown roster pieces
- An 80% interval around every player rating
- A likely series score and simulated win-probability range
- A probability and likely winner for every selected or estimated map
- Evidence quality and selected-versus-estimated map context

## Web App

Run the Flask app:

```powershell
python scripts\run_web.py
```

Then open:

```text
http://127.0.0.1:5000
```

The web app can run predictions, choose an explicit map order for Bo1/Bo3/Bo5, update data, compare candidate models, and display series, map, team, and player forecasts. Leave every map on **Auto** before a veto; select every map after a veto is known. Partial or duplicate map selections are rejected.

Use **Update database & model** to resume missing 2025 event matches, merge new 2026 matches and player-map rows, preserve and refresh rosters, update news and the upcoming schedule, then retrain only when the usable dataset or model choice changed. The job runs in the background and reports progress in the page.

Use **Train & evaluate** to force a fresh chronological evaluation without scraping. Each task can stay on **Auto** or use a manually selected candidate. Auto chooses on the validation period, evaluates once on the later untouched test period, and falls back to the baseline when a match/map candidate misses the held-out log-loss and Brier safeguards. A manual override remains visibly marked even when it misses those safeguards.

The 2025 and 2026 team registries are season-specific. Unclassified matches are usable only when both opponents belong to that season's Tier 1 registry; this prevents Tier 2 opponents from leaking in through a Tier 1 team's general match page. Coverage below 20 games is an evidence-quality warning, not a minimum or maximum training limit.

You can run the same team setup from the terminal:

```powershell
python scripts\seed_teams.py
```

## Train Models

Train the player-rating, series-win, and map-win models on every stored match up to the selected season:

```powershell
python scripts\train.py --season-year 2026 --recent-days 60 --refresh-matches
```

This writes:

- `artifacts/models/player_rating_model.pkl`
- `artifacts/models/team_win_model.pkl`
- `artifacts/models/map_win_model.pkl`
- `artifacts/models/training_metrics.json`

The trainer fingerprints the usable dataset and skips retraining when neither the data, model version, nor requested model selection changed. Pass `--force` or use **Train & evaluate** to rebuild deliberately.

The player trainer compares squared-loss gradient boosting, absolute-loss gradient boosting, and Extra Trees on a chronological calibration fold. It predicts the residual change from a leakage-safe last-10 baseline and trains separate lower and upper quantile models for the rating range.

The series and map trainers compare an Elo-residual model, a direct gradient-boosted classifier, and logistic regression. Each chooses the lowest calibration log loss, then reports accuracy, log loss, and Brier score on a later untouched test fold. The map model is trained on one row per team per real map, with both sides of a match kept in the same split.

Probability candidates are blended with their baseline, temperature-scaled symmetrically, and capped so the raw learner cannot move more than 20 percentage points away from its baseline before blending. At prediction time, the learned map forecast is also weighted by its measured held-out reliability instead of replacing the stable forecast wholesale.

A trained series or map model is enabled only when it beats its baseline on both held-out log loss and Brier score. Otherwise the predictor falls back to the explainable Elo/form calculation.

The web app shows a 20-game coverage benchmark in the Tier 1 Teams panel. Low coverage means confidence should be lower; all valid available rows are still used.

Training uses:

- Curated Tier 1 history plus discounted promotion evidence, with no arbitrary minimum or maximum training-row limit
- A 365-day sample-weight half-life, so an otherwise equal one-year-old example has half the influence of a current example
- Modest event-importance weights, capped so playoffs and LAN matches matter more without dominating the data
- Last 60 days as a recent-form feature
- Only matches before the target match when building features
- Chronological train/calibration/test splits grouped by match, so mirrored team rows and players from one match cannot leak across folds
- Dedicated map examples with map identity, map form, opponent interaction, picker/decider context when known, and roster state
- Candidate-model selection on calibration data, followed by one final evaluation on later test data
- Rolling-origin backtests to measure stability across several points in time
- Last-10 player form, coin-flip probability, and Elo as explicit baselines
- Accuracy, MAE, log loss, Brier score, and skill against those baselines
- Low-variance baseline blending and symmetric temperature calibration fitted only on the calibration fold
- Quantile intervals adjusted on calibration data and measured on untouched test data

Old player and team history is retained because it can still contain useful matchup and identity information, but it is never treated as equally current. VLR's 60-day player/agent view remains a recent-form feature rather than the whole model.

## How The Rating Works

Base rating starts near `1.00`, similar to common esports stat ratings.

It uses:

- VLR Rating 2.0 when available
- A time-decayed 60-day rating blended with a slower 180-day stability estimate
- Last 3, last 5, and last 10 map form
- Overall player average
- ACS, kill/death ratio, and assists as backup signals
- Uncapped 60-day effective sample size, freshness, and reliability shrinkage
- Shrunk player volatility and trained-model residual error for the rating interval
- Learned 10th/90th residual quantiles, calibrated toward 80% held-out coverage

News is parsed into typed events with confidence and expiry. Performance and availability are kept separate:

- Injury, illness, wrist issues, missing an event: negative adjustment
- Benched, released, departs, retires: negative adjustment
- Returns can restore availability
- Joins, signs, substitutes, and roster completion change lineup uncertainty, not expected strength by themselves
- Team news affects every player lightly
- Direct player mentions affect that player more strongly

The output includes `news_reasons` and `news_urls` so you can see why a rating moved.

## How Match Prediction Works

The match predictor:

1. Builds predicted player ratings for both teams.
2. Uses current VLR rosters when `vlr_rosters.csv` is available.
3. Replays match history sequentially to build leakage-safe Elo, strength-of-schedule, roster-continuity, and map-pool features.
4. Decays Elo and map-pool evidence with age, with controlled carry-over across seasons, patches, and roster changes.
5. Uses the chosen Bo1/Bo3/Bo5 map order when supplied; otherwise estimates an order from recent map appearances, picks, and deciders and represents that uncertainty.
6. Produces an opponent-specific probability for each map with the dedicated map model when it passed held-out safeguards.
7. Combines the map series with the independently trained series model and the explainable Elo/form score according to measured held-out skill.
8. Shrinks weak evidence toward 50/50 and applies probability calibration when available.
9. Runs correlated Monte Carlo simulations with map-specific probabilities, shared team/series shocks, and player-performance volatility for Bo1, Bo3, or Bo5.

Small datasets are deliberately dampened toward 50/50. The map list in the UI is derived from stored competitive data rather than a hardcoded patch pool, because tournament pools can differ by event.

## How Roster Changes Work

Roster changes are treated as uncertainty, not as an automatic penalty.

If a team adds a strong player, team strength can go up because that player's own historical rating follows them to the new roster. If the player has little or no known data, they receive a conservative average rating and higher uncertainty.

In short:

- Current roster decides who is eligible for the lineup.
- Player history follows the player, even across teams.
- New-to-team players lower confidence, not expected strength.
- Inactive players are saved in `vlr_rosters.csv` but not used in the active lineup by default.

## Important Data Note

Scraped rows include `match_id`, `match_date`, `map_id`, `map_number`, `map_name`, `map_veto`, `map_pick_team`, `map_pick_type`, `map_veto_order`, and `map_data_source`. Player form is calculated from real map rows, while team form groups those rows back into unique matches. This prevents map-detail URLs and VLR's aggregate table from inflating samples.

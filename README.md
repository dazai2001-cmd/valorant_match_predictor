# Valorant Match Predictor

This project predicts a rating for each player in a Valorant matchup using two signals:

1. Recent player performance from VLR match stat pages.
2. Recent VLR news that may affect players or teams, such as injuries, benchings, returns, roster moves, substitutes, and momentum.

The rating is intentionally explainable. A player gets a base rating from match stats, then a news adjustment is applied when VLR news mentions their team or the player directly.

## Setup

Install the packages used by the scraper:

```powershell
pip install -r requirements.txt
```

The code is organized as an importable package under `src/valorant_predictor`. Use the scripts in `scripts/` from the project root for normal workflows.

## Project Layout

```text
src/valorant_predictor/   Core scraper, features, training, and prediction code
web/                      Flask app, templates, and static assets
scripts/                  Command entry points
data/                     CSV inputs and prediction outputs
artifacts/models/         Trained model files and training metrics
archive/old_scrapers/     Early scratch scrapers and experiments
```

## Collect Data

Scrape recent matches for the teams listed in `config.py`:

```powershell
python scripts\scrape.py --matches --limit-per-team 25
```

Match rows are saved to `data/vlr_matches.csv`.

Scrape recent VLR news:

```powershell
python scripts\scrape.py --news --news-pages 3
```

Scrape current VLR rosters:

```powershell
python scripts\scrape.py --rosters
```

You can collect matches, news, and rosters together:

```powershell
python scripts\scrape.py --matches --news --rosters --limit-per-team 25 --news-pages 3
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

## Web App

Run the Flask app:

```powershell
python scripts\run_web.py
```

Then open:

```text
http://127.0.0.1:5000
```

The web app can run predictions, refresh matches/news/rosters, train models, and display team/player rating tables.

Use **Seed 2026 Tier 1** in the **Tier 1 Teams** panel once at the start of the season. This writes the 48 VCT Tier 1 teams, 12 from each league, into `data/vlr_teams.csv` with `league`, `tier`, `season_year`, and `active` fields. It also tries to resolve each team through VLR search so match and roster scrapes have the right VLR team URLs.

The predictor dropdown and refresh jobs use active `tier1` teams only. Regional VLR rankings are not used as the team database because they can include Tier 2 teams. If VLR URL resolution fails, rerun the seed step when VLR is reachable.

You can run the same team setup from the terminal:

```powershell
python scripts\seed_teams.py
```

## Train Models

Train the player-rating and team-win models on a season:

```powershell
python scripts\train.py --season-year 2026 --recent-days 60 --refresh-matches --limit-per-team 75
```

This writes:

- `artifacts/models/player_rating_model.pkl`
- `artifacts/models/team_win_model.pkl`
- `artifacts/models/training_metrics.json`

The trained player model predicts a player's next VLR Rating 2.0. The trained team model predicts whether one team beats another from pre-match team features.

If the dataset is still small, the trainer may skip the team model instead of saving a weak classifier. Scrape more season matches with a higher `--limit-per-team` before trusting the team model.

Training uses:

- Current season as the main dataset
- Last 60 days as a recent-form feature
- Previous year only as fallback history when current-season samples are thin
- Only matches before the target match when building features

The project does not use all-time player history as the default because old Valorant data goes stale quickly. VLR's 60-day player/agent view is useful as a recent-form feature, but it is too narrow to be the whole model by itself.

## How The Rating Works

Base rating starts near `1.00`, similar to common esports stat ratings.

It uses:

- VLR Rating 2.0 when available
- Weighted recent rating over recent maps
- Last 3, last 5, and last 10 map form
- Overall player average
- ACS, kill/death ratio, and assists as backup signals
- Recent map sample size and reliability shrinkage

News impact is then applied as a percentage adjustment. Examples:

- Injury, illness, wrist issues, missing an event: negative adjustment
- Benched, released, departs, retires: negative adjustment
- Returns, joins, signs, roster completed: positive adjustment
- Team news affects every player lightly
- Direct player mentions affect that player more strongly

The output includes `news_reasons` and `news_urls` so you can see why a rating moved.

## How Match Prediction Works

The match predictor:

1. Builds predicted player ratings for both teams.
2. Uses current VLR rosters when `vlr_rosters.csv` is available.
3. Calculates team strength from average player rating, top-player impact, weak-link penalty, team form, and news.
4. Converts the strength difference into a map win probability.
5. Converts map probability into match probability for Bo1, Bo3, or Bo5.

Small datasets are deliberately dampened toward 50/50. Once you scrape more matches, the model becomes more willing to make stronger predictions.

## How Roster Changes Work

Roster changes are treated as uncertainty, not as an automatic penalty.

If a team adds a strong player, team strength can go up because that player's own historical rating follows them to the new roster. If the player has little or no known data, they receive a conservative average rating and higher uncertainty.

In short:

- Current roster decides who is eligible for the lineup.
- Player history follows the player, even across teams.
- New-to-team players lower confidence, not expected strength.
- Inactive players are saved in `vlr_rosters.csv` but not used in the active lineup by default.

## Important Data Note

Future scrapes include `match_id` and `match_date` when VLR exposes it. Those fields make rolling form more reliable because the code can sort maps chronologically instead of guessing from file order.

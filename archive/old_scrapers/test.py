import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.vlr.gg/',
})

team_pages = {
    "G2 Esports":      "https://www.vlr.gg/team/matches/11058/g2-esports/",
    "Xi Lai Gaming":   "https://www.vlr.gg/team/matches/13581/xi-lai-gaming/",
    "Fnatic":          "https://www.vlr.gg/team/matches/2593/fnatic/",
    "Rex Regum Qeon":  "https://www.vlr.gg/team/matches/878/rex-regum-qeon/",
    "Gen.G":           "https://www.vlr.gg/team/matches/17/gen-g/",
    "Sentinels":       "https://www.vlr.gg/team/matches/2/sentinels/",
    "Wolves Esports":  "https://www.vlr.gg/team/matches/13790/wolves-esports/",
    "Paper Rex":       "https://www.vlr.gg/team/matches/624/paper-rex/",
}

def get_recent_matches_for_team(team_url, limit=10):  # ← here!
    r = session.get(team_url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    matches = set()
    for a in soup.select("a.wf-card.fc-flex.m-item"):
        href = a.get("href", "")
        if href.startswith("/"):
            matches.add("https://www.vlr.gg" + href)
            if len(matches) >= limit:
                break
    return list(matches)

def parse_match_page(url):
    r = session.get(url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    # team names
    titles = [t.text.strip() for t in soup.select(".match-header .wf-title-med")]
    if len(titles) != 2:
        return []
    t1, t2 = titles

    # ——— extract scores & winner ———
    loser_span  = soup.select_one(".match-header-vs-score-loser")
    winner_span = soup.select_one(".match-header-vs-score-winner")
    if loser_span and winner_span:
        team1_score = int(loser_span.text.strip())
        team2_score = int(winner_span.text.strip())
        winner      = t2
    else:
        team1_score = team2_score = None
        winner      = None

    # per-player stats…
    data = []
    for row in soup.select(".vm-stats .wf-table-inset tbody tr"):
        cols = row.find_all("td")
        if len(cols) < 5:
            continue
        # …your existing parsing here…
        data.append({
            "match_url":  url,
            "team":       team,
            "opponent":   opponent,
            "winner":     winner,
            "score1":     team1_score,
            "score2":     team2_score,
            "player":     player,
            "agent":      agent,
            "acs":        acs,
            "kills":      kills,
            "deaths":     deaths,
        })
    return data

    except Exception as e:
        print(f"⚠️ Failed on {url}: {e}")
        return []

if __name__ == "__main__":
    all_urls = set()
    for name, page in team_pages.items():
        print(f"🔗 Collecting {name}")
        urls = get_recent_matches_for_team(page, limit=10)  # ← test with 10
        print(f"  → {len(urls)} found")
        all_urls.update(urls)
        time.sleep(random.uniform(1.0, 2.0))

    print(f"\n🌐 Scraping {len(all_urls)} matches…\n")
    records = []
    for i, murl in enumerate(all_urls, 1):
        print(f"[{i}/{len(all_urls)}] {murl}")
        records.extend(parse_match_page(murl))
        time.sleep(random.uniform(1.2, 2.5))

    df = pd.DataFrame(records)
    df.to_csv("vlr_matches_test.csv", index=False)
    print(f"\n✅ Saved {len(df)} rows to CSV")

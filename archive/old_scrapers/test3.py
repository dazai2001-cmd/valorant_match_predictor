import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
TEAM_MATCH_LIMIT = 5   # how many recent matches per team to grab (for quick tests)
TEAM_PAGES = {
    "G2 Esports":     "https://www.vlr.gg/team/matches/11058/g2-esports/",
    "Xi Lai Gaming":  "https://www.vlr.gg/team/matches/13581/xi-lai-gaming/",
    "Fnatic":         "https://www.vlr.gg/team/matches/2593/fnatic/",
    "Rex Regum Qeon": "https://www.vlr.gg/team/matches/878/rex-regum-qeon/",
    "Gen.G":          "https://www.vlr.gg/team/matches/17/gen-g/",
    "Sentinels":      "https://www.vlr.gg/team/matches/2/sentinels/",
    "Wolves Esports": "https://www.vlr.gg/team/matches/13790/wolves-esports/",
    "Paper Rex":      "https://www.vlr.gg/team/matches/624/paper-rex/",
}
OUT_CSV = "vlr_matches.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def get_team_match_urls(team_url):
    """Grab up to TEAM_MATCH_LIMIT match URLs from a team's 'matches' page."""
    r = requests.get(team_url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.select("a.wf-card.fc-flex.m-item")[:TEAM_MATCH_LIMIT]:
        href = a.get("href", "")
        if href.startswith("/"):
            urls.append("https://www.vlr.gg" + href)
    return urls

def parse_match(url):
    """Scrape winner + per-player stats from a match page."""
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    # --- parse teams + series score/winner ---
    team_elems  = soup.select(".match-header-link .wf-title-med")
    teams       = [t.text.strip() for t in team_elems]
    # get the two raw numbers
    score_spans = soup.select(".match-header-vs-score span")
    try:
        s1, s2 = int(score_spans[0].text), int(score_spans[1].text)
    except:
        return []  # no result yet
    winner = teams[0] if s1 > s2 else teams[1]

    # --- parse player stats rows ---
    out = []
    rows = soup.select(".vm-stats-game tbody tr")
    for r in rows:
        tds = r.find_all("td")
        if len(tds) < 5:
            continue
        player = tds[0].text.strip().split()[0]
        # figure out which team this row belongs to by matching the name
        row_team = teams[0] if teams[0][:3].upper() in tds[0].text.upper() else teams[1]
        opponent = teams[1] if row_team == teams[0] else teams[0]

        # agent
        img = tds[1].find("img")
        agent = img["title"].strip() if img and img.has_attr("title") else "Unknown"
        # ACS
        acs = int(tds[3].text.strip().split("\n")[0])
        # kills/deaths
        k, d = tds[4].text.strip().split("/")[0:2]
        kills, deaths = int(k), int(d)

        out.append({
            "match_url": url,
            "team": row_team,
            "opponent": opponent,
            "winner": winner,
            "team_score": s1 if row_team == teams[0] else s2,
            "opp_score":  s2 if row_team == teams[0] else s1,
            "player": player,
            "agent": agent,
            "acs": acs,
            "kills": kills,
            "deaths": deaths,
            "is_winner": row_team == winner
        })
    return out

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # 1) gather all the match URLs
    all_urls = []
    for name, url in TEAM_PAGES.items():
        print(f"→ grabbing matches for {name}")
        murls = get_team_match_urls(url)
        print(f"   found {len(murls)}")
        all_urls.extend(murls)
        time.sleep(random.uniform(1.0, 2.0))
    all_urls = list(dict.fromkeys(all_urls))  # dedupe
    print(f"⚡ total unique matches: {len(all_urls)}\n")

    # 2) scrape each match
    records = []
    for i, m in enumerate(all_urls, 1):
        print(f"[{i}/{len(all_urls)}] scraping {m}")
        records += parse_match(m)
        time.sleep(random.uniform(1.2, 2.5))

    # 3) save
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅  Wrote {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()

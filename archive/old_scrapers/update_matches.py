import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
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

csv_path = "vlr_matches.csv"
existing_urls = set()

# Step 0: Load previously saved matches
if os.path.exists(csv_path):
    try:
        df_existing = pd.read_csv(csv_path)
        existing_urls = set(df_existing["match_url"].unique())
        print(f"🗂️  Loaded {len(existing_urls)} existing match URLs.")
    except Exception as e:
        print("⚠️  Couldn't read existing CSV:", e)

def get_recent_matches_for_team(team_url, limit=75):
    r = session.get(team_url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    matches = set()
    for a in soup.select("a.wf-card.fc-flex.m-item"):
        href = a.get("href", "")
        if href.startswith("/"):
            full_url = "https://www.vlr.gg" + href
            matches.add(full_url)
            if len(matches) >= limit:
                break
    return list(matches)

def parse_match_page(url):
    try:
        r = session.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        teams = [t.text.strip() for t in soup.select(".match-header .wf-title-med")]
        if len(teams) != 2:
            return []
        t1, t2 = teams

        score_block = soup.select_one(".match-header-vs-score")
        winner = None
        if score_block:
            s = score_block.get_text(strip=True).replace("\n", "").replace("\t", "")
            if ":" in s:
                try:
                    s1, s2 = map(int, s.split(":")[:2])
                    winner = t1 if s1 > s2 else t2
                except ValueError:
                    pass

        rows = soup.select(".vm-stats-game .wf-table-inset tbody tr")
        data = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 5:
                continue
            try:
                player = tds[0].text.strip().split()[0]
                team = t1 if t1[:3].upper() in tds[0].text.upper() else t2
                img = tds[1].find("img")
                agent = img["title"].strip() if img and img.has_attr("title") else "Unknown"
                acs = int(tds[3].text.strip().split("\n")[0])
                k, d = tds[4].text.strip().replace("\n", "/").split("/")[:2]
                kills, deaths = int(k), int(d)
            except:
                continue
            data.append({
                "match_url": url,
                "team": team,
                "opponent": t2 if team == t1 else t1,
                "winner": winner,
                "player": player,
                "agent": agent,
                "acs": acs,
                "kills": kills,
                "deaths": deaths,
            })
        return data
    except Exception as e:
        print(f"⚠️  Failed on {url}: {e}")
        return []

# Step 1: Collect new match URLs only
all_match_urls = set()
for name, page in team_pages.items():
    print(f"🔍  Checking for new matches: {name}")
    urls = get_recent_matches_for_team(page, limit=75)
    new_urls = [u for u in urls if u not in existing_urls]
    print(f"   → {len(new_urls)} new URLs found")
    all_match_urls.update(new_urls)
    time.sleep(random.uniform(1.0, 2.0))

# Step 2: Parse only new matches
print(f"\n📦  Total new matches to scrape: {len(all_match_urls)}\n")
records = []
for i, match_url in enumerate(all_match_urls, 1):
    print(f"[{i:>3}/{len(all_match_urls)}] {match_url}")
    records.extend(parse_match_page(match_url))
    time.sleep(random.uniform(1.2, 2.5))

# Step 3: Append to existing CSV
df_new = pd.DataFrame(records)
if os.path.exists(csv_path):
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(csv_path, index=False)
print(f"\n✅  Appended {len(df_new)} new rows. Total now: {len(df_final)} rows saved.")

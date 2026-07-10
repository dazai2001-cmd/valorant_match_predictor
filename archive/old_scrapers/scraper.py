import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0',
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

def get_match_links(team_url, limit=20):
    r = session.get(team_url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    matches = set()
    for card in soup.select("a.wf-card.fc-flex.m-item"):
        href = card.get("href", "")
        if href.startswith("/") and "vs" in href:
            matches.add("https://www.vlr.gg" + href)
            if len(matches) >= limit:
                break
    return list(matches)

def parse_match(url):
    try:
        r = session.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        teams = [t.text.strip() for t in soup.select(".match-header .wf-title-med")]
        if len(teams) != 2:
            return []

        t1, t2 = teams
        winner = None

        score_block = soup.select_one(".match-header-vs-score .js-spoiler")
        if score_block:
            scores = score_block.find_all("span")
            if len(scores) >= 3:
                try:
                    s1 = int(scores[0].text)
                    s2 = int(scores[2].text)
                    winner = t1 if s1 > s2 else t2
                except:
                    pass

        rows = soup.select(".vm-stats-game .wf-table-inset tbody tr")
        data = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 5:
                continue
            try:
                player = tds[0].text.strip().split()[0]
                team_name = t1 if t1[:3].upper() in tds[0].text.upper() else t2
                img = tds[1].find("img")
                agent = img["title"].strip() if img and img.has_attr("title") else "Unknown"
                acs = int(tds[3].text.strip().split("\n")[0])
                k, d = tds[4].text.strip().replace("\n", "/").split("/")[:2]
                kills, deaths = int(k), int(d)
            except:
                continue
            data.append({
                "match_url": url,
                "team": team_name,
                "opponent": t2 if team_name == t1 else t1,
                "winner": winner,
                "player": player,
                "agent": agent,
                "acs": acs,
                "kills": kills,
                "deaths": deaths,
            })
        return data
    except Exception as e:
        print(f"⚠️  Error parsing {url}: {e}")
        return []

# Collect match URLs
all_urls = set()
for team, url in team_pages.items():
    print(f"🔗 Collecting matches for {team}...")
    team_urls = get_match_links(url, limit=75)
    print(f" → {len(team_urls)} URLs found")
    all_urls.update(team_urls)
    time.sleep(random.uniform(1.0, 2.0))

print(f"\n🌐 Total unique matches to scrape: {len(all_urls)}\n")

# Scrape each match
results = []
for i, match_url in enumerate(all_urls, 1):
    print(f"[{i}/{len(all_urls)}] {match_url}")
    results.extend(parse_match(match_url))
    time.sleep(random.uniform(1.2, 2.5))

df = pd.DataFrame(results)
df.to_csv("vlr_matches.csv", index=False)
print(f"\n✅ Saved {len(df)} rows to 'vlr_matches.csv'")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# — config —
LIMIT_PER_TEAM = 2   # adjust as needed
OUTPUT_CSV = "vlr_matches.csv"

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

# — session with headers —
session = requests.Session()
session.headers.update({
    'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://www.vlr.gg/',
})

def get_match_links(team_url, limit=LIMIT_PER_TEAM):
    """Fetch up to `limit` unique match URLs from a team's match list page."""
    r = session.get(team_url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.select("a.wf-card.fc-flex.m-item"):
        href = a.get("href", "")
        if href.startswith("/") and "vs" in href:
            full = "https://www.vlr.gg" + href
            if full not in urls:
                urls.append(full)
            if len(urls) >= limit:
                break
    return urls


def parse_match_page(url):
    """Parse one match page for teams, scores, winner, and performance stats."""
    recs = []
    try:
        r = session.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        # — teams & score/winner —
        titles = [t.text.strip() for t in soup.select(".match-header .wf-title-med")]
        if len(titles) != 2:
            return recs
        t1, t2 = titles

        loser_span  = soup.select_one(".match-header-vs-score-loser")
        winner_span = soup.select_one(".match-header-vs-score-winner")
        if loser_span and winner_span:
            s1 = int(loser_span.text.strip())
            s2 = int(winner_span.text.strip())
            winner = t1 if s1 > s2 else t2
        else:
            s1 = s2 = None
            winner = None

        # — per-player performance rows —
        for row in soup.select(".vm-stats .vm-stats-game tbody tr"):
            cols = row.find_all("td")
            if not cols:
                continue

            # first cell: player & team
            first = cols[0]
            parts = list(first.stripped_strings)
            if len(parts) < 2:
                continue
            player    = parts[0]
            team_name = parts[1]
            opponent  = t2 if team_name == t1 else t1

            # next cells up to ACS are agent picks
            i = 1
            agents = []
            while i < len(cols):
                txt = cols[i].get_text(strip=True)
                # ACS is the first purely-numeric-or-decimal cell
                if re.match(r"^\d+(\.\d+)?$", txt):
                    break
                img = cols[i].find("img")
                agents.append(img["title"].strip() if img and img.has_attr("title") else txt)
                i += 1

            # now i points at ACS
            try:
                acs = float(cols[i].text.strip())
            except:
                continue

            # K/D/A
            if i+1 >= len(cols):
                continue
            kda = cols[i+1].get_text(strip=True)
            kda_parts = kda.split("/")
            if len(kda_parts) < 3:
                continue
            kills, deaths, assists = map(int, kda_parts[:3])

            # further stats
            def clean_int(cell):   return int(cell.get_text(strip=True).replace("+","").replace("%",""))
            def clean_float(cell): return float(cell.get_text(strip=True).replace("%",""))

            try:
                plus1 = clean_int(cols[i+2])
                kast  = clean_float(cols[i+3])
                adr   = float(cols[i+4].text.strip())
                hs    = clean_float(cols[i+5])
                fk    = int(cols[i+6].text.strip())
                fd    = int(cols[i+7].text.strip())
                plus2 = clean_int(cols[i+8])
            except:
                continue

            recs.append({
                "match_url":    url,
                "team":         team_name,
                "opponent":     opponent,
                "score1":       s1,
                "score2":       s2,
                "winner":       winner,
                "player":       player,
                "agents":       ";".join(agents),
                "acs":          acs,
                "kills":        kills,
                "deaths":       deaths,
                "assists":      assists,
                "+/-_start":    plus1,
                "kast":         kast,
                "adr":          adr,
                "hs_pct":       hs,
                "first_kills":  fk,
                "first_deaths": fd,
                "+/-_end":      plus2,
            })
    except Exception as e:
        print(f"⚠️ Failed parsing {url}: {e}")
    return recs


def main():
    # collect all match URLs
    all_urls = set()
    for team, url in TEAM_PAGES.items():
        print(f"🔗 Collecting up to {LIMIT_PER_TEAM} matches for {team}")
        links = get_match_links(url, limit=LIMIT_PER_TEAM)
        for link in links:
            all_urls.add(link)
        time.sleep(random.uniform(1.0, 2.0))

    print(f"\n🌐 Will scrape {len(all_urls)} unique matches.\n")

    # scrape each match
    records = []
    for idx, murl in enumerate(sorted(all_urls), 1):
        print(f"[{idx}/{len(all_urls)}] {murl}")
        records.extend(parse_match_page(murl))
        time.sleep(random.uniform(1.2, 2.5))

    # save to CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df)} rows to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()

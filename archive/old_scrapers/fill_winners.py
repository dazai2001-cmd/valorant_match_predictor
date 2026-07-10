import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

# Masters Toronto 2025 teams and their match history URLs
team_urls = {
    "G2 Esports": "https://www.vlr.gg/team/matches/11058/g2-esports",
    "Xi Lai Gaming": "https://www.vlr.gg/team/matches/13581/xi-lai-gaming",
    "Fnatic": "https://www.vlr.gg/team/matches/2593/fnatic",
    "Rex Regum Qeon": "https://www.vlr.gg/team/matches/878/rex-regum-qeon",
    "Gen.G": "https://www.vlr.gg/team/matches/17/gen-g",
    "Sentinels": "https://www.vlr.gg/team/matches/2/sentinels",
    "Wolves Esports": "https://www.vlr.gg/team/matches/13790/wolves-esports",
    "Paper Rex": "https://www.vlr.gg/team/matches/624/paper-rex",
}

headers = {"User-Agent": "Mozilla/5.0"}
match_data = []

def get_match_results(team_name, url):
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    cards = soup.select("a.wf-card")
    
    for card in cards:
        href = card.get("href")
        match_url = "https://www.vlr.gg" + href if href else None
        
        teams = card.select("span.m-item-team-name")
        scores = card.select("div.m-item-result span")
        if not match_url or len(teams) != 2 or len(scores) != 2:
            continue

        team1 = teams[0].get_text(strip=True)
        team2 = teams[1].get_text(strip=True)
        score1 = int(scores[0].text.strip())
        score2 = int(scores[1].text.strip())

        winner = team1 if score1 > score2 else team2

        match_data.append({
            "match_url": match_url,
            "team": team1,
            "opponent": team2,
            "winner": winner
        })
        match_data.append({
            "match_url": match_url,
            "team": team2,
            "opponent": team1,
            "winner": winner
        })

for team, url in team_urls.items():
    print(f"Scraping: {team}")
    get_match_results(team, url)
    time.sleep(1.5)

# Save as CSV
match_winners_df = pd.DataFrame(match_data)
match_winners_df.drop_duplicates(subset=["match_url", "team"], inplace=True)
match_winners_df.to_csv("team_match_winners.csv", index=False)
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valorant_predictor.features.form_calculations import clean_match_data
from valorant_predictor.training.train_models import team_rows_from_cleaned
from valorant_predictor.vct_events import annotate_vct_matches
from valorant_predictor.vlrggapi_client import (
    extract_vlrggapi_match_metadata,
    parse_map_veto,
)
from valorant_predictor.vlr_client import (
    canonical_team_lookup,
    canonical_team_name,
    canonicalize_match_dataframe,
    complete_match_ids,
    filter_tier1_match_candidates,
    get_team_match_urls,
    match_coverage_report,
    merge_match_history,
    parse_match_page,
    parse_team_match_candidates_page,
    parse_upcoming_matches_page,
    partition_registry_tier1_history,
    scrape_matches,
    scrape_rosters,
)


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.encoding = "utf-8"

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, html):
        self.html = html

    def get(self, _url, timeout=20):
        return FakeResponse(self.html)


def player_row(player, rating="1.20", acs="240", kills="20", deaths="14", assists="6"):
    return f"""
      <tr>
        <td class="mod-player"><a href="/player/1/{player}"><span class="text-of">{player}</span></a></td>
        <td class="mod-agents"><img title="Viper"></td>
        <td class="mod-stat"><span class="side mod-both">{rating}</span></td>
        <td class="mod-stat"><span class="side mod-both">{acs}</span></td>
        <td class="mod-stat"><span class="side mod-both">{kills}</span></td>
        <td class="mod-stat"><span class="side mod-both">{deaths}</span></td>
        <td class="mod-stat"><span class="side mod-both">{assists}</span></td>
      </tr>
    """


def stats_table(prefix):
    rows = "".join(player_row(f"{prefix}{index}") for index in range(5))
    return f'<table class="wf-table-inset"><tbody>{rows}</tbody></table>'


def overview_player_row(player, player_id):
    return f"""
      <div class="ovw-row">
        <div class="ovw-cell mod-player">
          <div class="ovw-player">
            <a href="/player/{player_id}/{player}"><div class="ovw-player-name">{player}</div></a>
          </div>
          <div class="ovw-agents"><img title="Viper" alt="viper"></div>
        </div>
        <div class="ovw-cell" data-col="rating2"><span class="side mod-both">1.20</span></div>
        <div class="ovw-cell" data-col="acs"><span class="side mod-both">240</span></div>
        <div class="ovw-cell mod-kda">
          <span class="ovw-kda-stat" data-col="kills"><span class="side mod-both">20</span></span>
          <span class="ovw-kda-stat" data-col="deaths"><span class="side mod-both">14</span></span>
          <span class="ovw-kda-stat" data-col="assists"><span class="side mod-both">6</span></span>
        </div>
      </div>
    """


def overview_stats():
    first = "".join(overview_player_row(f"new-a{index}", 100 + index) for index in range(5))
    second = "".join(overview_player_row(f"new-b{index}", 200 + index) for index in range(5))
    return f'<div class="ovw-table"><div class="ovw-row mod-head"></div>{first}<div class="ovw-row mod-head"></div>{second}</div>'


def match_html():
    aggregate = stats_table("aggregate-a") + stats_table("aggregate-b")
    maps = stats_table("a") + stats_table("b")
    return f"""
      <div class="match-header">
        <a href="/team/3478/pcific-esports/"><div class="wf-title-med">PCIFIC Esports</div></a>
        <div class="match-header-vs-score">2 : 1</div>
        <a href="/team/14419/giantx/"><div class="wf-title-med">GIANTX</div></a>
      </div>
      <span data-utc-ts="2026-05-12 12:00:00"></span>
      <div class="match-header-event">
        <a href="/event/123/vct-2026-emea-stage-1">
          VCT 2026: EMEA Stage 1
          <div class="match-header-event-series">Playoffs: Upper Final</div>
        </a>
      </div>
      <div class="match-header-note">GIANTX ban Haven; PCIFIC pick Lotus; Ascent remains</div>
      <div class="vm-stats-game mod-active" data-game-id="all">{aggregate}</div>
      <div class="vm-stats-game" data-game-id="268077">
        <div class="vm-stats-game-header">
          <div class="team"><div class="score">13</div></div>
          <div class="map"><span>Lotus <span class="picked">PICK</span></span></div>
          <div class="team mod-right"><div class="score">9</div></div>
        </div>
        {maps}
      </div>
    """


def overview_match_html():
    return f"""
      <div class="match-header">
        <a href="/team/3478/pcific-esports/"><div class="wf-title-med">PCIFIC Esports</div></a>
        <div class="match-header-vs-score">2 : 1</div>
        <a href="/team/14419/giantx/"><div class="wf-title-med">GIANTX</div></a>
      </div>
      <span data-utc-ts="2026-05-12 12:00:00"></span>
      <div class="match-header-note">GIANTX ban Haven; PCIFIC pick Lotus; Ascent remains</div>
      <div class="vm-stats-game" data-game-id="268077">
        <div class="vm-stats-game-header">
          <div class="team"><div class="score">13</div></div>
          <div class="map"><span>Lotus <span class="picked">PICK</span></span></div>
          <div class="team mod-right"><div class="score">9</div></div>
        </div>
        {overview_stats()}
      </div>
    """


def match_row(match_id, match_url, team, player, match_date, map_id=1, scope="map"):
    return {
        "match_url": match_url,
        "match_id": match_id,
        "match_date": match_date,
        "map_id": map_id,
        "map_number": 1,
        "map_name": "Lotus",
        "stat_scope": scope,
        "team": team,
        "opponent": "Other",
        "winner": team,
        "player": player,
        "agents": "Viper",
        "vlr_rating": 1.2,
        "acs": 240,
        "kills": 20,
        "deaths": 14,
        "assists": 6,
        "is_winner": True,
        "team_score": 2,
        "opp_score": 1,
    }


class VlrPipelineTests(unittest.TestCase):
    def test_roster_refresh_preserves_a_team_when_its_page_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "rosters.csv"
            pd.DataFrame(
                [
                    {"team": "Team A", "player": "Old A", "player_id": 1},
                    {"team": "Team B", "player": "Keep B", "player_id": 2},
                ]
            ).to_csv(output, index=False)
            pages = {"Team A": "a", "Team B": "b"}

            def parsed(_session, team, _url):
                if team == "Team A":
                    return [{"team": "Team A", "player": "New A", "player_id": 3}]
                return []

            with (
                patch("valorant_predictor.vlr_client.make_session", return_value=object()),
                patch("valorant_predictor.vlr_client.parse_team_roster_page", side_effect=parsed),
                patch("valorant_predictor.vlr_client.time.sleep"),
                patch("valorant_predictor.storage.sync_dataframe"),
            ):
                roster = scrape_rosters(output_csv=output, team_pages=pages)

        self.assertEqual(set(roster["player"]), {"New A", "Keep B"})
        self.assertEqual(roster.attrs["scrape_summary"]["failed_teams"], ["Team B"])

    def test_event_annotation_adds_tier_and_promotion_context(self):
        frame = pd.DataFrame(
            [
                match_row(
                    100,
                    "https://www.vlr.gg/100/a-vs-b",
                    "Promoted Team",
                    "p1",
                    "2025-10-01",
                )
            ]
        )
        annotated = annotate_vct_matches(
            frame,
            {
                100: {
                    "event_id": 99,
                    "season_year": 2025,
                    "event_name": "VCT 2025: Ascension",
                    "event_series": "Grand Final",
                    "competition_tier": "promotion",
                    "competition_strength_weight": 0.55,
                }
            },
            {"promotedteam"},
        )

        self.assertEqual(annotated.iloc[0]["event_id"], 99)
        self.assertEqual(annotated.iloc[0]["competition_tier"], "promotion")
        self.assertEqual(annotated.iloc[0]["competition_strength_weight"], 0.55)
        self.assertTrue(annotated.iloc[0]["team_promoted_next_season"])

    def test_match_discovery_uses_main_cards_and_unique_match_ids(self):
        html = """
          <a class="wf-card fc-flex m-item" href="/100/team-a-vs-team-b-event"></a>
          <a class="wf-card" href="/100/team-a-vs-team-b-event/?game=11"></a>
          <a class="wf-card" href="/100/team-a-vs-team-b-event/?game=12"></a>
          <a class="wf-card fc-flex m-item" href="/101/team-a-vs-team-c-event"></a>
        """
        urls = get_team_match_urls(FakeSession(html), "https://www.vlr.gg/team/matches/1/team-a/", 50)

        self.assertEqual(
            urls,
            [
                "https://www.vlr.gg/100/team-a-vs-team-b-event",
                "https://www.vlr.gg/101/team-a-vs-team-c-event",
            ],
        )

    def test_match_discovery_has_no_default_training_cap(self):
        cards = "".join(
            f'<a class="wf-card fc-flex m-item" href="/{match_id}/team-a-vs-team-b-event"></a>'
            for match_id in range(100, 175)
        )
        urls = get_team_match_urls(
            FakeSession(cards),
            "https://www.vlr.gg/team/matches/1/team-a/",
            None,
        )

        self.assertEqual(len(urls), 75)

    def test_match_card_parser_extracts_prefetch_metadata(self):
        html = """
          <a class="wf-card fc-flex m-item" href="/701052/jdg-vs-trace-vct-2026">
            <div class="m-item-event"><div>VCT 2026: China Stage 2</div>Group Stage</div>
            <span class="m-item-team-name">JDG Esports</span>
            <span class="m-item-team-name">Trace Esports</span>
            <div class="m-item-date"><div>2026/07/21</div>2:00 am</div>
          </a>
        """

        candidates = parse_team_match_candidates_page(BeautifulSoup(html, "html.parser"))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["match_date"], "2026-07-21")
        self.assertEqual(candidates[0]["event_name"], "VCT 2026: China Stage 2")
        self.assertEqual(candidates[0]["teams"], ["JDG Esports", "Trace Esports"])

    def test_prefetch_filter_rejects_wrong_season_and_non_tier1_matches(self):
        candidates = [
            {"match_id": 1, "match_date": "2026-01-01", "event_name": "VCT 2026: EMEA", "teams": ["Team A", "Team B"]},
            {"match_id": 2, "match_date": "2025-01-01", "event_name": "VCT 2025: EMEA", "teams": ["Team A", "Team B"]},
            {"match_id": 3, "match_date": "2026-01-02", "event_name": "Challengers 2026: Europe", "teams": ["Team A", "Team B"]},
            {"match_id": 4, "match_date": "2026-01-03", "event_name": "EWC 2026: Europe Qualifier", "teams": ["Team A", "Academy Team"]},
            {"match_id": 5, "match_date": "2026-01-04", "event_name": "VCT 2026: Ascension", "teams": ["Promoted Team", "Academy Team"]},
        ]

        relevant, rejected = filter_tier1_match_candidates(
            candidates,
            2026,
            ["Team A", "Team B"],
        )

        self.assertEqual([candidate["match_id"] for candidate in relevant], [1, 5])
        self.assertEqual(rejected["wrong_season"], 1)
        self.assertEqual(rejected["explicit_non_tier1_event"], 1)
        self.assertEqual(rejected["non_tier1_matchup"], 1)

    def test_upcoming_parser_keeps_only_tier1_matchups(self):
        html = """
          <div class="wf-label mod-large">Sat, July 11, 2026</div>
          <a class="wf-module-item match-item" href="/708425/bbl-esports-vs-nrg-event">
            <div class="match-item-time" data-utc-ts="1783763100">9:45 AM</div>
            <div class="match-item-vs-team-name">BBL Esports</div>
            <div class="match-item-vs-team-name">NRG</div>
            <div class="match-item-event-name">Esports World Cup 2026</div>
            <div class="match-item-event-series">Playoffs: Semifinals</div>
          </a>
          <a class="wf-module-item match-item" href="/708426/tier2-vs-nrg-event">
            <div class="match-item-time">11:00 AM</div>
            <div class="match-item-vs-team-name">Unknown Team</div>
            <div class="match-item-vs-team-name">NRG</div>
          </a>
        """
        pages = {
            "BBL Esports": "https://www.vlr.gg/team/matches/397/bbl-esports/",
            "NRG": "https://www.vlr.gg/team/matches/10386/nrg/",
        }
        upcoming = parse_upcoming_matches_page(BeautifulSoup(html, "html.parser"), pages)

        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming.iloc[0]["team1"], "BBL Esports")
        self.assertEqual(upcoming.iloc[0]["event_stage"], "Semifinals")

    def test_match_parser_emits_only_real_map_rows(self):
        url = "https://www.vlr.gg/673541/pcific-esports-vs-giantx-event/?game=268077"
        rows = parse_match_page(FakeSession(match_html()), url)

        self.assertEqual(len(rows), 10)
        self.assertEqual({row["map_id"] for row in rows}, {268077})
        self.assertEqual({row["map_name"] for row in rows}, {"Lotus"})
        self.assertEqual({row["stat_scope"] for row in rows}, {"map"})
        self.assertEqual({row["match_url"] for row in rows}, {url.split("?")[0].rstrip("/")})
        self.assertEqual(rows[0]["team_id"], 3478)
        self.assertEqual(rows[0]["map_team_score"], 13)
        self.assertEqual(rows[0]["map_pick_team"], "PCIFIC Esports")
        self.assertEqual(rows[0]["map_pick_type"], "pick")
        self.assertEqual(rows[0]["map_veto_order"], 2)
        self.assertEqual(rows[0]["event_name"], "VCT 2026: EMEA Stage 1")
        self.assertEqual(rows[0]["event_series"], "Playoffs: Upper Final")
        self.assertFalse(any(row["player"].startswith("aggregate") for row in rows))

    def test_match_parser_supports_current_overview_rows(self):
        rows = parse_match_page(
            FakeSession(overview_match_html()),
            "https://www.vlr.gg/673541/pcific-esports-vs-giantx-event",
        )

        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["player"], "new-a0")
        self.assertEqual(rows[0]["player_id"], 100)
        self.assertEqual(rows[0]["agents"], "Viper")
        self.assertEqual(rows[0]["vlr_rating"], 1.2)
        self.assertEqual(rows[0]["map_team_score"], 13)
        self.assertEqual(rows[-1]["map_opp_score"], 13)

    def test_vlrggapi_payload_normalizes_veto_and_map_metadata(self):
        payload = {
            "data": {
                "segments": [
                    {
                        "map_vetos": "NRG ban Split; BBL ban Fracture; NRG pick Haven; BBL pick Breeze; Lotus remains",
                        "teams": [{"name": "BBL Esports"}, {"name": "NRG"}],
                        "maps": [
                            {"map_name": "Haven", "picked_by": "PICK"},
                            {"map_name": "Breeze", "picked_by": "PICK"},
                            {"map_name": "Lotus", "picked_by": ""},
                        ],
                    }
                ]
            }
        }

        metadata = extract_vlrggapi_match_metadata(payload)
        actions = parse_map_veto(payload["data"]["segments"][0]["map_vetos"], ["BBL Esports", "NRG"])

        self.assertEqual(actions[2]["team"], "NRG")
        self.assertEqual(metadata["maps"][0]["map_pick_team"], "NRG")
        self.assertEqual(metadata["maps"][1]["map_pick_team"], "BBL Esports")
        self.assertEqual(metadata["maps"][2]["map_pick_type"], "decider")

    def test_team_ids_and_aliases_resolve_to_registry_names(self):
        pages = {
            "PCFIC Esports": "https://www.vlr.gg/team/matches/3478/pcific-esports/",
            "JD Gaming": "https://www.vlr.gg/team/matches/13576/jdg-esports/",
            "DRX": "https://www.vlr.gg/team/matches/8185/kiwoom-drx/",
            "KRU Esports": "https://www.vlr.gg/team/matches/2355/kru-esports/",
        }
        frame = pd.DataFrame(
            [
                {
                    "team": "PCIFIC Esports",
                    "team_id": 3478.0,
                    "opponent": "JDG Esports",
                    "opponent_id": 13576,
                }
            ]
        )
        canonical = canonicalize_match_dataframe(frame, pages)
        lookup = canonical_team_lookup(pages)

        self.assertEqual(canonical.iloc[0]["team"], "PCFIC Esports")
        self.assertEqual(canonical.iloc[0]["opponent"], "JD Gaming")
        self.assertEqual(canonical_team_name("KIWOOM DRX", lookup), "DRX")
        self.assertEqual(canonical_team_name("KRÜ Esports", lookup), "KRU Esports")

    def test_history_merge_replaces_legacy_match_and_keeps_other_matches(self):
        pages = {"Team A": "https://www.vlr.gg/team/matches/1/team-a/"}
        legacy = match_row(100, "https://www.vlr.gg/100/a-vs-b", "Team A", "old", "2026-01-01", map_id=None, scope="legacy")
        retained = match_row(99, "https://www.vlr.gg/99/a-vs-c", "Team A", "kept", "2025-12-01")
        incoming = match_row(100, "https://www.vlr.gg/100/a-vs-b?game=1", "Team A", "new", "2026-01-01")

        merged = merge_match_history(pd.DataFrame([legacy, retained]), pd.DataFrame([incoming]), pages)

        self.assertEqual(set(merged["match_id"].astype(int)), {99, 100})
        self.assertNotIn("old", set(merged["player"]))
        self.assertIn("new", set(merged["player"]))
        self.assertIn("kept", set(merged["player"]))

    def test_complete_match_does_not_require_veto_metadata(self):
        rows = []
        for index in range(10):
            row = match_row(
                100,
                "https://www.vlr.gg/100/a-vs-b",
                "Team A" if index < 5 else "Team B",
                f"p{index}",
                "2026-01-01",
            )
            row.update(
                {
                    "player_id": index + 1,
                    "event_name": "VCT 2026: EMEA",
                    "map_team_score": 13 if index < 5 else 9,
                    "map_opp_score": 9 if index < 5 else 13,
                    "map_veto": "",
                }
            )
            rows.append(row)

        self.assertEqual(complete_match_ids(pd.DataFrame(rows)), {100})

    def test_registry_partition_quarantines_non_tier1_opponents(self):
        registry = pd.DataFrame(
            [
                {"team": "Team A", "tier": "tier1", "active": True, "season_year": 2026},
                {"team": "Team B", "tier": "tier1", "active": True, "season_year": 2026},
            ]
        )
        tier1 = match_row(1, "https://www.vlr.gg/1/a-vs-b", "Team A", "p1", "2026-01-01")
        tier1.update({"opponent": "Team B", "competition_tier": "unknown", "event_tier": "unknown"})
        qualifier = match_row(2, "https://www.vlr.gg/2/a-vs-c", "Team A", "p2", "2026-01-02")
        qualifier.update({"opponent": "Academy Team", "competition_tier": "tier1", "event_tier": "tier1"})
        promotion = match_row(3, "https://www.vlr.gg/3/c-vs-d", "Promoted Team", "p3", "2026-01-03")
        promotion.update({"opponent": "Academy Team", "competition_tier": "promotion", "event_tier": "promotion"})

        relevant, excluded = partition_registry_tier1_history(
            pd.DataFrame([tier1, qualifier, promotion]),
            registry,
        )

        self.assertEqual(set(relevant["match_id"]), {1, 3})
        self.assertEqual(set(excluded["match_id"]), {2})
        self.assertEqual(relevant.loc[relevant["match_id"].eq(1), "competition_tier"].iloc[0], "tier1")

    def test_normal_match_update_does_not_call_expensive_helper(self):
        pages = {
            "Team A": "https://www.vlr.gg/team/matches/1/team-a/",
            "Team B": "https://www.vlr.gg/team/matches/2/team-b/",
        }
        candidate = {
            "match_id": 100,
            "match_url": "https://www.vlr.gg/100/a-vs-b",
            "match_date": "2026-01-01",
            "event_name": "VCT 2026: Test",
            "teams": ["Team A", "Team B"],
        }
        rows = []
        for index in range(10):
            team = "Team A" if index < 5 else "Team B"
            opponent = "Team B" if index < 5 else "Team A"
            row = match_row(100, candidate["match_url"], team, f"p{index}", "2026-01-01")
            row.update(
                {
                    "opponent": opponent,
                    "player_id": index + 1,
                    "event_name": candidate["event_name"],
                    "event_tier": "tier1",
                    "competition_tier": "tier1",
                    "map_team_score": 13 if index < 5 else 9,
                    "map_opp_score": 9 if index < 5 else 13,
                }
            )
            rows.append(row)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch("valorant_predictor.vlr_client.make_session", return_value=object()),
                patch("valorant_predictor.vlr_client.get_team_match_candidates", return_value=[candidate]),
                patch("valorant_predictor.vlr_client.parse_match_page", return_value=rows) as parser,
                patch("valorant_predictor.vlr_client.vlrggapi_is_healthy") as health,
                patch("valorant_predictor.vlr_client.fetch_vlrggapi_match_metadata") as details,
                patch("valorant_predictor.vlr_client.time.sleep"),
                patch("valorant_predictor.storage.sync_match_data"),
            ):
                output = scrape_matches(
                    output_csv=root / "matches.csv",
                    coverage_csv=root / "coverage.csv",
                    excluded_csv=root / "excluded.csv",
                    team_pages=pages,
                    season_year=2026,
                    pause_seconds=0,
                )

        self.assertEqual(len(output), 10)
        parser.assert_called_once()
        health.assert_not_called()
        details.assert_not_called()

    def test_coverage_counts_match_ids_for_the_requested_season(self):
        pages = {"Team A": "https://www.vlr.gg/team/matches/1/team-a/"}
        rows = [
            match_row(100, "https://www.vlr.gg/100/a-vs-b", "Team A", "p1", "2026-01-01"),
            match_row(100, "https://www.vlr.gg/100/a-vs-b?game=1", "Team A", "p2", "2026-01-01"),
            match_row(90, "https://www.vlr.gg/90/a-vs-c", "Team A", "p3", "2025-12-01"),
            match_row(101, "https://www.vlr.gg/101/a-vs-d", "Team A", "p4", "2026-02-01", map_id=None, scope="legacy"),
        ]

        coverage = match_coverage_report(pd.DataFrame(rows), pages, min_matches_per_team=2, season_year=2026)

        self.assertEqual(int(coverage.iloc[0]["matches"]), 1)
        self.assertEqual(int(coverage.iloc[0]["matches_total"]), 3)
        self.assertEqual(int(coverage.iloc[0]["legacy_matches"]), 1)
        self.assertEqual(coverage.iloc[0]["status"], "under_target")

    def test_cleaning_removes_aggregates_and_duplicate_map_rows(self):
        map_entry = match_row(100, "https://www.vlr.gg/100/a-vs-b", "Team A", "p1", "2026-01-01")
        aggregate = match_row(100, "https://www.vlr.gg/100/a-vs-b", "Team A", "p1", "2026-01-01", map_id=None, scope="aggregate")
        opponent = match_row(100, "https://www.vlr.gg/100/a-vs-b", "Other", "p2", "2026-01-01")
        opponent["opponent"] = "Team A"
        opponent["winner"] = "Team A"
        opponent["is_winner"] = False
        cleaned = clean_match_data(pd.DataFrame([map_entry, map_entry, aggregate, opponent]))
        team_rows = team_rows_from_cleaned(cleaned)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(len(team_rows), 2)
        self.assertEqual(team_rows["match_key"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()

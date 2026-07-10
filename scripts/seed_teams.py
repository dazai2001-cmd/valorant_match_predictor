from _path import add_src_to_path


add_src_to_path()

from valorant_predictor.team_registry import VCT_TIER1_SEASON, seed_vct_tier1_teams


if __name__ == "__main__":
    _, meta = seed_vct_tier1_teams(season_year=VCT_TIER1_SEASON, resolve_missing=True)
    print(
        f"Seeded {meta['seeded_count']} VCT Tier 1 teams for {meta['season_year']}. "
        f"{meta['resolved_count']} have VLR match URLs."
    )
    if meta["unresolved"]:
        print(f"Unresolved VLR URLs: {', '.join(meta['unresolved'])}")
    if meta.get("resolution_error"):
        print(f"VLR resolution stopped early: {meta['resolution_error']}")

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import requests

from .features.form_calculations import normalize_map_name


def _segments(payload: dict):
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return []
    segments = data.get("segments", [])
    return segments if segments is not None else []


def vlrggapi_get(
    session: requests.Session,
    base_url: str,
    path: str,
    params: dict | None = None,
    timeout: int = 60,
) -> dict:
    if not base_url:
        raise ValueError("VLRGGAPI_BASE_URL is not configured.")
    response = session.get(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise ValueError(payload.get("message") or f"vlrggapi request failed: {path}")
    return payload


def vlrggapi_is_healthy(
    session: requests.Session,
    base_url: str,
    timeout: int = 5,
) -> bool:
    try:
        vlrggapi_get(session, base_url, "/v2/health", timeout=timeout)
        return True
    except (requests.RequestException, TypeError, ValueError):
        return False


def fetch_vlrggapi_events(
    session: requests.Session,
    base_url: str,
    status: str = "completed",
    page: int = 1,
) -> list[dict]:
    payload = vlrggapi_get(
        session,
        base_url,
        "/v2/events",
        params={"q": status, "page": int(page)},
    )
    segments = _segments(payload)
    return [item for item in segments if isinstance(item, dict)]


def fetch_vlrggapi_event_detail(
    session: requests.Session,
    base_url: str,
    event_id: int | str,
) -> dict:
    payload = vlrggapi_get(
        session,
        base_url,
        f"/v2/event/{int(event_id)}",
        timeout=90,
    )
    segments = _segments(payload)
    return segments if isinstance(segments, dict) else {}


def fetch_vlrggapi_event_matches(
    session: requests.Session,
    base_url: str,
    event_id: int | str,
) -> list[dict]:
    payload = vlrggapi_get(
        session,
        base_url,
        "/v2/events/matches",
        params={"event_id": int(event_id)},
        timeout=90,
    )
    segments = _segments(payload)
    return [item for item in segments if isinstance(item, dict)]


def _entity_key(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_value.lower())


def resolve_veto_team(label: str, teams: Iterable[str]) -> str:
    label_key = _entity_key(label)
    if not label_key:
        return ""
    candidates = []
    for team in teams:
        team_key = _entity_key(team)
        first_word_key = _entity_key(str(team).split()[0])
        if label_key == team_key or label_key == first_word_key:
            return str(team)
        if len(label_key) >= 3 and label_key in team_key:
            candidates.append(str(team))
    return candidates[0] if len(candidates) == 1 else str(label).strip()


def parse_map_veto(veto_text: str, teams: Iterable[str] = ()) -> list[dict]:
    actions = []
    for order, raw_part in enumerate(re.split(r"\s*;\s*", str(veto_text or "")), start=1):
        part = re.sub(r"\s+", " ", raw_part).strip(" .")
        if not part:
            continue
        action_match = re.match(r"^(.+?)\s+(ban|pick)\s+(.+)$", part, flags=re.IGNORECASE)
        remains_match = re.match(r"^(.+?)\s+remains?$", part, flags=re.IGNORECASE)
        if action_match:
            team_label, action, map_name = action_match.groups()
            actions.append(
                {
                    "order": order,
                    "action": action.lower(),
                    "map_name": normalize_map_name(map_name),
                    "team": resolve_veto_team(team_label, teams),
                }
            )
        elif remains_match:
            actions.append(
                {
                    "order": order,
                    "action": "decider",
                    "map_name": normalize_map_name(remains_match.group(1)),
                    "team": "",
                }
            )
    return actions


def map_metadata_from_veto(veto_text: str, teams: Iterable[str] = ()) -> dict[str, dict]:
    metadata = {}
    for action in parse_map_veto(veto_text, teams):
        if action["action"] not in {"pick", "decider"}:
            continue
        metadata[_entity_key(action["map_name"])] = {
            "map_pick_team": action["team"] if action["action"] == "pick" else "",
            "map_pick_type": action["action"],
            "map_veto_order": action["order"],
        }
    return metadata


def extract_vlrggapi_match_metadata(payload: dict) -> dict:
    segments = _segments(payload)
    if isinstance(segments, dict):
        segments = [segments]
    if not segments:
        return {"map_veto": "", "maps": []}
    segment = segments[0] or {}
    teams = [team.get("name", "") for team in segment.get("teams", []) if isinstance(team, dict)]
    veto_text = str(segment.get("map_vetos", "") or "").strip()
    veto_maps = map_metadata_from_veto(veto_text, teams)
    maps = []
    for index, map_data in enumerate(segment.get("maps", []), start=1):
        if not isinstance(map_data, dict):
            continue
        map_name = normalize_map_name(map_data.get("map_name", ""))
        metadata = veto_maps.get(_entity_key(map_name), {})
        picked_by = resolve_veto_team(map_data.get("picked_by", ""), teams)
        pick_type = metadata.get("map_pick_type", "")
        if not pick_type and picked_by:
            pick_type = "pick"
        maps.append(
            {
                "map_number": index,
                "map_name": map_name,
                "map_pick_team": metadata.get("map_pick_team") or picked_by,
                "map_pick_type": pick_type or "unknown",
                "map_veto_order": metadata.get("map_veto_order"),
            }
        )
    return {"map_veto": veto_text, "maps": maps}


def fetch_vlrggapi_match_metadata(
    session: requests.Session,
    base_url: str,
    match_id: int,
) -> dict:
    if not base_url:
        return {"map_veto": "", "maps": []}
    payload = vlrggapi_get(
        session,
        base_url,
        "/v2/match/details",
        params={"match_id": int(match_id)},
        timeout=90,
    )
    return extract_vlrggapi_match_metadata(payload)


def enrich_match_records_with_map_metadata(
    records: list[dict],
    metadata: dict,
    source: str = "vlrggapi",
) -> list[dict]:
    if not records:
        return records
    by_number = {
        int(item["map_number"]): item
        for item in metadata.get("maps", [])
        if item.get("map_number") is not None
    }
    by_name = {
        _entity_key(item.get("map_name", "")): item
        for item in metadata.get("maps", [])
        if item.get("map_name")
    }
    for record in records:
        item = by_number.get(int(record.get("map_number") or 0)) or by_name.get(
            _entity_key(record.get("map_name", "")),
            {},
        )
        record["map_veto"] = metadata.get("map_veto", record.get("map_veto", ""))
        for field in ["map_pick_team", "map_pick_type", "map_veto_order"]:
            if item.get(field) not in (None, ""):
                record[field] = item[field]
        record["map_data_source"] = source
    return records

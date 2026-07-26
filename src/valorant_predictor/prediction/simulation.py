import hashlib
import math

import numpy as np


def _logit(probability: float) -> float:
    probability = min(0.999, max(0.001, probability))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _series_probability(map_probabilities: list[float], best_of: int) -> float:
    needed = (best_of // 2) + 1
    states = {(0, 0): 1.0}
    for map_index in range(best_of):
        probability = map_probabilities[map_index % len(map_probabilities)]
        next_states = {}
        for (wins_a, wins_b), state_probability in states.items():
            if wins_a >= needed or wins_b >= needed:
                next_states[(wins_a, wins_b)] = next_states.get((wins_a, wins_b), 0.0) + state_probability
                continue
            next_states[(wins_a + 1, wins_b)] = next_states.get((wins_a + 1, wins_b), 0.0) + state_probability * probability
            next_states[(wins_a, wins_b + 1)] = next_states.get((wins_a, wins_b + 1), 0.0) + state_probability * (1.0 - probability)
        states = next_states
    return sum(probability for (wins_a, _), probability in states.items() if wins_a >= needed)


def series_probability_from_maps(
    map_probabilities: dict[str, float] | list[float],
    best_of: int,
    map_order: list[str] | None = None,
) -> float:
    """Return the exact series probability for a known or estimated map order."""
    if best_of not in {1, 3, 5}:
        raise ValueError("best_of must be 1, 3, or 5.")
    if isinstance(map_probabilities, dict):
        if not map_probabilities:
            return 0.5
        names = list(map_probabilities)
        preferred = [name for name in (map_order or []) if name in map_probabilities]
        preferred.extend(name for name in names if name not in preferred)
        probabilities = [float(map_probabilities[name]) for name in preferred]
    else:
        probabilities = [float(value) for value in map_probabilities]
    if not probabilities:
        return 0.5
    probabilities = [min(0.98, max(0.02, value)) for value in probabilities]
    base_probabilities = list(probabilities)
    while len(probabilities) < best_of:
        probabilities.append(
            base_probabilities[len(probabilities) % len(base_probabilities)]
        )
    return _series_probability(probabilities[:best_of], best_of)


def equivalent_map_probability(series_probability: float, best_of: int) -> float:
    if best_of not in {1, 3, 5}:
        raise ValueError("best_of must be 1, 3, or 5.")
    target = min(0.98, max(0.02, float(series_probability)))
    if best_of == 1:
        return target
    low, high = 0.001, 0.999
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        estimated = _series_probability([midpoint] * best_of, best_of)
        if estimated < target:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def reverse_simulation_result(result: dict) -> dict:
    output = dict(result)
    output["team1_win_probability"] = 1.0 - float(result["team1_win_probability"])
    output["probability_low"] = 1.0 - float(result["probability_high"])
    output["probability_high"] = 1.0 - float(result["probability_low"])
    output["scenario_mean_probability"] = 1.0 - float(
        result["scenario_mean_probability"]
    )
    output["map_probabilities"] = {
        name: 1.0 - float(probability)
        for name, probability in result.get("map_probabilities", {}).items()
    }
    reversed_scores = {}
    for score, probability in result.get("score_probabilities", {}).items():
        first, second = score.split("-", maxsplit=1)
        reversed_scores[f"{second}-{first}"] = probability
    output["score_probabilities"] = dict(sorted(reversed_scores.items()))
    first, second = str(result["likely_score"]).split("-", maxsplit=1)
    output["likely_score"] = f"{second}-{first}"
    return output


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _map_indices(
    map_names: list[str],
    preferred_order: list[str],
    rng: np.random.Generator,
    uncertainty: float,
) -> list[int]:
    preferred = [name for name in preferred_order if name in map_names]
    preferred.extend(name for name in map_names if name not in preferred)
    if len(preferred) > 1 and rng.random() < uncertainty:
        swap_index = int(rng.integers(0, len(preferred) - 1))
        preferred[swap_index], preferred[swap_index + 1] = (
            preferred[swap_index + 1],
            preferred[swap_index],
        )
    return [map_names.index(name) for name in preferred]


def simulate_series(
    map_probabilities: dict[str, float] | list[float],
    best_of: int,
    performance_volatility: float,
    simulations: int = 10000,
    scenario_count: int = 1200,
    seed: int = 7,
    map_order: list[str] | None = None,
    veto_uncertainty: float = 0.25,
) -> dict:
    if best_of not in {1, 3, 5}:
        raise ValueError("best_of must be 1, 3, or 5.")

    if isinstance(map_probabilities, dict):
        map_names = sorted(map_probabilities)
        probabilities = [float(map_probabilities[name]) for name in map_names]
    else:
        probabilities = [float(value) for value in map_probabilities]
        map_names = [f"Map {index + 1}" for index in range(len(probabilities))]
    if not probabilities:
        probabilities = [0.5]
        map_names = ["Unknown"]

    probabilities = [min(0.95, max(0.05, probability)) for probability in probabilities]
    rng = np.random.default_rng(seed)
    volatility = min(0.85, max(0.05, float(performance_volatility)))
    veto_uncertainty = min(1.0, max(0.0, float(veto_uncertainty)))
    preferred_order = map_order or map_names

    scenario_probabilities = []
    for _ in range(scenario_count):
        team_shock = float(rng.normal(0.0, volatility))
        ordered = _map_indices(map_names, preferred_order, rng, veto_uncertainty)
        scenario_maps = [
            _sigmoid(
                _logit(probabilities[index])
                + team_shock
                + float(rng.normal(0.0, volatility * 0.30))
            )
            for index in ordered[: min(best_of, len(ordered))]
        ]
        while len(scenario_maps) < best_of:
            scenario_maps.append(scenario_maps[len(scenario_maps) % len(probabilities)])
        scenario_probabilities.append(_series_probability(scenario_maps, best_of))

    needed = (best_of // 2) + 1
    scores = {}
    wins = 0
    for _ in range(simulations):
        ordered = _map_indices(map_names, preferred_order, rng, veto_uncertainty)
        series_shock = float(rng.normal(0.0, volatility * 0.80))
        wins_a = 0
        wins_b = 0
        map_index = 0
        while wins_a < needed and wins_b < needed:
            probability_index = ordered[map_index % len(ordered)]
            probability = _sigmoid(
                _logit(probabilities[probability_index])
                + series_shock
                + float(rng.normal(0.0, volatility * 0.45))
            )
            if rng.random() < probability:
                wins_a += 1
            else:
                wins_b += 1
            map_index += 1
        wins += int(wins_a >= needed)
        score = f"{wins_a}-{wins_b}"
        scores[score] = scores.get(score, 0) + 1

    scenario_array = np.asarray(scenario_probabilities, dtype=float)
    likely_score = max(scores, key=scores.get)
    return {
        "team1_win_probability": wins / simulations,
        "probability_low": float(np.quantile(scenario_array, 0.10)),
        "probability_high": float(np.quantile(scenario_array, 0.90)),
        "scenario_mean_probability": float(scenario_array.mean()),
        "likely_score": likely_score,
        "score_probabilities": {
            score: count / simulations
            for score, count in sorted(scores.items())
        },
        "map_probabilities": dict(zip(map_names, probabilities)),
        "likely_map_order": [
            name for name in preferred_order if name in map_names
        ][:best_of],
        "simulations": simulations,
        "scenarios": scenario_count,
    }

"""Frozen, paired traffic demand for the four generalisation scenarios."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from src.simulation.metrics import APPROACHES


SOURCE_ROUTES = Path(__file__).resolve().parents[2] / "simulation/routes/demand.rou.xml"
TOTAL_ARRIVAL_RATE = 0.48  # Four original independent 0.12-per-second flows.
BALANCED = (0.25, 0.25, 0.25, 0.25)
NS_HEAVY = (0.35, 0.35, 0.15, 0.15)
EW_HEAVY = (0.15, 0.15, 0.35, 0.35)
DESTINATIONS = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass(frozen=True)
class Scenario:
    """Directional shares for consecutive generation windows."""

    name: str
    windows: tuple[tuple[float, float, float, float], ...]

    def shares_at(self, second: int, generation_seconds: int) -> tuple[float, ...]:
        """Select a window using integer time boundaries, starting at zero."""
        if not 0 <= second < generation_seconds:
            raise ValueError("second is outside the generation period")
        return self.windows[min(len(self.windows) - 1, second * len(self.windows) // generation_seconds)]

    def probabilities_at(self, second: int, generation_seconds: int) -> dict[str, float]:
        """Return independent Bernoulli arrival probabilities per approach."""
        return dict(zip(APPROACHES, (TOTAL_ARRIVAL_RATE * share for share in self.shares_at(second, generation_seconds))))


SCENARIOS = {
    item.name: item for item in (
        Scenario("balanced", (BALANCED,)),
        Scenario("ns_heavy", (NS_HEAVY,)),
        Scenario("ew_heavy", (EW_HEAVY,)),
        Scenario("changing", (BALANCED, NS_HEAVY, EW_HEAVY)),
    )
}


def _route_tree(source: Path) -> ET.Element:
    root = ET.parse(source).getroot()
    for flow in root.findall("flow"):
        root.remove(flow)
    if len(root.findall("route")) != 4 or not root.findall("vType"):
        raise ValueError("expected four canonical routes and a vehicle type")
    return root


def demand_bytes(scenario: Scenario, seed: int, generation_seconds: int, *, source: Path = SOURCE_ROUTES) -> bytes:
    """Generate explicit scheduled vehicles from the original routes and type."""
    if generation_seconds <= 0:
        raise ValueError("generation_seconds must be positive")
    root = _route_tree(source)
    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    for second in range(generation_seconds):
        probabilities = scenario.probabilities_at(second, generation_seconds)
        for approach in APPROACHES:
            if rng.random() < probabilities[approach]:
                counts[approach] += 1
                ET.SubElement(root, "vehicle", {
                    "id": f"{approach}_flow.{counts[approach]}",
                    "type": "passenger",
                    "route": f"{approach}_to_{DESTINATIONS[approach]}",
                    "depart": str(second),
                    "departLane": "best",
                })
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def prepare_demand(path: Path, scenario: Scenario, seed: int, generation_seconds: int) -> dict[str, int]:
    """Write a route file once, or verify that an existing one is identical."""
    content = demand_bytes(scenario, seed, generation_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"existing demand differs: {path}")
    return scheduled_counts(path)


def scheduled_counts(path: Path) -> dict[str, int]:
    """Count explicit route vehicles by their immutable approach ID prefix."""
    counts: Counter[str] = Counter()
    for vehicle in ET.parse(path).getroot().findall("vehicle"):
        approach = vehicle.attrib["id"].split("_flow.", 1)[0]
        if approach not in APPROACHES:
            raise ValueError(f"unexpected vehicle approach: {approach}")
        counts[approach] += 1
    return {approach: counts[approach] for approach in APPROACHES}

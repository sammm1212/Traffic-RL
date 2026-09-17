"""Traffic-demand scenario preparation shared by controller experiments."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMAND_PATH = ROOT / "simulation" / "routes" / "demand.rou.xml"


def write_demand_scenario(
    destination: Path,
    *,
    generation_seconds: int,
    source: Path = DEFAULT_DEMAND_PATH,
) -> Path:
    """Copy the canonical demand and stop every flow at a common time.

    The probabilities, routes, and vehicle parameters remain unchanged. The
    resulting scenario can therefore be reused by any future controller.
    """
    if generation_seconds <= 0:
        raise ValueError("generation_seconds must be a positive integer")

    tree = ET.parse(source)
    flows = tree.findall("flow")
    if not flows:
        raise ValueError(f"demand file contains no flows: {source}")
    for flow in flows:
        flow.set("end", str(generation_seconds))

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination

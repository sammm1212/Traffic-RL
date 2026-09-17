from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "simulation" / "network"


class ConfigurationTests(unittest.TestCase):
    def test_fixed_time_program_has_safe_cycle(self) -> None:
        phases = ET.parse(NETWORK / "intersection.tll.xml").findall(".//phase")

        self.assertEqual(
            [(phase.get("duration"), phase.get("state")) for phase in phases],
            [
                # Controlled links are ordered north, east, south, west.
                ("30", "GrGr"),
                ("3", "yryr"),
                ("1", "rrrr"),
                ("30", "rGrG"),
                ("3", "ryry"),
                ("1", "rrrr"),
            ],
        )

    def test_demand_covers_all_straight_through_routes(self) -> None:
        routes_file = ROOT / "simulation" / "routes" / "demand.rou.xml"
        routes = {
            route.get("id"): route.get("edges")
            for route in ET.parse(routes_file).findall("route")
        }

        self.assertEqual(
            routes,
            {
                "north_to_south": "north_in south_out",
                "south_to_north": "south_in north_out",
                "east_to_west": "east_in west_out",
                "west_to_east": "west_in east_out",
            },
        )

    def test_connections_are_straight_through(self) -> None:
        connections = ET.parse(NETWORK / "intersection.con.xml").findall("connection")

        self.assertEqual(len(connections), 4)
        self.assertEqual(
            {(connection.get("from"), connection.get("to")) for connection in connections},
            {
                ("north_in", "south_out"),
                ("south_in", "north_out"),
                ("east_in", "west_out"),
                ("west_in", "east_out"),
            },
        )

    def test_compiled_network_signal_controls_all_movements(self) -> None:
        connections = ET.parse(NETWORK / "intersection.net.xml").findall("connection")
        controlled = [connection for connection in connections if connection.get("tl") == "center"]

        self.assertEqual(len(controlled), 4)
        self.assertEqual(
            {connection.get("linkIndex") for connection in controlled},
            {"0", "1", "2", "3"},
        )

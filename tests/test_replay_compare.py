"""Checks for the immutable final-evaluation comparison replay."""

import csv
import json
from pathlib import Path
from shutil import copy2

import pytest

from src.visualization.replay import JunctionPanel
from src.visualization.replay_compare import (
    CAR_LENGTH, CAR_PITCH, CAR_WIDTH, CHECKPOINT_SHA256, MAP_SIZE, ROAD_WIDTH,
    CompareApp, PairPlayer, SPEEDS, SHORT_PHASE_LABELS, STOP_GAP, demand_period, digest,
    load_pair, parse_args, queue_geometry, summary_rows,
)


@pytest.fixture(scope="module")
def pair():
    return load_pair("balanced", 4000)


def test_pair_selection_provenance_and_metrics(pair):
    assert pair.duration == 300
    assert pair.demand_sha256 == digest(Path("results/final_evaluation/demand/balanced/seed_4000.rou.xml"))
    assert CHECKPOINT_SHA256 == digest(Path("results/training/dqn_100_episode_min_green.pt"))
    assert pair.fixed[0].second == pair.dqn[0].second == 0
    assert pair.fixed[-1].second == pair.dqn[-1].second == 299
    assert pair.fixed[-1].replay.throughput == 109
    assert pair.dqn[-1].replay.throughput == 112
    assert pair.fixed[-1].waiting_per_inserted == pytest.approx(1301 / 131)
    assert pair.queue_scale == max(frame.replay.queues[direction]
                                   for recording in (pair.fixed, pair.dqn)
                                   for frame in recording for direction in frame.replay.queues)
    assert parse_args([]).scenario == "balanced"
    assert parse_args([]).seed == 4000
    with pytest.raises(SystemExit):
        parse_args(["--seed", "3999"])


def test_shared_playback_seek_speed_and_restart(pair):
    player = PairPlayer(pair)
    player.set_speed(5)
    player.advance(0.1)
    player.advance(0.1)
    assert player.second == 1
    assert pair.fixed[player.second].second == pair.dqn[player.second].second
    player.toggle()
    player.advance(10)
    assert player.second == 1
    player.seek(120)
    assert player.second == 120 and not player.playing
    assert pair.fixed[player.second].replay.time == pair.dqn[player.second].replay.time
    player.seek(999)
    assert player.second == 299 and not player.playing
    player.restart()
    assert player.second == 0 and player.playing and player.speed == 5
    player.advance(60)
    assert player.second == 299 and not player.playing


def test_ten_times_uses_elapsed_time_and_stops_at_summary(pair):
    player = PairPlayer(pair)
    assert SPEEDS == (1, 2, 5, 10)
    player.set_speed(10)
    player.advance(0.09)
    assert player.second == 0
    player.advance(0.11)
    assert player.second == 2
    player.advance(12.0)
    assert player.second == 122
    assert pair.fixed[player.second].second == pair.dqn[player.second].second
    player.toggle()
    player.advance(10)
    assert player.second == 122
    player.seek(295)
    assert not player.playing
    player.toggle()
    player.advance(0.5)
    assert player.second == 299 and not player.playing
    player.advance(20)
    assert player.second == 299
    assert pair.fixed[player.second].replay.throughput == 109
    assert pair.dqn[player.second].replay.throughput == 112
    player.restart()
    assert player.second == 0 and player.playing and player.speed == 10
    player.advance(30)
    assert player.second == 299 and not player.playing


def test_queue_geometry_incoming_lanes_and_overflow():
    import pygame
    area = pygame.Rect(0, 0, MAP_SIZE, MAP_SIZE)
    cx, cy = area.center
    expected = {"north": ((cx - ROAD_WIDTH // 4, cy - ROAD_WIDTH // 2), (0, -1)),
                "south": ((cx + ROAD_WIDTH // 4, cy + ROAD_WIDTH // 2), (0, 1)),
                "east": ((cx + ROAD_WIDTH // 2, cy - ROAD_WIDTH // 4), (1, 0)),
                "west": ((cx - ROAD_WIDTH // 2, cy + ROAD_WIDTH // 4), (-1, 0))}
    capacities = []
    for approach, (stop, backwards) in expected.items():
        geometry = queue_geometry(area, ROAD_WIDTH, approach, 3)
        assert len(geometry.centres) == 3 and geometry.overflow == 0
        for index, centre in enumerate(geometry.centres):
            assert (centre[0] - stop[0]) * backwards[0] + (centre[1] - stop[1]) * backwards[1] >= STOP_GAP + CAR_LENGTH // 2
            assert centre == (geometry.centres[0][0] + index * backwards[0] * CAR_PITCH,
                              geometry.centres[0][1] + index * backwards[1] * CAR_PITCH)
            assert area.collidepoint(centre)
        long = queue_geometry(area, ROAD_WIDTH, approach, 16)
        assert len(long.centres) + long.overflow == 16
        assert long.overflow > 0 and area.collidepoint(long.overflow_centre)
        capacities.append(len(long.centres))
    assert len(set(capacities)) == 1
    assert queue_geometry(area, ROAD_WIDTH, "north", 0).centres == ()


def test_phase_colours_periods_and_summary(pair):
    panel = object.__new__(JunctionPanel)
    for phase in (0, 1, 2, 3, 4, 5):
        ns = panel._signal_colour(phase, movement="ns")
        ew = panel._signal_colour(phase, movement="ew")
        assert ns == (panel.GREEN if phase == 0 else panel.YELLOW if phase == 1 else panel.RED)
        assert ew == (panel.GREEN if phase == 3 else panel.YELLOW if phase == 4 else panel.RED)
    assert demand_period("changing", 99).startswith("Balanced")
    assert demand_period("changing", 100).startswith("NS-heavy")
    assert demand_period("changing", 200).startswith("EW-heavy")
    assert summary_rows(pair)[0] == ("Completed vehicles", "109", "112", "+3")
    assert summary_rows(pair)[2][0] == "Queue s / inserted"
    assert summary_rows(pair)[3][3] == "+6"
    assert [SHORT_PHASE_LABELS[index] for index in range(6)] == [
        "NS GREEN", "NS YELLOW", "ALL RED", "EW GREEN", "EW YELLOW", "ALL RED"]


def test_car_orientation_signal_pixels_and_panel_bounds(pair, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    try:
        app = CompareApp(pair)
        app.panel = JunctionPanel(pygame)
        screen = pygame.Surface((app.WIDTH, app.HEIGHT))
        for approach, windscreen_offset in {
            "north": (0, CAR_LENGTH // 2 - 4),
            "south": (0, -CAR_LENGTH // 2 + 4),
            "east": (-CAR_LENGTH // 2 + 4, 0),
            "west": (CAR_LENGTH // 2 - 4, 0),
        }.items():
            screen.fill((0, 0, 0))
            app._car(screen, (30, 30), approach)
            assert screen.get_at((30, 30))[:3] == (89, 182, 239)
            x, y = windscreen_offset
            assert screen.get_at((30 + x, 30 + y))[:3] == (25, 60, 83)
            extent = (CAR_WIDTH, CAR_LENGTH) if approach in ("north", "south") else (CAR_LENGTH, CAR_WIDTH)
            assert extent[0] != extent[1]
        area = pygame.Rect(35, 158, MAP_SIZE, MAP_SIZE)
        signal_positions = ((area.centerx - ROAD_WIDTH // 2 - 17, area.centery - ROAD_WIDTH // 2 - 5),
                            (area.centerx + ROAD_WIDTH // 2 + 5, area.centery - ROAD_WIDTH // 2 - 17))
        for phase in range(6):
            screen.fill((0, 0, 0))
            app._signals(screen, area, phase)
            assert screen.get_at(signal_positions[0])[:3] == app.panel._signal_colour(phase, movement="ns")
            assert screen.get_at(signal_positions[1])[:3] == app.panel._signal_colour(phase, movement="ew")
        app.draw(screen)
        assert pygame.Rect(20, 103, 610, 500).contains(pygame.Rect(35, 158, MAP_SIZE, MAP_SIZE))
        assert pygame.Rect(650, 103, 610, 500).contains(pygame.Rect(665, 158, MAP_SIZE, MAP_SIZE))
        assert app.progress.bottom < min(button.y for button, _, _ in app.buttons)
        assert max(button.right for button, _, _ in app.buttons) < app.WIDTH
        assert max(button.bottom for button, _, _ in app.buttons) < app.HEIGHT
    finally:
        pygame.quit()


def test_both_panels_draw_recorded_queue_counts_with_identical_rules(pair, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    try:
        app = CompareApp(pair)
        screen = pygame.Surface((app.WIDTH, app.HEIGHT))
        drawn = []
        monkeypatch.setattr(app, "_car", lambda _screen, centre, approach: drawn.append((centre, approach)))
        for frame, area in ((pair.fixed[200], pygame.Rect(35, 158, MAP_SIZE, MAP_SIZE)),
                            (pair.dqn[200], pygame.Rect(665, 158, MAP_SIZE, MAP_SIZE))):
            drawn.clear()
            app._queues(screen, area, frame)
            for approach in ("north", "south", "east", "west"):
                expected = queue_geometry(area, ROAD_WIDTH, approach, frame.replay.queues[approach])
                assert [centre for centre, direction in drawn if direction == approach] == list(expected.centres)
                assert len(expected.centres) + expected.overflow == frame.replay.queues[approach]
    finally:
        pygame.quit()


@pytest.fixture
def copied_pair(tmp_path):
    data = tmp_path / "final_evaluation"
    (data / "demand/balanced").mkdir(parents=True)
    (data / "records/balanced").mkdir(parents=True)
    copy2("results/final_evaluation/episodes.csv", data / "episodes.csv")
    copy2("results/final_evaluation/demand/balanced/seed_4000.rou.xml",
          data / "demand/balanced/seed_4000.rou.xml")
    for controller in ("fixed_time", "dqn_15"):
        for suffix in ("seconds.csv", "manifest.json"):
            name = f"seed_4000_{controller}_{suffix}"
            copy2(Path("results/final_evaluation/records/balanced") / name,
                  data / "records/balanced" / name)
    return data


def test_rejects_missing_and_mismatched_demand(copied_pair):
    manifest = copied_pair / "records/balanced/seed_4000_dqn_15_manifest.json"
    manifest.unlink()
    with pytest.raises(ValueError, match="missing or malformed"):
        load_pair(data=copied_pair)
    copy2("results/final_evaluation/records/balanced/seed_4000_dqn_15_manifest.json", manifest)
    value = json.loads(manifest.read_text())
    value["demand_sha256"] = "wrong"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_pair(data=copied_pair)


def test_rejects_timeline_even_with_updated_record_hash(copied_pair):
    record = copied_pair / "records/balanced/seed_4000_fixed_time_seconds.csv"
    manifest = copied_pair / "records/balanced/seed_4000_fixed_time_manifest.json"
    with record.open(newline="") as source:
        rows = list(csv.DictReader(source))
    rows[10]["second"] = "9"
    with record.open("w", newline="") as output:
        writer = csv.DictWriter(output, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    value = json.loads(manifest.read_text())
    value["seconds_sha256"] = digest(record)
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="timeline mismatch"):
        load_pair(data=copied_pair)


def test_pygame_dummy_controls_and_draw(pair, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = CompareApp(pair)
    pg = app.pg
    pg.init()
    try:
        screen = pg.display.set_mode((app.WIDTH, app.HEIGHT))
        app.panel = JunctionPanel(pg)
        app.draw(screen)
        assert app.handle_key(pg.K_2)
        assert app.player.speed == 2
        assert app.handle_key(pg.K_0)
        assert app.player.speed == 10
        app.draw(screen)
        ten_button = next(bounds for bounds, action, value in app.buttons if action == "speed" and value == 10)
        app.handle_click(ten_button.center)
        assert app.player.speed == 10
        assert app.handle_key(pg.K_RIGHT)
        assert app.player.second == 5
        app.draw(screen)
        app.handle_click((app.progress.x + app.progress.width // 2, app.progress.y + 5))
        assert app.player.second in (149, 150)
        assert pair.fixed[app.player.second].second == pair.dqn[app.player.second].second
        app.handle_key(pg.K_r)
        assert app.player.second == 0 and app.player.speed == 10
        app.player.seek(299)
        app.draw(screen)
        assert not app.handle_key(pg.K_ESCAPE)
    finally:
        pg.quit()

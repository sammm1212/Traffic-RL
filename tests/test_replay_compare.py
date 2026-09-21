"""Checks for the immutable final-evaluation comparison replay."""

import csv
import json
from pathlib import Path
from shutil import copy2

import pytest

from src.visualization.replay import JunctionPanel
from src.visualization.replay_compare import (
    CHECKPOINT_SHA256, CompareApp, PairPlayer, demand_period, digest, load_pair,
    parse_args, summary_rows,
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
        assert app.handle_key(pg.K_RIGHT)
        assert app.player.second == 5
        app.draw(screen)
        app.handle_click((app.progress.x + app.progress.width // 2, app.progress.y + 5))
        assert app.player.second in (149, 150)
        assert pair.fixed[app.player.second].second == pair.dqn[app.player.second].second
        app.handle_key(pg.K_r)
        assert app.player.second == 0
        app.player.seek(299)
        app.draw(screen)
        assert not app.handle_key(pg.K_ESCAPE)
    finally:
        pg.quit()

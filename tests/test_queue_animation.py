"""Schematic motion must remain bounded and independent of replay metrics."""

import pytest

from src.visualization.replay import APPROACHES, JunctionPanel, ReplayFrame
from src.visualization.replay_compare import (
    ANIMATION_SECONDS, CAR_COLOURS, MAP_SIZE, ROAD_WIDTH, CompareApp, QueueAnimation,
    load_pair, queue_geometry,
)


def frame(**queues):
    phase = queues.pop("phase", 0)
    return ReplayFrame(1, {name: queues.get(name, 0) for name in APPROACHES}, phase, 17)


@pytest.fixture
def area():
    import pygame
    return pygame.Rect(35, 158, MAP_SIZE, MAP_SIZE)


@pytest.mark.parametrize("approach", APPROACHES)
def test_arrivals_join_rear_without_overlap(area, approach):
    animation = QueueAnimation(area)
    before, after = frame(**{approach: 2}), frame(**{approach: 4})
    animation.reset(before)
    old_ids = [car.identifier for car in animation.queues[approach]]
    animation.advance(before, after)
    cars = animation.queues[approach]
    assert len(cars) == 4
    assert [car.identifier for car in cars[:2]] == old_ids
    assert [car.start_slot for car in cars[2:]] == [4, 5]
    assert [car.end_slot for car in cars] == [0, 1, 2, 3]
    origin = queue_geometry(area, ROAD_WIDTH, approach, 1).centres[0]
    starts = [point for direction, point, _ in animation.positions(0) if direction == approach]
    ends = [point for direction, point, _ in animation.positions(ANIMATION_SECONDS) if direction == approach]
    assert ends == list(queue_geometry(area, ROAD_WIDTH, approach, 4).centres)
    assert len(set(ends)) == 4
    for start, end in zip(starts[2:], ends[2:]):
        assert (abs(start[0] - origin[0]) + abs(start[1] - origin[1])) > (
            abs(end[0] - origin[0]) + abs(end[1] - origin[1]))
        assert area.collidepoint(start)
    assert after.total_queue == 4 and after.throughput == 17


@pytest.mark.parametrize("approach,green", [("north", 0), ("south", 0),
                                            ("east", 3), ("west", 3)])
def test_green_departures_take_front_cars_and_advance_survivors(area, approach, green):
    animation = QueueAnimation(area)
    before, after = frame(**{approach: 5}), frame(**{approach: 3}, phase=green)
    animation.reset(before)
    ids = [car.identifier for car in animation.queues[approach]]
    animation.advance(before, after)
    assert [car.identifier for car in animation.departing] == ids[:2]
    assert [car.identifier for car in animation.queues[approach]] == ids[2:]
    assert [car.start_slot for car in animation.queues[approach]] == [2, 3, 4]
    assert [car.end_slot for car in animation.queues[approach]] == [0, 1, 2]
    start = animation.positions(0)
    middle = animation.positions(ANIMATION_SECONDS / 2)
    assert len({point for _, point, _ in middle}) == len(middle)
    for departure in animation.departing:
        assert departure.end_slot < departure.start_slot
    assert len([car for car in animation.positions(ANIMATION_SECONDS) if car[2]]) == 0
    assert len(start) == 5 and after.total_queue == 3


@pytest.mark.parametrize("phase", [1, 2, 3, 4, 5])
def test_non_green_ns_decrease_never_crosses(area, phase):
    animation = QueueAnimation(area)
    before, after = frame(north=3), frame(north=1, phase=phase)
    animation.reset(before)
    animation.advance(before, after)
    assert animation.departing == []
    assert len(animation.queues["north"]) == 1
    assert not any(crossing for _, _, crossing in animation.positions(0))


@pytest.mark.parametrize("phase", [0, 1, 2, 4, 5])
def test_non_green_ew_decrease_never_crosses(area, phase):
    animation = QueueAnimation(area)
    before, after = frame(east=3), frame(east=1, phase=phase)
    animation.reset(before)
    animation.advance(before, after)
    assert animation.departing == []


def test_overflow_bounds_visual_objects_and_keeps_recorded_count(area):
    animation = QueueAnimation(area)
    before, after = frame(north=20), frame(north=30)
    animation.reset(before)
    animation.advance(before, after)
    geometry = queue_geometry(area, ROAD_WIDTH, "north", 30)
    assert len(animation.queues["north"]) == len(geometry.centres)
    assert geometry.overflow == 30 - len(geometry.centres)
    assert after.total_queue == 30
    assert all(area.collidepoint(point) for _, point, _ in animation.positions(0))
    for progress in (0, .1, .2, .32, ANIMATION_SECONDS):
        points = [point for _, point, _ in animation.positions(progress)]
        assert len(points) == len(set(points))

    # Revealed markers in a shrinking overflow queue wait for space at the rear.
    animation.reset(frame(north=12))
    animation.advance(frame(north=12), frame(north=9, phase=0))
    for progress in (0, .1, .2, .32, ANIMATION_SECONDS):
        points = [point for _, point, _ in animation.positions(progress)]
        assert len(points) == len(set(points))


def test_player_speed_pause_seek_restart_and_skipped_frames(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pair = load_pair()
    app = CompareApp(pair)
    app.pg.init()
    app.panel = JunctionPanel(app.pg)
    screen = app.pg.Surface((app.WIDTH, app.HEIGHT))
    app.player.advance(1.1)
    app.draw(screen)
    assert app._animation_second == 1
    frozen = app._animations[0].positions(app.player._fraction)
    app.player.toggle()
    app.player.advance(20)
    assert app._animations[0].positions(app.player._fraction) == frozen
    app.player.toggle()
    fraction = app.player._fraction
    app.player.set_speed(10)
    assert app.player._fraction == fraction
    app.player.advance(0.2)
    app.draw(screen)
    assert app._animation_second == app.player.second == 3
    for animation, recording in zip(app._animations, (pair.fixed, pair.dqn)):
        assert len(animation.queues["north"]) == len(queue_geometry(
            animation.area, ROAD_WIDTH, "north", recording[3].replay.queues["north"]).centres)
        assert not animation.departing
    app.handle_key(app.pg.K_LEFT)
    assert app._animation_second == app.player.second == 0
    app.handle_key(app.pg.K_r)
    assert app._animation_second == app.player.second == 0
    app.player.seek(299)
    app.draw(screen)
    assert not app.player.playing and not any(a.departing for a in app._animations)
    app.set_pair(load_pair("changing", 4000))
    assert app.player.second == 0 and app._animation_second == 0
    assert app.pair.scenario == "changing"
    app.pg.quit()


def test_duration_scales_with_replay_speed_and_both_panels_use_same_rules(area):
    for speed in (1, 2, 5, 10):
        assert ANIMATION_SECONDS / speed == pytest.approx({1: .4, 2: .2, 5: .08, 10: .04}[speed])
    left, right = QueueAnimation(area), QueueAnimation(area.move(630, 0))
    before, after = frame(west=4), frame(west=2, phase=3)
    for animation in (left, right):
        animation.reset(before)
        animation.advance(before, after)
    assert [(car.start_slot, car.end_slot, car.crossing) for car in left.departing] == [
        (car.start_slot, car.end_slot, car.crossing) for car in right.departing]


def test_colours_are_deterministic_and_stable_for_visual_car_lifetime(area):
    context = ("balanced", 4000, "fixed")
    before, arrival = frame(north=3), frame(north=5)
    departure = frame(north=3, phase=0)
    first = QueueAnimation(area, context)
    first.reset(before)
    initial = {car.identifier: car.colour for car in first.queues["north"]}
    first.advance(before, arrival)
    spawned = {car.identifier: car.colour for car in first.queues["north"]}
    assert set(spawned.values()) <= set(CAR_COLOURS)
    assert all(spawned[identifier] == colour for identifier, colour in initial.items())
    assert len(set(spawned.values())) > 1
    first.advance(arrival, departure)
    for car in (*first.departing, *first.queues["north"]):
        assert car.colour == spawned[car.identifier]

    second = QueueAnimation(area, context)
    second.reset(before)
    second.advance(before, arrival)
    assert [car.colour for car in second.queues["north"]] == list(spawned.values())
    first.reset(before)
    first.advance(before, arrival)
    assert [car.colour for car in first.queues["north"]] == list(spawned.values())


def test_colour_context_and_renderer(monkeypatch, area):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pair = load_pair()
    app = CompareApp(pair)
    assert app._animations[0].context == ("balanced", 4000, "fixed")
    assert app._animations[1].context == ("balanced", 4000, "dqn")
    recorded = []
    monkeypatch.setattr(app, "_car", lambda _screen, centre, approach, colour=CAR_COLOURS[0]:
                        recorded.append((centre, approach, colour)))
    app.pg.init()
    try:
        animation = QueueAnimation(area, ("balanced", 4000, "fixed"))
        animation.reset(frame(north=3))
        app._queues(app.pg.Surface((app.WIDTH, app.HEIGHT)), area,
                    pair.fixed[0], animation)
        assert [colour for _, _, colour in recorded] == [
            car.colour for car in animation.queues["north"]]
    finally:
        app.pg.quit()

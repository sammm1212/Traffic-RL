"""Accounting tests for the frozen signal-allocation diagnostic."""

import pytest

from evaluate_signal_allocation import (
    _decision_summary, _paired, _window_comparisons, _windows, interval,
    phase_kind, queue_relation, summarize_seconds, window_index,
)


def second(index, phase, ns=0, ew=0, changed=0):
    kind, direction = phase_kind(phase)
    row = {"second": index, "scenario": "changing", "seed": 3000,
           "controller": "dqn", "phase_kind": kind,
           "active_green": direction or "", "executed_direction_change": changed,
           "ns_queue": ns, "ew_queue": ew, "throughput_cumulative": 0}
    for approach in ("north", "south", "east", "west"):
        row[f"{approach}_scheduled"] = 0
        row[f"{approach}_inserted"] = 0
        row[f"{approach}_completed"] = 0
    return row


def test_phase_mapping_and_unknown_phase():
    assert [phase_kind(i)[0] for i in range(6)] == [
        "ns_green", "yellow", "all_red", "ew_green", "yellow", "all_red"]
    with pytest.raises(ValueError):
        phase_kind(6)


def test_duration_switch_and_directional_queues():
    rows = [second(0, 0, 2, 1), second(1, 1, 1, 2),
            second(2, 2), second(3, 3, 0, 4, 1), second(4, 3, 0, 3)]
    rows[0]["north_inserted"] = 1
    rows[3]["east_completed"] = 1
    result = summarize_seconds(rows)
    assert (result["ns_green_seconds"], result["ew_green_seconds"],
            result["yellow_seconds"], result["all_red_seconds"]) == (1, 2, 1, 1)
    assert result["transition_seconds"] == 2
    assert result["executed_direction_changes"] == 1
    assert result["ew_green_share"] == pytest.approx(2 / 3)
    assert result["ns_queue_seconds"] == 3
    assert result["ew_queue_seconds"] == 10
    assert result["ns_inserted"] == result["ew_completed"] == 1
    assert result["equal_queue_share"] == pytest.approx(1 / 5)


def test_equal_zero_and_eligibility_decisions():
    assert queue_relation(0, 0) == "equal"
    assert queue_relation(1, 0) == "NS_greater"
    assert queue_relation(0, 1) == "EW_greater"
    decisions = [
        {"scenario": "balanced", "queue_relation": "equal", "switch_eligible": 0,
         "requested_action": "EW", "switch_blocked": 1},
        {"scenario": "balanced", "queue_relation": "equal", "switch_eligible": 1,
         "requested_action": "NS", "switch_blocked": 0},
    ]
    rows = _decision_summary(decisions)
    blocked = next(r for r in rows if r["queue_relation"] == "equal" and r["switch_eligible"] == 0)
    eligible = next(r for r in rows if r["queue_relation"] == "equal" and r["switch_eligible"] == 1)
    assert blocked["blocked_switches"] == 1
    assert eligible["requested_ns_share"] == 1


def test_window_boundaries_and_crossing_transition():
    assert [window_index(i) for i in (0, 99, 100, 199, 200, 299)] == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        window_index(300)
    rows = [second(i, 0) for i in range(300)]
    for i, phase in ((99, 1), (100, 1), (101, 1), (102, 2), (103, 3)):
        rows[i] = second(i, phase, changed=int(i == 103))
    windows = _windows(rows)
    assert [r["transition_seconds"] for r in windows] == [1, 3, 0]
    assert [r["executed_direction_changes"] for r in windows] == [0, 1, 0]
    assert windows[1]["ns_queue_at_start"] == windows[0]["ns_queue_at_end"]
    assert windows[2]["ew_queue_at_start"] == windows[1]["ew_queue_at_end"]
    assert all(sum(r[f"{p}_seconds"] for p in ("ns_green", "ew_green", "yellow", "all_red")) == 100 for r in windows)


def test_within_seed_window_changes():
    rows = []
    for seed, shares in ((3000, (0.2, 0.3, 0.5)), (3001, (0.4, 0.6, 0.7))):
        for index, share in enumerate(shares, 1):
            row = {"seed": seed, "controller": "dqn", "window": index,
                   "ew_green_share": share}
            for key in ("ns_green_seconds", "ew_green_seconds", "yellow_seconds", "all_red_seconds",
                        "executed_direction_changes", "ns_green_share", "mean_ns_queue", "mean_ew_queue",
                        "ns_scheduled", "ew_scheduled", "ns_inserted", "ew_inserted",
                        "ns_completed", "ew_completed", "ns_queue_at_start", "ew_queue_at_start",
                        "ns_queue_at_end", "ew_queue_at_end"):
                row[key] = 0
            rows.append(row)
    changes, means = _window_comparisons(rows)
    assert [round(r["ew_share_change"], 3) for r in changes if r["first_window"] == 2] == [0.2, 0.1]
    assert interval([r["ew_share_change"] for r in changes if r["first_window"] == 2])["mean"] == pytest.approx(0.15)
    assert next(r for r in means if r["window"] == 3)["ew_green_share"] == pytest.approx(0.6)


def test_paired_demand_and_seed_difference():
    base = {"scenario": "balanced", "seed": 3000, "demand_sha256": "same"}
    values = {key: 1 for key in ("ew_green_seconds", "transition_seconds", "executed_direction_changes", "ns_queue_seconds", "ew_queue_seconds", "vehicles_completed")}
    pairs, summary = _paired([{**base, **values, "controller": "dqn"},
                              {**base, **values, "controller": "fixed_time", "ew_green_seconds": 2}])
    assert next(r for r in pairs if r["metric"] == "ew_green_seconds")["dqn_minus_fixed"] == -1
    assert next(r for r in summary if r["metric"] == "ew_green_seconds")["n"] == 1
    with pytest.raises(ValueError, match="hash mismatch"):
        _paired([{**base, **values, "controller": "dqn"},
                 {**base, **values, "controller": "fixed_time", "demand_sha256": "other"}])
    assert interval([1.0, 3.0])["mean"] == 2.0

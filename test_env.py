"""Smoke-test the SUMO/TraCI simulation layer from the repository root."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import traci

from gymnasium.utils.env_checker import check_env

from src.environment.traffic_env import TrafficEnvironment
from src.simulation.run import DEFAULT_SEED, build_sumo_command
from src.simulation.traffic import (
    INCOMING_LANES,
    TRAFFIC_LIGHT_ID,
    TrafficSimulation,
)


DEFAULT_STEPS = 40


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse smoke-test options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="run with sumo-gui")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Start SUMO, print one observation, and always close TraCI."""
    args = parse_args(argv)

    if args.steps <= 0:
        raise ValueError("steps must be a positive integer")

    started = False

    try:
        sumo_command = build_sumo_command(
            gui=args.gui,
            seed=args.seed,
            steps=args.steps,
        )

        traci.start(sumo_command)

        started = True

        simulation = TrafficSimulation(
            traci,
            reload_args=sumo_command[1:],
        )

        # TraCI exposes the complete SUMO phase program, including safe yellow
        # and all-red transitions, so the smoke test can display it for review.
        logics = traci.trafficlight.getAllProgramLogics("center")

        for logic in logics:
            print("Program:", logic.programID)

            for i, phase in enumerate(logic.phases):
                print(
                    f"Phase {i}: "
                    f"duration={phase.duration}, "
                    f"state={phase.state}"
                )

        available_lanes = set(traci.lane.getIDList())
        missing_lanes = set(INCOMING_LANES.values()) - available_lanes

        if missing_lanes:
            raise RuntimeError(
                f"SUMO network is missing lanes: {sorted(missing_lanes)}"
            )

        if TRAFFIC_LIGHT_ID not in traci.trafficlight.getIDList():
            raise RuntimeError(
                f"SUMO network is missing traffic light {TRAFFIC_LIGHT_ID!r}"
            )

        

        simulation.set_traffic_light(0)
        print(simulation.observe().light_phase)

        simulation.set_traffic_light(1)
        print(simulation.observe().light_phase)

        for _ in range(args.steps):
            simulation.step()

        metrics = simulation.observe()

        print(f"Incoming lanes: {INCOMING_LANES}")
        print(f"Traffic-light ID: {TRAFFIC_LIGHT_ID}")
        print(f"Simulation time: {metrics.time:.1f}")

        for approach in ("north", "south", "east", "west"):
            print(
                f"{approach.capitalize()} queue length: "
                f"{metrics.approaches[approach].queue_length}"
            )

        print(f"Traffic-light phase: {metrics.light_phase}")
        print(f"Vehicles generated: {metrics.vehicles_generated}")
        print(f"Vehicles completed: {metrics.vehicles_completed}")

        # Gymnasium's checker validates reset/step return values and declared
        # observation/action spaces against a live SUMO-backed environment.
        env = TrafficEnvironment(
            simulation=simulation,
            decision_interval=5,
        )

        check_env(env)

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)

        for _ in range(20):
            simulation.step()

        print("Time before reset:", simulation.observe().time)

        observation, info = env.reset()

        print("Time after reset:", simulation.observe().time)
        print("Observation after reset:", observation)

        metrics = simulation.observe()

        print("Vehicles generated after reset:", metrics.vehicles_generated)
        print("Vehicles completed after reset:", metrics.vehicles_completed)

        observation, info = env.reset()

        print("Initial observation:", observation)

        # Exercise the complete N/S-to-E/W safe transition through TraCI.

        simulation.set_traffic_light(0)

        before = simulation.observe().time

        observation, reward, terminated, truncated, info = env.step(1)

        after = simulation.observe().time

        print("\nN/S -> E/W")
        print("Final phase:", simulation.observe().light_phase)
        print("Time advanced:", after - before)
        print("Observation:", observation)
        print("Reward:", reward)


        # Exercise the reverse E/W-to-N/S safe transition.

        before = simulation.observe().time

        observation, reward, terminated, truncated, info = env.step(0)

        after = simulation.observe().time

        print("\nE/W -> N/S")
        print("Final phase:", simulation.observe().light_phase)
        print("Time advanced:", after - before)
        print("Observation:", observation)
        print("Reward:", reward)


    finally:
        if started:
            traci.close()


if __name__ == "__main__":
    main()

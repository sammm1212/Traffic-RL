"""Run a checkpoint with SUMO process and TraCI startup diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryFile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import traci

from src.simulation.run import build_sumo_command
from validate_extended import evaluate_one, load_policy


def socket_state(pid: int) -> str:
    """Describe SUMO TCP sockets without opening a probe connection."""
    result = subprocess.run(
        ["lsof", "-nP", "-a", "-p", str(pid), "-iTCP"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or f"no TCP socket reported (lsof exit {result.returncode})"


def main() -> None:
    """Print startup details, run selected seeds, and show the full failure traceback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "results/extended_training/checkpoints/episode_0025.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000])
    args = parser.parse_args()

    agent, state = load_policy(args.checkpoint)
    seconds = state["configuration"]["episode_seconds"]
    command = build_sumo_command(gui=False, seed=args.seeds[0], steps=seconds)
    version = subprocess.run([command[0], "--version"], capture_output=True, text=True, check=True)
    print(f"Python: {sys.version.split()[0]}  TraCI: {traci.__version__}", flush=True)
    print(f"SUMO executable: {command[0]}", flush=True)
    print(f"SUMO version: {version.stdout.splitlines()[0]}", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except OSError:
        print("Localhost bind failed before SUMO startup; check terminal socket permissions.", flush=True)
        raise

    full_command = [*command, "--remote-port", str(port)]
    print(f"Port: {port}\nComplete SUMO command: {full_command!r}", flush=True)
    with TemporaryFile(mode="w+t", encoding="utf-8") as log:
        process = subprocess.Popen(full_command, stdout=log, stderr=subprocess.STDOUT)
        connection = None
        try:
            print(f"SUMO PID: {process.pid}; initial exit status: {process.poll()}", flush=True)
            connection = traci.connect(
                port=port, host="127.0.0.1", proc=process, numRetries=3,
            )
            print(f"SUMO TCP sockets after connect:\n{socket_state(process.pid)}", flush=True)
            for seed in args.seeds:
                result = evaluate_one(agent, seed, seconds, connection=connection)
                print(
                    f"Seed {seed}: time={connection.simulation.getTime()}s, {result}",
                    flush=True,
                )
            print(f"SUMO exit status before close: {process.poll()}", flush=True)
        except BaseException:
            print(f"SUMO exit status at failure: {process.poll()}", flush=True)
            print(f"SUMO TCP sockets at failure:\n{socket_state(process.pid)}", flush=True)
            traceback.print_exc()
            raise
        finally:
            try:
                if connection is not None:
                    connection.close(wait=False)
            finally:
                if process.poll() is None and connection is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.seek(0)
                print(f"SUMO final exit status: {process.returncode}", flush=True)
                print(f"SUMO stdout/stderr:\n{log.read().strip() or '(no output)'}", flush=True)


if __name__ == "__main__":
    main()

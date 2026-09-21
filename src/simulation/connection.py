"""Bounded, diagnostic TraCI startup for multi-episode validation."""

from __future__ import annotations

from contextlib import contextmanager
import socket
import subprocess
from tempfile import TemporaryFile
from typing import Iterator, Sequence

import traci


def _stop_process(process: subprocess.Popen, *, terminate: bool = False) -> None:
    """Reap SUMO, terminating it if TraCI could not close it normally."""
    if terminate and process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def sumo_connection(command: Sequence[str]) -> Iterator[traci.connection.Connection]:
    """Launch one SUMO process, expose TraCI, and always reap the process."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    except OSError as exc:
        raise RuntimeError(
            "Cannot bind a localhost TraCI port. Run validation in a terminal "
            "that permits local sockets."
        ) from exc

    full_command = [*command, "--remote-port", str(port)]
    with TemporaryFile(mode="w+t", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                full_command, stdout=log, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise RuntimeError(f"Cannot launch SUMO: {full_command!r}") from exc

        connection = None
        try:
            try:
                connection = traci.connect(
                    port=port, host="127.0.0.1", proc=process, numRetries=3,
                )
            except Exception as exc:
                _stop_process(process, terminate=True)
                log.seek(0)
                output = log.read().strip() or "(no SUMO output)"
                raise RuntimeError(
                    f"TraCI startup failed on port {port}; SUMO exit status "
                    f"{process.returncode}; command: {full_command!r}; "
                    f"SUMO stdout/stderr:\n{output}"
                ) from exc
            yield connection
        finally:
            try:
                if connection is not None:
                    connection.close(wait=False)
            finally:
                _stop_process(process, terminate=connection is None)

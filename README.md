# Traffic RL — Day 1 SUMO Infrastructure

This repository currently contains a deliberately small SUMO simulation: one
four-way, signalised junction with one lane per approach and straight-through
traffic only. SUMO owns the fixed-time signal program; Python connects through
TraCI and advances the simulation without changing the lights.

The Python layer records reusable traffic observations and fixed-time baseline
metrics. No reinforcement-learning code is included at this stage.

## Layout

```text
simulation/
  network/   editable nodes, edges, connections, signal plan, and generated network
  routes/    stochastic demand from all four approaches
  config/    SUMO configuration
src/simulation/        TraCI adapter, metrics, recording, and command-line runner
src/visualization/     standalone playback of recorded traffic episodes
tests/       configuration and runner tests
```

The generated `intersection.net.xml` is committed because SUMO runs from a
compiled network. Its small source XML files remain alongside it so the network
geometry and signal logic can be reviewed and regenerated.

## Setup

Python 3.11 or later is recommended. Create and activate a virtual environment,
then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The `eclipse-sumo` Python package supplies SUMO, `sumo-gui`, `netconvert`, TraCI,
and `sumolib` on supported platforms.

## Run

Headless, for 600 simulated seconds with seed 42:

```bash
python -m src.simulation.run --steps 600 --seed 42
```

With the graphical interface:

```bash
python -m src.simulation.run --gui --steps 600 --seed 42
```

To smoke-test the raw SUMO/TraCI connection and print one observation from
`TrafficSimulation`:

```bash
.venv/bin/python test_env.py
```

The smoke test checks that the canonical incoming lanes and the `center`
traffic light exist before advancing the simulation. Pass `--gui` to inspect
the same run in SUMO's graphical interface.

Optionally record every observation for later visualisation. The extension
selects CSV or JSON format:

```bash
python -m src.simulation.run --steps 600 --seed 42 --record results/run-42.csv
```

`TrafficSimulation.step()` returns a `TrafficMetrics` observation. Its
`state_vector()` method returns north, south, east, and west queues followed by
the current signal phase; this is observation data only, not an RL environment.

The run summary reports **mean queueing delay per generated vehicle**. This is
integrated halted-vehicle occupancy (vehicle-seconds) divided by the number of
vehicles generated. It is deliberately distinct from the per-step
`mean_waiting_time` recording field, which uses SUMO's native continuous
waiting-time values for vehicles currently on the incoming lanes.

`--steps` is the simulation end time in seconds. The demand file defines flows
for up to 3600 seconds; raise both values if a longer experiment is needed.
Using the same seed reproduces the same probabilistic vehicle arrivals.

## Replay a recorded episode

The replay tool reads the existing per-step CSV or JSON recording format. It
does not connect to SUMO or generate traffic state. First create a recording
when running an episode, then replay it:

```bash
python -m src.simulation.run --steps 600 --seed 42 --record results/replays/baseline-42.csv
python -m src.visualization.replay results/replays/baseline-42.csv
```

Use Space or the on-screen button to play/pause, R to restart, number keys 1–4
to select 1×, 2×, 5×, or 10× playback, and Q or Escape to close. The replay
shows recorded queues, signal phase, throughput, and the recording's current
mean waiting-time value, labelled **Current mean vehicle wait**. This live value
applies only to vehicles currently on the incoming lanes and is distinct from
the experiment-level mean queueing delay per generated vehicle. Its
single-recording panel is independent of playback state so a second panel can
later display a comparison recording.

## Multi-seed fixed-time baseline

Run the default 20-episode baseline with consecutive seeds 1 through 20:

```bash
python -m src.experiments.fixed_time_baseline --episodes 20
```

Each episode uses the canonical demand probabilities and fixed-time signal
program. Demand runs for 900 simulated seconds, followed by a 300-second
clearance period. Episode-level CSV output is written to
`results/baseline/fixed_time_baseline.csv`; `--start-seed`,
`--generation-seconds`, `--clearance-seconds`, and `--output` are available for
controlled experiments. Throughput is reported as completed vehicles per
simulated hour over the full 1,200-second episode. Reusing the same seed and
parameters reproduces the same result.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Regenerate the network

After editing one of the network source files, run:

```bash
netconvert \
  --node-files simulation/network/intersection.nod.xml \
  --edge-files simulation/network/intersection.edg.xml \
  --connection-files simulation/network/intersection.con.xml \
  --tllogic-files simulation/network/intersection.tll.xml \
  --output-file simulation/network/intersection.net.xml
```

The signal cycle is 30 seconds north/south green, 3 seconds yellow, 1 second
all-red, 30 seconds east/west green, 3 seconds yellow, and 1 second all-red.

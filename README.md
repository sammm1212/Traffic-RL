# Reinforcement Learning for Adaptive Traffic Signal Control

## Project overview

This project investigates whether a reinforcement-learning traffic-light
controller can improve traffic flow at a simulated four-way junction compared
with traditional control strategies. The deliberately small scenario has one
incoming lane from each compass direction and straight-through traffic only.

[SUMO](https://eclipse.dev/sumo/) provides the microscopic traffic simulation,
and TraCI connects the Python code to the running simulator. Gymnasium exposes
that connection as a reinforcement-learning environment. The current agent is
a Deep Q-Network (DQN) implemented in this repository with PyTorch rather than a
pre-built traffic-RL agent.

## Final evaluation presentation replay

Launch the paired Pygame replay from the repository root:

```bash
.venv/bin/python -m src.visualization.replay_compare --scenario balanced --seed 4000
```

The default is **balanced, seed 4000**, the first seed in the independent
evaluation range. Select `balanced`, `ns_heavy`, `ew_heavy`, or `changing` and
any seed from 4000 through 4029. The viewer reads the existing per-second
records in `results/final_evaluation/records/`; no export, SUMO run, demand
generation, model loading, or training is required. It verifies the frozen
checkpoint hash, recorded CSV hashes, saved demand-file hash, matched
timelines, and final metrics before opening the window. If a record is missing
or invalid, it reports the problem instead of creating a replacement.

Both panels use the same recorded second and queue-bar scale. The bars show
aggregate stopped vehicles by approach, not individual vehicle positions.
The signal colour shows the phase governing the displayed second. During
playback, “Queue s / inserted” means accumulated halted-vehicle seconds
divided by vehicles inserted up to that second. The end card compares the
selected seed only, with DQN-minus-fixed differences; it does not show the
30-seed aggregate result. For changing demand, the label and timeline markers
show the scheduled arrival periods at seconds 0, 100, and 200; existing queues
continue across those boundaries.

Controls: **Space** play/pause; **R** restart; **1**, **2**, **5** playback
speed; **Left/Right** seek by five recorded seconds; click the progress bar
to seek; **Esc** exit. Playback pauses at the final second, and R starts again
without loading SUMO or the checkpoint. For a headless rendering check, set
`SDL_VIDEODRIVER=dummy`; use a normal desktop display for the presentation.

## Research question

> Can a Deep Q-Network learn an adaptive traffic signal policy that reduces
> congestion and vehicle delay compared with fixed-time traffic control?

## Current project status

The following components are present:

- A SUMO network for one signalised four-way junction, with stochastic flows
  from all approaches and reproducible SUMO seeds.
- A 68-second fixed-time signal program: 30 seconds north-south green, three
  seconds yellow, one second all-red, then the equivalent east-west phases.
- A reusable TraCI adapter that advances SUMO, reads incoming-lane queues,
  vehicle counts and waiting times, reads or changes the traffic-light phase,
  and counts departed and completed vehicles.
- Per-step and run-level metrics, optional CSV/JSON recording, and a Pygame
  replay viewer.
- A multi-seed fixed-time experiment and a resumable comparison of fixed-time
  and random controllers.
- A Gymnasium environment with reset and step behavior, a random policy, a
  replay buffer, a PyTorch DQN, and an epsilon-greedy training script.

The DQN path is an early prototype. It can collect transitions and perform
gradient updates, but the repository does not contain saved models, trained
checkpoints, evaluation results, or evidence yet that the learned controller
outperforms either baseline. Its hyperparameters and queue-based reward should
therefore be treated as experimental.

## Reinforcement-learning formulation

### State / observation space

`TrafficEnvironment` declares a five-value, non-negative `float32` observation:

```text
[north_queue, south_queue, east_queue, west_queue, traffic_phase]
```

- The first four values are SUMO's current halted-vehicle counts on the incoming
  lane for each approach.
- `traffic_phase` is SUMO's numeric phase index. In the configured program,
  phase 0 is north-south green, phases 1 and 2 are its yellow and all-red
  transition, phase 3 is east-west green, and phases 4 and 5 are the reverse
  yellow and all-red transition.

### Action space

The action space is `Discrete(2)`. Actions select a target principal green; they
do not mean “keep” and “switch”:

```text
0 = select north-south green (SUMO phase 0)
1 = select east-west green (SUMO phase 3)
```

If the requested green is already active, it is held for the five-second
decision interval. When changing movement, the simulation spends three seconds
on yellow and one second on all-red before applying the new green. This preserves
the configured safety transition while still advancing exactly one decision
interval.

### Reward and episode handling

After each action interval, the reward is calculated exactly as:

```text
reward = -(north_queue + south_queue + east_queue + west_queue)
```

The controller is therefore penalized once per decision for vehicles currently
halted across the junction. The reward does not directly include waiting time,
throughput, switching cost, or discounting; discounting is applied later when
the DQN constructs its learning target.

`reset()` asks TraCI to reload the same SUMO command and clears generated and
completed vehicle counters. `step()` applies the selected target green, returns
the next observation and reward, and truncates the episode when simulated time
reaches 300 seconds. It does not currently use a separate terminal condition.

The DQN uses two hidden layers of 64 units, experience replay, epsilon-greedy
action selection, Smooth L1 loss, discounted bootstrap targets, and a periodically
synchronized target network. These are implemented locally in `src/agents/`.

## Baseline experiments

Baselines establish performance levels against which a trained agent can later
be evaluated:

- `src.experiments.fixed_time_baseline` runs SUMO's unchanged signal program
  over consecutive seeds. It writes per-episode vehicle counts, completion
  percentage, mean and maximum total queue, cumulative queueing delay, mean
  queueing delay per generated vehicle, and hourly throughput.
- `run_baseline_experiments.py` runs the random and fixed-time controllers over
  the same default seeds. It records total queue-based reward and generated and
  completed vehicle counts, then reports means and sample standard deviations.
  Completed seed/controller pairs are saved incrementally so interrupted runs
  can resume.

No tracked output file currently provides a numerical baseline result, so this
README does not claim any measured performance advantage.

## Repository structure

```text
Traffic-RL/
├── simulation/
│   ├── config/                 # SUMO run configuration
│   ├── network/                # Junction sources and compiled network
│   └── routes/                 # Stochastic straight-through demand
├── src/
│   ├── agents/                 # Random policy, DQN, and replay buffer
│   ├── environment/            # Gymnasium wrapper around the simulation
│   ├── experiments/            # Fixed-time multi-seed experiment
│   ├── simulation/             # TraCI adapter, metrics, recording, runner
│   └── visualization/          # Recorded-episode Pygame replay
├── tests/                      # Unit and SUMO integration tests
├── run_baseline_experiments.py # Random-versus-fixed resumable comparison
├── run_random_agent.py         # One random-controller episode
├── test_env.py                 # Live SUMO/Gymnasium smoke test
├── train.py                    # Prototype DQN training loop
├── requirements.txt
└── README.md
```

The compiled `intersection.net.xml` is tracked alongside its editable SUMO XML
sources so the geometry and signal logic can be inspected and regenerated.

## Installation

Python 3.11 or later is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SUMO must be installed and its `sumo` executable must be available. The pinned
`eclipse-sumo` dependency supplies SUMO, `sumo-gui`, TraCI, and `sumolib` on
supported platforms. With a separate system installation, configure that
installation as required by SUMO (commonly by setting `SUMO_HOME`) and ensure
its `bin` directory is on `PATH`.

The requirements also install Gymnasium, NumPy, and PyTorch for the RL code,
plus pytest for the test suite.

## Running the project

Run the fixed-time simulation headlessly, optionally recording every step:

```bash
python -m src.simulation.run --steps 600 --seed 42
python -m src.simulation.run --steps 600 --seed 42 --record results/run-42.csv
```

Add `--gui` to either simulation command to use `sumo-gui`. Replay a recording
without reconnecting to SUMO:

```bash
python -m src.visualization.replay results/run-42.csv
```

Run the fixed-time multi-seed experiment or random-versus-fixed comparison:

```bash
python -m src.experiments.fixed_time_baseline --episodes 20
python run_baseline_experiments.py
```

Exercise the Gymnasium environment against a live SUMO process, or run one
random-policy episode:

```bash
python test_env.py
python run_random_agent.py
```

The current prototype training loop can be started with:

```bash
python train.py
```

It runs three 300-second episodes and prints episode reward, mean training loss,
and epsilon. It does not save the trained network.

## Testing

Run the existing suite with:

```bash
pytest
```

The tests cover SUMO configuration and straight-through routes, TraCI command
construction and simulation counters, traffic metrics and recordings,
reproducible baseline scenarios, experiment result handling, replay parsing and
playback, DQN output/training behavior, and replay-buffer storage and sampling.
Some tests launch headless SUMO, so the simulator must be available.

## Known limitations

- The junction has one incoming lane per direction, straight-through vehicles
  only, and no pedestrians, turning traffic, neighboring intersections, or
  emergency-vehicle behavior.
- Queueing delay is integrated halted-vehicle occupancy in vehicle-seconds. The
  per-step `mean_waiting_time` field instead uses SUMO's current incoming-lane
  waiting-time values; these are related but distinct metrics.
- The environment's seed is passed to Gymnasium, while reproducible SUMO demand
  is determined by the command used to launch or reload SUMO. The training
  script does not seed Python, NumPy, or PyTorch.
- The prototype has no model persistence, formal trained-agent evaluation, or
  reported comparison results yet.

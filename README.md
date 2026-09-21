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

Both panels use the same recorded second. Small cars are schematic markers for
recorded halted-vehicle queues, not actual SUMO positions or trajectories.
They briefly join the rear when a recorded queue grows; when it shrinks, front
markers move through the junction only if the recorded interval had a compatible
green signal. Otherwise they disappear without depicting a crossing. The motion
illustrates aggregate queue changes and does not reconstruct individual vehicle
movements or change the traffic simulation or evaluation results. During motion,
the direction labels show the current recorded counts. Up to six settled cars
fit per incoming approach; a `+N` marker shows any additional cars. Animation
lasts 0.4 simulation seconds (about 0.04 real seconds at 10×). It pauses with
the replay and resets when seeking, restarting, or skipping recorded frames.

Car colours are schematic, drawn from a fixed palette using the scenario,
seed, controller side, approach, and visual car creation order. They stay fixed
for each marker and repeat on the same replay.

The prominent stop-line signals and phase label show the actual recorded phase
governing the displayed second. During playback, “Queue s / inserted” means
accumulated halted-vehicle seconds divided by vehicles inserted up to that
second. The end card compares the selected seed only, with DQN-minus-fixed
differences; it does not show the 30-seed aggregate result. For changing demand,
the label and timeline markers show the scheduled arrival periods at seconds 0,
100, and 200; existing queues continue across those boundaries.

Controls: **Space** play/pause; **R** restart; **1**, **2**, **5**, **0** select
1×, 2×, 5×, 10× playback; **Left/Right** seek by five recorded seconds; click
the progress bar to seek; **Esc** exit. Playback pauses at the final second,
and R starts again without loading SUMO or the checkpoint. For a headless
rendering check, set `SDL_VIDEODRIVER=dummy`; use a normal desktop display for
the presentation.

## Research question

> Can a Deep Q-Network learn an adaptive traffic signal policy that reduces
> congestion and vehicle delay compared with fixed-time traffic control?

## Current project status

The repository now contains:

- A SUMO network for one signalised four-way junction, with stochastic flows
  from all approaches and reproducible SUMO seeds.
- A 68-second fixed-time signal program: 30 seconds north-south green, three
  seconds yellow, one second all-red, then the equivalent east-west phases.
- A reusable TraCI adapter that advances SUMO, reads incoming-lane queues,
  vehicle counts and waiting times, reads or changes the traffic-light phase,
  and counts departed and completed vehicles.
- Per-step and run-level metrics, optional CSV/JSON recording, a single-recording
  Pygame replay, and the paired presentation replay described above.
- A multi-seed fixed-time experiment and a resumable comparison of fixed-time
  and random controllers.
- A Gymnasium environment, random policy, replay buffer, PyTorch DQN, and
  training metrics and checkpoints from the 100-episode experiment.
- An isolated 700-episode training experiment with resumable checkpoints and
  a separate validation set; see [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md).
- Paired fixed-time/DQN evaluations, four frozen-demand scenarios, signal
  allocation diagnostics, minimum-green sensitivity analysis, and a completed
  independent final evaluation. The presentation replay uses that final data.

The final result is scenario-dependent: the frozen 15-second minimum-green DQN
reduced mean queue and the project's waiting measure in all four tested demand
profiles, but completed more vehicles than fixed time only in balanced and
north-south-heavy demand. See [Final independent evaluation](#final-independent-evaluation)
for the prespecified comparison and its limits.

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
The standard `train.py` run lasts 100 episodes of 300 seconds and writes an
episode-metrics CSV and a final checkpoint under `results/training/`. The
minimum-green checkpoint used below was trained with a 10-second guard; the
15-second presentation controller applies a different guard to those same
frozen network weights.

## Baseline experiments

Baselines establish performance levels against which the trained agent is
evaluated:

- `src.experiments.fixed_time_baseline` runs SUMO's unchanged signal program
  over consecutive seeds. It writes per-episode vehicle counts, completion
  percentage, mean and maximum total queue, cumulative queueing delay, mean
  queueing delay per generated vehicle, and hourly throughput.
- `run_baseline_experiments.py` runs the random and fixed-time controllers over
  the same default seeds. It records total queue-based reward and generated and
  completed vehicle counts, then reports means and sample standard deviations.
  Completed seed/controller pairs are saved incrementally so interrupted runs
  can resume.

Baseline outputs are saved in `results/baseline/`, `results/baseline_per_seed.csv`,
and `results/baseline_summary.csv`. These are separate from the matched-demand
final evaluation below.

## Final independent evaluation

The frozen checkpoint `results/training/dqn_100_episode_min_green.pt` was
evaluated against unchanged fixed time on 30 new matched seeds (4000–4029) for
each of four 300-second demand scenarios. The 15-second minimum-green setting
was selected using development seeds 3000–3029 before the final seeds were
evaluated. The same checkpoint weights were used for all settings; the final
evaluation did not retrain the DQN. The checked-in audit reports that all 600
episodes passed its validation.

The table gives mean DQN-minus-fixed differences. Brackets are two-sided 95%
paired intervals across the 30 seeds in each scenario. Waiting is accumulated
halted-vehicle seconds divided by inserted vehicles, not the mean journey delay
of completed vehicles.

| Demand scenario | Completed vehicles | Waiting (s/inserted) | Mean halted queue |
|---|---:|---:|---:|
| Balanced | +1.63 [+0.36, +2.91] | −2.76 [−3.26, −2.27] | −1.33 [−1.59, −1.08] |
| NS-heavy | +2.87 [+1.75, +3.99] | −3.03 [−3.51, −2.56] | −1.42 [−1.64, −1.20] |
| EW-heavy | −1.77 [−3.01, −0.52] | −1.78 [−2.27, −1.28] | −0.87 [−1.12, −0.62] |
| Changing | −1.37 [−2.27, −0.46] | −2.68 [−3.17, −2.19] | −1.29 [−1.54, −1.05] |

All four scenarios have 0.48 expected arrivals per second. Balanced demand
shares arrivals equally; NS-heavy and EW-heavy shift the directional shares;
changing demand uses balanced, NS-heavy, and EW-heavy in consecutive 100-second
windows. Results, supporting signal-allocation comparisons, protocol, record
hashes, and interpretation are in
[results/final_evaluation/report.md](results/final_evaluation/report.md) and
[results/final_evaluation/interpretation.md](results/final_evaluation/interpretation.md).
The 10-, 20-, and 30-second minimum-green comparisons are exploratory; see
[results/min_green_sensitivity/report.md](results/min_green_sensitivity/report.md).

The saved studies are organised as follows:

| Stage | Script | Saved output |
|---|---|---|
| Standard DQN training and diagnostics | `train.py`, `scripts/plot_training_metrics.py` | `results/training/` |
| Isolated extended training and checkpoint selection | `train_extended.py`, `validate_extended.py` | `results/extended_training/` |
| Earlier paired fixed-time/DQN comparison | `evaluate_paired.py` | `results/evaluation/paired_30_min_green/` |
| Four-scenario development study | `evaluate_generalisation.py` | `results/generalisation/` |
| Signal allocation and green-duration diagnostics | `evaluate_signal_allocation.py`, `evaluate_min_green_sensitivity.py` | `results/signal_allocation/`, `results/min_green_sensitivity/` |
| Independent final study | `evaluate_final.py` | `results/final_evaluation/` |

The development and sensitivity studies informed the prespecified 15-second
setting. The final study used separate seeds and preserved its own protocol and
per-second records.

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
│   ├── experiments/            # Baselines and paired demand generation
│   ├── simulation/             # TraCI adapter, metrics, recording, runner
│   └── visualization/          # Single and paired Pygame replays
├── tests/                      # Unit and SUMO integration tests
├── run_baseline_experiments.py # Random-versus-fixed resumable comparison
├── run_random_agent.py         # One random-controller episode
├── test_env.py                 # Live SUMO/Gymnasium smoke test
├── train.py                    # 100-episode DQN training
├── train_extended.py           # Isolated resumable 700-episode experiment
├── validate_extended.py       # Extended-checkpoint validation
├── evaluate_*.py               # Paired, scenario, diagnostic, and final studies
├── results/                    # Saved metrics, checkpoints, and reports
├── EXTENDED_TRAINING.md        # Extended experiment protocol
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
Matplotlib for plots, Pygame for replays, and pytest for the test suite.

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

Run the standard 100-episode training experiment with:

```bash
python train.py
```

It records reward, loss, epsilon, waiting, queue, and throughput to
`results/training/training_metrics.csv` and saves a checkpoint at
`results/training/dqn_100_episode.pt`. This command trains a new model; the
presentation replay and report viewing do not require it. The isolated
extended training and validation commands are documented in
[EXTENDED_TRAINING.md](EXTENDED_TRAINING.md).

To verify the saved final evaluation and rebuild its report from existing
records without running SUMO, use:

```bash
.venv/bin/python evaluate_final.py --report-only
```

The final evaluator's normal mode runs SUMO and writes evaluation artifacts;
use the checked-in reports and replay to inspect the completed experiment.

## Testing

Run the existing suite with:

```bash
pytest
```

The tests cover SUMO configuration and straight-through routes, TraCI command
construction and simulation counters, traffic metrics and recordings,
reproducible baseline scenarios, experiment result handling, replay parsing,
playback and schematic animations, DQN training, and saved-evaluation validation.
Some tests launch headless SUMO, so the simulator must be available. If the
local readline extension causes pytest to crash on startup, run:

```bash
.venv/bin/python -c 'import sys; sys.modules["readline"] = None; import pytest; raise SystemExit(pytest.main(["-q"]))'
```

## Known limitations

- The junction has one incoming lane per direction, straight-through vehicles
  only, and no pedestrians, turning traffic, neighboring intersections, or
  emergency-vehicle behavior.
- Queueing delay is integrated halted-vehicle occupancy in vehicle-seconds. The
  per-step `mean_waiting_time` field instead uses SUMO's current incoming-lane
  waiting-time values; these are related but distinct metrics.
- Reproducible SUMO demand is determined by the seed and route file used to
  launch or reload SUMO. The standard 100-episode `train.py` run reuses its
  seeded demand realisation; the isolated extended experiment uses a different
  SUMO seed for each training episode.
- The final evaluation covers one junction, four specified demand profiles,
  and 300-second episodes without a separate clearance period. Its results do
  not establish performance at other intersections or under every demand mix.
- The replay has per-second aggregate queue records, not vehicle IDs or paths.
  Its animated, coloured cars are presentation markers and cannot establish
  which actual SUMO vehicles crossed or completed a journey.

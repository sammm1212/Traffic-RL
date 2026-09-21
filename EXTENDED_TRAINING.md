# Controlled extended training

The extended experiment is separate from `train.py` and its existing 100-episode
files. It starts from random network weights; it does not load the 100-episode
checkpoint. The network, reward, optimizer, replay size, target update interval,
five-second decision interval, and ten-second minimum green are imported from
the existing implementation. Training episodes use SUMO seeds 20001–20700.
The route file contains four `probability="0.12"` flows, so SUMO samples
departures with its seed. Validation uses 2000–2009; the prior paired test uses
1000–1029. The older DQN evaluation uses 100–109.

Run the full experiment only when ready:

```sh
.venv/bin/python train_extended.py
```

This writes `results/extended_training/training_metrics.csv` and
`results/extended_training/checkpoints/episode_0025.pt` through
`episode_0700.pt`, at 25-episode intervals. Each checkpoint includes both
networks, optimizer, epsilon, replay transitions, training step, completed
episode, configuration, and Python/NumPy/Torch RNG states. The CSV contains
the traffic seed, reward, mean loss, epsilon, mean waiting time, mean queue,
and throughput for every completed episode.

To resume, give the most recent checkpoint in the same output directory:

```sh
.venv/bin/python train_extended.py --resume results/extended_training/checkpoints/episode_0125.pt
```

The trainer trims CSV rows beyond that checkpoint and reruns those episodes.
The output directory must be empty for a fresh run. A checkpoint from a run
with different settings is rejected.

After training, evaluate every intermediate checkpoint greedily:

```sh
.venv/bin/python validate_extended.py
```

The validation procedure uses the same minimum-green environment and 300-second
episodes. It writes per-seed results, checkpoint means, and the selected best
checkpoint to `results/extended_training/validation/`. Selection minimizes
mean waiting time, then mean queue length, then maximizes throughput, then
minimizes vehicles remaining. Training reward and prior test results are not
used for selection. Keep seeds 1000–1029 untouched until the final comparison.

Checkpoint files contain Python objects for replay and RNG state. Resume and
validation load them with PyTorch pickle support, so only use checkpoints
created locally by this project.

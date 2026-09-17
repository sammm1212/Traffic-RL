# Traffic RL Project

## Project purpose

This is a seven-day reinforcement learning project investigating whether
a reinforcement-learning traffic-light controller can outperform a fixed-time
controller at a single four-way junction.

The traffic simulation uses SUMO.

The reinforcement-learning implementation is the core assessed work and MUST
be implemented separately by the project author.

## Current project phase

We are currently implementing Day 1 only:

- SUMO simulation
- Python/SUMO integration using TraCI
- fixed-time baseline
- traffic generation
- metrics collection
- tests

DO NOT implement reinforcement learning.

Specifically do not implement:

- DQN
- Q-learning
- neural networks
- PyTorch models
- replay buffers
- epsilon-greedy policies
- target networks
- reward optimisation
- Gymnasium RL environments

We will implement those manually later.

## Simulation scope

Build a deliberately simple and understandable simulation:

- one four-way intersection
- approaches from north, south, east and west
- one incoming lane per direction is sufficient
- straight-through traffic only initially
- no pedestrians
- no turning traffic unless required by SUMO
- no multiple intersections
- no emergency vehicles

Traffic should be stochastic but reproducible with configurable random seeds.

## Traffic lights

There should be two principal traffic movements:

1. north-south green
2. east-west green

Include safe yellow/intermediate phases where required.

The fixed-time baseline should approximately use:

- north-south green: 30 seconds
- transition/yellow
- east-west green: 30 seconds
- transition/yellow
- repeat

## Python integration

Use TraCI to:

- launch SUMO
- step the simulation
- inspect lanes/vehicles
- read traffic-light state
- collect traffic metrics
- close SUMO cleanly

The code should support:

- `sumo-gui` for debugging/demo
- headless `sumo` for experiments

Do not tightly couple the Python controller to the graphical interface.

## Metrics

At minimum collect:

- queue length by approach
- total queue length
- average queue length
- vehicle waiting time
- average waiting time
- throughput / vehicles completing the junction

Prefer reusable functions/classes because the RL controller will later use
the same observations.

## Engineering requirements

- Python 3
- clear directory structure
- type hints where sensible
- short functions
- descriptive names
- docstrings for public classes/functions
- no unnecessary framework dependencies
- deterministic behaviour when a seed is supplied

Add tests for code that does not require manually inspecting SUMO GUI output.

## Important architectural constraint

Separate:

simulation/
    SUMO configuration/network/routes

src/
    Python simulation control and metrics

The future RL agent must be able to use the SUMO interface without rewriting
the underlying traffic simulation.

## Validation

Before finishing:

1. run tests
2. run a short headless SUMO simulation
3. verify vehicles enter and leave the network
4. verify the traffic light cycles
5. verify metrics are produced
6. report exactly what files were created or modified
7. explain any assumptions or known limitations

Do not make unrelated changes.
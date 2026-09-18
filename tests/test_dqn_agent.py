import torch

from src.agents.dqn_agent import DQN, DQNAgent


def test_dqn_output_shape():
    model = DQN()

    state = torch.tensor(
        [[2.0, 3.0, 7.0, 4.0, 0.0]],
        dtype=torch.float32,
    )

    output = model(state)

    assert output.shape == (1, 2)


def test_action_is_valid():
    agent = DQNAgent()

    state = [2.0, 3.0, 7.0, 4.0, 0.0]

    action = agent.select_action(state)

    assert action in [0, 1]


def test_epsilon_decays():
    agent = DQNAgent()

    initial_epsilon = agent.epsilon

    agent.decay_epsilon()

    assert agent.epsilon < initial_epsilon


def test_train_step_returns_loss():
    agent = DQNAgent()

    batch = [
        (
            [1.0, 2.0, 3.0, 4.0, 0.0],
            0,
            -10.0,
            [2.0, 2.0, 3.0, 4.0, 0.0],
            False,
        ),
        (
            [2.0, 1.0, 4.0, 3.0, 1.0],
            1,
            -8.0,
            [2.0, 1.0, 3.0, 2.0, 1.0],
            False,
        ),
        (
            [3.0, 3.0, 2.0, 2.0, 0.0],
            0,
            -6.0,
            [2.0, 2.0, 2.0, 2.0, 0.0],
            False,
        ),
        (
            [1.0, 1.0, 1.0, 1.0, 1.0],
            1,
            -4.0,
            [0.0, 0.0, 0.0, 0.0, 1.0],
            True,
        ),
    ]

    loss = agent.train_step(batch)

    assert isinstance(loss, float)
    assert loss >= 0


def test_target_network_updates():
    agent = DQNAgent()

    with torch.no_grad():
        for parameter in agent.policy_network.parameters():
            parameter.add_(1.0)

    differences_before = [
        torch.equal(policy_param, target_param)
        for policy_param, target_param in zip(
            agent.policy_network.parameters(),
            agent.target_network.parameters(),
        )
    ]

    assert not all(differences_before)

    agent.update_target_network()

    for policy_param, target_param in zip(
        agent.policy_network.parameters(),
        agent.target_network.parameters(),
    ):
        assert torch.equal(policy_param, target_param)
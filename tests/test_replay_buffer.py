from src.agents.replay_buffer import ReplayBuffer


def test_replay_buffer_push():
    buffer = ReplayBuffer(capacity=10)

    buffer.push(
        [1, 2, 3, 4, 0],
        1,
        -10,
        [2, 2, 2, 3, 1],
        False,
    )

    assert len(buffer) == 1


def test_replay_buffer_sample():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.push(
            [i, i, i, i, 0],
            0,
            -i,
            [i + 1, i, i, i, 0],
            False,
        )

    batch = buffer.sample(3)

    assert len(batch) == 3
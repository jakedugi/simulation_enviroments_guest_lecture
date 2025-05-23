import reverb
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.trajectories import trajectory


class ReplayBuffer:
    """Wrapper around Reverb uniform table"""
    def __init__(self, spec, max_len: int, sequence_length: int = 2, table_name: str = "uniform_table"):
        self._server = reverb.Server([reverb.Table(
            table_name,
            max_size=max_len,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=trajectory.from_episode(spec).spec_with_batch()
        )])
        self._rb = reverb_replay_buffer.ReverbReplayBuffer(
            spec, sequence_length, table_name, local_server=self._server
        )
        self._observer = reverb_utils.ReverbAddTrajectoryObserver(
            self._rb.py_client, table_name, sequence_length
        )

    @property
    def observer(self):
        return self._observer

    def dataset(self, batch_size: int, num_steps: int = 2):
        return self._rb.as_dataset(sample_batch_size=batch_size,
                                   num_steps=num_steps).prefetch(3)

    def close(self):
        self._observer.close()
        self._server.stop()
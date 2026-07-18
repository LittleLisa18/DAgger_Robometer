import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_streaming(self, obs: Dict, *, on_actions_ready=None) -> Dict:
        """Streaming inference — sends partial actions via *on_actions_ready* callback.

        Default implementation falls back to ``infer`` (no streaming).
        Subclasses that support streaming should override this.
        """
        return self.infer(obs)

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

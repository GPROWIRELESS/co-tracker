"""Causal target-anchored windows; all frame I/O is inherited from DVSSDataset."""

import numpy as np
from .dataset import DVSSDataset


class TemporalDVSSDataset(DVSSDataset):
    def __init__(self, items, context_len=16, frame_step=2, **kwargs):
        if context_len < 2:
            raise ValueError(
                "Temporal datasets need context_len>=2; use DVSSDataset for frames"
            )
        self.context_len = context_len
        super().__init__(
            items, segment_len=context_len, frame_step=frame_step, **kwargs
        )

    def _anchors(self, length):
        return np.arange(0, length, self.segment_stride, dtype=np.int64)

    def _frame_indices(self, anchor, length):
        return np.maximum(
            anchor - np.arange(self.context_len - 1, -1, -1) * self.frame_step, 0
        ).astype(np.int64)

    def _required_indices(self, anchor, length):
        # Context may contain missing gaze; only the target label enters loss.
        return np.array([anchor], dtype=np.int64)

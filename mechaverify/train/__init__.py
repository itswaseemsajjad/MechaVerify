"""Adaptation helpers (SFT + RLVR).

The GPU-bound training itself lives in the notebooks under ``notebooks/``;
this package holds the *testable*, CPU-only pieces:

    - data.py : turn a MechaVerify manifest into chat-format SFT / RL records
                and the GRPO reward functions (wrapping the deterministic verifier)
"""

from mechaverify.train.data import (  # noqa: F401
    make_reward_funcs,
    target_completion,
    to_chat,
    to_rl_records,
    to_sft_records,
)

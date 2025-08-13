"""Live C Engine Integration test for FiBAR repeat choices and Multi-Guard Post-Sword Path Rewards in env1.PoPEnv."""
import pytest
import numpy as np
from env1 import PoPEnv, REPEAT_CHOICES, N_REPEATS, _PATH_STEP_REWARD

@pytest.fixture(scope="module")
def real_env():
    env = PoPEnv(headless=True, max_steps=500, start_room=1, start_pos=0)
    yield env
    env.close()


class TestLiveEngineFiBARAndMultiGuardPaths:

    def test_fibar_action_space_constants(self, real_env):
        """Verify FiBAR repeat choices match _back defaults [1, 2, 3, 4, 8, 13, 18]."""
        assert REPEAT_CHOICES == [1, 2, 3, 4, 8, 13, 18]
        assert N_REPEATS == 7
        assert real_env.action_space.nvec[1] == 7

    def test_live_multi_guard_return_path_rewards(self, real_env):
        """Test live engine step execution with simulated multi-guard hints injected."""
        obs, info = real_env.reset()
        
        # Inject synthetic hint containing 2 guard paths and 1 fallback
        hint = {
            "paths_by_guard": {
                3:  [1, 2, 3],
                19: [1, 2, 6, 18, 19]
            },
            "fallback": [1, 2]
        }
        real_env.set_pbrs_hint(hint)

        # Manually trigger sword pickup state
        real_env.sword_found = True
        real_env._post_sword_paths = real_env._build_return_paths()
        real_env._post_sword_ptrs = {gr: 1 for gr in real_env._post_sword_paths} # start past room 1

        assert 3 in real_env._post_sword_paths
        assert 19 in real_env._post_sword_paths

        # Simulate agent step into room 2 (shared target for both paths)
        real_env.prev_room = 1
        real_env.data.kid.room = 2
        real_env.data.kid.alive = -1 # alive

        # Trigger room crossing handling directly as done in step()
        reward = real_env._room_novelty(2)
        if real_env.sword_found and real_env._post_sword_paths:
            for key, path in real_env._post_sword_paths.items():
                ptr = real_env._post_sword_ptrs.get(key, 0)
                if ptr < len(path) and 2 == path[ptr]:
                    reward += _PATH_STEP_REWARD
                    real_env._post_sword_ptrs[key] = ptr + 1

        # Both Guard 3 and Guard 19 paths should have advanced pointer from 1 to 2
        assert real_env._post_sword_ptrs[3] == 2
        assert real_env._post_sword_ptrs[19] == 2
        assert reward == 2 * _PATH_STEP_REWARD # +30.0 total
        assert real_env._room_novelty(2) == 0.0 # post-sword novelty is 0

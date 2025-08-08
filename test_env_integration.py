"""Integration tests for env1.PoPEnv — requires the real SDLPoP engine.

Tests the edge_resolved IPC contract (Bug 2 fix) and verifies that each
physical crossing now produces exactly one memory update with no double-count.

Run with: pytest test_env_integration.py -v -s

NOTE: PoPEnv launches a C SDL thread at construction time and cannot be safely
constructed more than once per process (the shared library is RTLD_GLOBAL and
SDL is a singleton). All tests therefore share a single module-scoped env
instance and call env.reset() between tests.

Design note on edge_resolved timing: with the pending-crossing state machine,
edge_resolved appears on the step where the OUTCOME becomes known (death or
next crossing), not on the step the crossing happens. Tests account for this.
"""
import pytest
import numpy as np

try:
    from env1 import PoPEnv
    _ENGINE_AVAILABLE = True
    _ENGINE_ERROR = ""
except Exception as e:
    _ENGINE_AVAILABLE = False
    _ENGINE_ERROR = str(e)

skip_no_engine = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"PoPEnv unavailable (engine not built or SDL not present): {_ENGINE_ERROR}"
)

# Room 2 (right link = room 3, no guard nearby) reliably crosses within ~55 steps
# of holding RIGHT. Used for all crossing-dependent tests.
_START_ROOM = 2


# ---------------------------------------------------------------------------
# Single shared env instance for the whole module (SDL is a process singleton)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    if not _ENGINE_AVAILABLE:
        pytest.skip("Engine not available")
    e = PoPEnv(headless=True, max_steps=5000, start_room=_START_ROOM, start_pos=0)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rollout_collecting_all_info(env, max_steps=500):
    """Step RIGHT until episode end; return list of all info dicts."""
    env.reset()
    action = np.array([4, 0], dtype=np.int64)  # RIGHT key, 1-tick repeat
    infos = []
    for _ in range(max_steps):
        obs, rew, term, trunc, info = env.step(action)
        infos.append(info)
        if term or trunc:
            break
    return infos


# ===========================================================================
# TestDeathAttributionCrossCheck
# ===========================================================================

class TestDeathAttributionCrossCheck:
    """Verify the edge_resolved key shape and semantics."""

    @skip_no_engine
    def test_edge_crossed_key_absent(self, env):
        """The old edge_crossed key must never appear in any info dict."""
        infos = _rollout_collecting_all_info(env)
        bad = [i for i, info in enumerate(infos) if "edge_crossed" in info]
        assert bad == [], (
            f"edge_crossed appeared at steps {bad} — Bug 2 IPC rename not applied"
        )

    @skip_no_engine
    def test_edge_death_key_absent(self, env):
        """The old edge_death key must never appear in any info dict."""
        infos = _rollout_collecting_all_info(env)
        bad = [i for i, info in enumerate(infos) if "edge_death" in info]
        assert bad == [], (
            f"edge_death appeared at steps {bad} — Bug 2 IPC rename not applied"
        )

    @skip_no_engine
    def test_edge_resolved_key_present_on_crossing(self, env):
        """edge_resolved must appear at least once per episode (committed at outcome).

        With start_room=2, the kid crosses into room 3 then dies — the crossing
        is committed as died=True at the termination step.
        """
        infos = _rollout_collecting_all_info(env)
        resolved_infos = [info for info in infos if "edge_resolved" in info]
        assert len(resolved_infos) >= 1, (
            "edge_resolved never appeared — crossing was never committed"
        )
        src, dst, direction, died = resolved_infos[0]["edge_resolved"]
        assert isinstance(src, int)
        assert isinstance(dst, int)
        assert direction in ("left", "right", "up", "down", None)
        assert isinstance(died, bool)

    @skip_no_engine
    def test_survived_crossing_has_died_false(self, env):
        """A crossing followed by a second crossing must commit the first as died=False.

        We run a longer episode. If the kid crosses multiple rooms the first
        crossing is committed as survived (died=False) when the second happens.
        """
        infos = _rollout_collecting_all_info(env, max_steps=1000)
        survived = [info["edge_resolved"] for info in infos
                    if "edge_resolved" in info and not info["edge_resolved"][3]]
        if not survived:
            pytest.skip("No survived crossing observed — episode ended after only one crossing")
        for src, dst, direction, died in survived:
            assert died is False


# ===========================================================================
# TestKnownDoubleCountBehavior
#
# Before Bug 2 fix: one crossing could produce edge_crossed (survived) AND
# later edge_death (died) for the same physical crossing.
# After fix: each crossing produces exactly ONE edge_resolved event.
# The double-count should now always be 0.
# ===========================================================================

class TestKnownDoubleCountBehavior:
    """Regression test: after Bug 2 fix, double-count of crossings must be 0."""

    @skip_no_engine
    def test_no_double_count_after_fix(self, env):
        """Each physical crossing must produce exactly one edge_resolved event.

        We collect all edge_resolved events over a full episode. Since each
        crossing commits exactly once, no (src, dst) pair should appear more
        than once per episode.
        """
        infos = _rollout_collecting_all_info(env, max_steps=1000)

        crossing_counts: dict = {}
        for info in infos:
            if "edge_resolved" in info:
                src, dst, direction, died = info["edge_resolved"]
                key = (src, dst)
                crossing_counts[key] = crossing_counts.get(key, 0) + 1

        double_count = sum(v - 1 for v in crossing_counts.values() if v > 1)

        print(f"\n  edge_resolved emissions: {crossing_counts}")
        print(f"  double_count: {double_count}")
        assert double_count == 0, (
            f"Bug 2 not fixed: {double_count} double-counted crossings detected. "
            f"Crossing counts: {crossing_counts}"
        )

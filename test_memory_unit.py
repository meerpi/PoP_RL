"""Dependency-light unit tests for agent1.py memory functions.

Extracts update_edge_memory / update_gate_memory / update_poi_memory out of
agent1.py via ast so the full PyTorch/gym stack is never imported.
Also tests _scan_gate_changes from env1 (Bug 4 fix) now that it is a
module-level importable function.
"""
import ast
import os
import time
import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Extract the three memory functions from agent1.py via AST (no imports needed)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT_PATH = os.path.join(_HERE, "agent1.py")

with open(_AGENT_PATH) as f:
    _src = f.read()

_tree = ast.parse(_src)
_target_fns = {"update_edge_memory", "update_gate_memory", "update_poi_memory"}
_fn_sources = {}
for node in ast.walk(_tree):
    if isinstance(node, ast.FunctionDef) and node.name in _target_fns:
        _fn_sources[node.name] = ast.get_source_segment(_src, node)

assert set(_fn_sources) == _target_fns, f"Missing functions: {_target_fns - set(_fn_sources)}"

_mod = types.ModuleType("_agent1_memory")
exec(compile("\n".join(_fn_sources.values()), "<agent1_memory>", "exec"), _mod.__dict__)

update_edge_memory = _mod.update_edge_memory
update_gate_memory = _mod.update_gate_memory
update_poi_memory  = _mod.update_poi_memory


# ---------------------------------------------------------------------------
# Import _scan_gate_changes from env1 directly (Bug 4: now module-level)
# ---------------------------------------------------------------------------
_ENV_PATH = os.path.join(_HERE, "env1.py")
with open(_ENV_PATH) as f:
    _env_src = f.read()

_env_tree = ast.parse(_env_src)
_scan_src = None
for node in ast.walk(_env_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_scan_gate_changes":
        _scan_src = ast.get_source_segment(_env_src, node)
        break

assert _scan_src is not None, "_scan_gate_changes not found at module level in env1.py"
_env_mod = types.ModuleType("_env1_scan")
_env_mod.__dict__["np"] = np
exec(compile(_scan_src, "<env1_scan>", "exec"), _env_mod.__dict__)
_scan_gate_changes = _env_mod._scan_gate_changes

TILE_GATE = 4  # from env1 constants


# ===========================================================================
# update_edge_memory tests
# ===========================================================================

class TestEdgeMemory:
    def _mem(self):
        return {"edges": {}, "gates": {}, "poi": {}}

    def test_first_call_creates_record(self):
        mem = self._mem()
        update_edge_memory(mem, 1, 2, "right", died=False)
        assert "1:2:right" in mem["edges"]

    def test_survived_increments_n(self):
        mem = self._mem()
        update_edge_memory(mem, 1, 2, "right", died=False)
        assert mem["edges"]["1:2:right"]["n"] == 1

    def test_ema_survived(self):
        mem = self._mem()
        update_edge_memory(mem, 1, 2, "right", died=False, alpha=0.1)
        assert mem["edges"]["1:2:right"]["death_ema"] == pytest.approx(0.0)

    def test_ema_died(self):
        mem = self._mem()
        update_edge_memory(mem, 1, 2, "right", died=True, alpha=0.1)
        assert mem["edges"]["1:2:right"]["death_ema"] == pytest.approx(0.1)

    def test_multiple_updates_ema(self):
        mem = self._mem()
        alpha = 0.05
        for _ in range(3):
            update_edge_memory(mem, 3, 4, "left", died=True, alpha=alpha)
        rec = mem["edges"]["3:4:left"]
        expected = 0.0
        for _ in range(3):
            expected = (1 - alpha) * expected + alpha * 1.0
        assert rec["death_ema"] == pytest.approx(expected)

    def test_separate_keys_per_direction(self):
        mem = self._mem()
        update_edge_memory(mem, 1, 2, "right", died=False)
        update_edge_memory(mem, 2, 1, "left",  died=True)
        assert "1:2:right" in mem["edges"]
        assert "2:1:left"  in mem["edges"]


# ===========================================================================
# update_gate_memory tests
# ===========================================================================

class TestGateMemory:
    def _mem(self):
        return {"edges": {}, "gates": {}, "poi": {}}

    def _sw(self, action="press", kind="opener"):
        return (5, 3, 1, kind, action)

    def _gc(self, is_open=True):
        return [(2, 4, 0, is_open)]

    def test_no_op_when_no_switch(self):
        mem = self._mem()
        update_gate_memory(mem, None, self._gc())
        assert mem["gates"] == {}

    def test_no_op_when_no_gate_changes(self):
        mem = self._mem()
        update_gate_memory(mem, self._sw(), [])
        assert mem["gates"] == {}

    def test_press_increments_press_count(self):
        mem = self._mem()
        update_gate_memory(mem, self._sw("press"), self._gc(is_open=True))
        c = mem["gates"]["5:3:1"]["candidates"]["2:4:0"]
        assert c["press_opened_count"] == 1
        assert c["press_closed_count"] == 0
        assert c["release_opened_count"] == 0
        assert c["release_closed_count"] == 0

    def test_press_closed_counted_separately(self):
        mem = self._mem()
        update_gate_memory(mem, self._sw("press"), self._gc(is_open=False))
        c = mem["gates"]["5:3:1"]["candidates"]["2:4:0"]
        assert c["press_closed_count"] == 1
        assert c["press_opened_count"] == 0

    def test_release_increments_release_count(self):
        mem = self._mem()
        update_gate_memory(mem, self._sw("release"), self._gc(is_open=True))
        c = mem["gates"]["5:3:1"]["candidates"]["2:4:0"]
        assert c["release_opened_count"] == 1
        assert c["release_closed_count"] == 0
        assert c["press_opened_count"] == 0
        assert c["press_closed_count"] == 0

    def test_one_shot_sets_true_when_threshold_met(self):
        mem = self._mem()
        threshold = 3
        for _ in range(threshold):
            update_gate_memory(mem, self._sw("press"), self._gc(is_open=True), threshold=threshold)
        assert mem["gates"]["5:3:1"]["one_shot"] is True

    def test_one_shot_false_below_threshold(self):
        mem = self._mem()
        threshold = 5
        for _ in range(threshold - 1):
            update_gate_memory(mem, self._sw("press"), self._gc(is_open=True), threshold=threshold)
        assert mem["gates"]["5:3:1"]["one_shot"] is False

    def test_KNOWN_BUG_one_shot_does_not_revert_on_contradicting_evidence(self):
        """Bug 1 regression: one_shot must revert to False after a release is observed.

        Sequence:
          1. Press 5x opening the gate -> one_shot becomes True
          2. Then observe a RELEASE event on the same gate
             This is direct proof the gate is NOT one-shot.
          3. one_shot must now be False.
        """
        mem = self._mem()
        threshold = 5
        for _ in range(threshold):
            update_gate_memory(mem, self._sw("press"), self._gc(is_open=True), threshold=threshold)
        assert mem["gates"]["5:3:1"]["one_shot"] is True

        # release event proves the gate responds to releases
        update_gate_memory(mem, self._sw("release"), self._gc(is_open=False), threshold=threshold)
        assert mem["gates"]["5:3:1"]["one_shot"] is False, (
            "Bug 1 not fixed: one_shot remained True even after contradicting release evidence"
        )


# ===========================================================================
# update_poi_memory tests
# ===========================================================================

class TestPoiMemory:
    def _mem(self):
        return {"edges": {}, "gates": {}, "poi": {}}

    def test_creates_poi_entry(self):
        mem = self._mem()
        update_poi_memory(mem, 3, 5, 1, "sword")
        assert "3:5:1:sword" in mem["poi"]

    def test_increments_n_seen(self):
        mem = self._mem()
        for _ in range(3):
            update_poi_memory(mem, 3, 5, 1, "sword")
        assert mem["poi"]["3:5:1:sword"]["n_seen"] == 3

    def test_separate_keys_per_kind(self):
        mem = self._mem()
        update_poi_memory(mem, 1, 0, 0, "sword")
        update_poi_memory(mem, 1, 0, 0, "potion_big")
        assert "1:0:0:sword" in mem["poi"]
        assert "1:0:0:potion_big" in mem["poi"]


# ===========================================================================
# _scan_gate_changes performance / correctness tests (Bug 4)
# ===========================================================================

class TestGateScanPerformance:
    def _make_arrays(self, n_gate_cells=10, n_flipped=3):
        fg = np.zeros(720, dtype=np.uint8)
        bg_old = np.zeros(720, dtype=np.uint8)
        bg_now = np.zeros(720, dtype=np.uint8)
        fg[:n_gate_cells] = TILE_GATE
        bg_old[:n_flipped] = 1   # < 2 -> closed
        bg_now[:n_flipped] = 2   # >= 2 -> open
        return fg, bg_old, bg_now

    def test_correctness_detects_flips(self):
        fg, bg_old, bg_now = self._make_arrays(n_gate_cells=5, n_flipped=3)
        results = _scan_gate_changes(fg, bg_old, bg_now, TILE_GATE)
        assert len(results) == 3
        for room, col, row, is_open in results:
            assert is_open is True

    def test_correctness_no_flip_when_unchanged(self):
        fg = np.zeros(720, dtype=np.uint8)
        fg[0] = TILE_GATE
        bg = np.zeros(720, dtype=np.uint8)
        results = _scan_gate_changes(fg, bg, bg.copy(), TILE_GATE)
        assert results == []

    def test_correctness_non_gate_tiles_ignored(self):
        fg = np.ones(720, dtype=np.uint8)  # TILE_FLOOR everywhere
        bg_old = np.zeros(720, dtype=np.uint8)
        bg_now = np.ones(720, dtype=np.uint8) * 3
        results = _scan_gate_changes(fg, bg_old, bg_now, TILE_GATE)
        assert results == []

    def test_room_col_row_encoding(self):
        """Cell at index 30 -> room 2, col 0, row 0."""
        fg = np.zeros(720, dtype=np.uint8)
        fg[30] = TILE_GATE
        bg_old = np.zeros(720, dtype=np.uint8)
        bg_now = np.zeros(720, dtype=np.uint8)
        bg_now[30] = 2  # open
        results = _scan_gate_changes(fg, bg_old, bg_now, TILE_GATE)
        assert len(results) == 1
        room, col, row, is_open = results[0]
        assert room == 2
        assert col == 0
        assert row == 0
        assert is_open is True

    def test_performance_vectorized_vs_reference(self):
        """Vectorized must be faster than the naive 720-cell Python loop."""
        fg = np.zeros(720, dtype=np.uint8)
        fg[::3] = TILE_GATE
        rng = np.random.default_rng(0)
        bg_old = rng.integers(0, 4, size=720, dtype=np.uint8)
        bg_now = rng.integers(0, 4, size=720, dtype=np.uint8)

        def reference_scan(fg_arr, bg_old, bg_now, tile_gate_const):
            changes = []
            for ri in range(24):
                off = ri * 30
                for ti in range(30):
                    t = fg_arr[off + ti] & 0x1F
                    if t == tile_gate_const:
                        was_open = bg_old[off + ti] >= 2
                        now_open = bg_now[off + ti] >= 2
                        if was_open != now_open:
                            changes.append((ri + 1, ti % 10, ti // 10, bool(now_open)))
            return changes

        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            _scan_gate_changes(fg, bg_old, bg_now, TILE_GATE)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N):
            reference_scan(fg, bg_old, bg_now, TILE_GATE)
        t_ref = time.perf_counter() - t0

        print(f"\n  vectorized: {t_vec*1000/N:.3f}ms/call  reference: {t_ref*1000/N:.3f}ms/call  speedup: {t_ref/t_vec:.1f}x")
        assert t_vec < t_ref, "Vectorized scan should be faster than reference loop"

    def test_results_match_reference(self):
        """Vectorized and naive loop must return identical sorted results."""
        fg = np.zeros(720, dtype=np.uint8)
        fg[::3] = TILE_GATE
        rng = np.random.default_rng(42)
        bg_old = rng.integers(0, 4, size=720, dtype=np.uint8)
        bg_now = rng.integers(0, 4, size=720, dtype=np.uint8)

        def reference_scan(fg_arr, bg_old, bg_now, tile_gate_const):
            changes = []
            for ri in range(24):
                off = ri * 30
                for ti in range(30):
                    t = fg_arr[off + ti] & 0x1F
                    if t == tile_gate_const:
                        was_open = bg_old[off + ti] >= 2
                        now_open = bg_now[off + ti] >= 2
                        if was_open != now_open:
                            changes.append((ri + 1, ti % 10, ti // 10, bool(now_open)))
            return changes

        vec = sorted(_scan_gate_changes(fg, bg_old, bg_now, TILE_GATE))
        ref = sorted(reference_scan(fg, bg_old, bg_now, TILE_GATE))
        assert vec == ref

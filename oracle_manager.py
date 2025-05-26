class OracleManager:
    """3-Phase Curriculum HRL Proxy (Level 1 graph navigation)."""
    PHASE_1_SUBGOALS = ["room1_sword", "room2_gate"]
    PHASE_2_SUBGOALS = ["room3_door_button", "room4_stairs"]
    PHASE_3_SUBGOALS = ["room5_guard", "room6_exit"]

"""
oracle_manager.py — HRL Oracle (Manager Proxy)

The worker never knows it's talking to a rule engine. It sees a 34-dim
command slot: (target_room_onehot, condition_onehot, scalars). When the
transformer manager replaces this oracle later, the interface is identical.

Observation layout (appended to env.PoPEnv obs):
  [base_obs          (537)]
  [target_room_onehot (24)]  — exactly one room = 1.0
  [condition_onehot    (5)]  — NAVIGATE/FIGHT/GET_SWORD/HEAL/LEVEL_EXIT
  [potion_loc          (2)]  — (row, col) when HEAL active, else 0
  [have_sword          (1)]
  [backtrack_on        (1)]  — 1 once Phase 3 unlocked
  [guard_alive         (1)]
  Total: 537 + 34 = 571

3-phase curriculum (class-level, shared across all envs):
  Phase 1: forward exploration only, NAVIGATE commands to unvisited neighbors
  Phase 2: all 5 conditions active, active routing to sword/guard/exit
  Phase 3: backtracking unlocked, return navigation enabled
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import PoPEnv, ALIVE, T_POTION, OBS_DIM as BASE_OBS_DIM

LEVEL1_GRAPH = {
    1:  [2],
    2:  [3, 6],
    3:  [2, 9],
    4:  [14, 19],
    5:  [1, 6, 21],
    6:  [2, 5, 8],
    7:  [8, 14, 17, 20],
    8:  [6, 7, 21],
    9:  [3],
    10: [15, 19],
    11: [5, 10, 12],
    12: [11, 23, 13],
    13: [12],
    14: [22],
    15: [16, 21],
    16: [15, 17],
    17: [16, 24, 18],
    18: [17],
    19: [4, 10],
    20: [4, 7, 12, 23],
    21: [5, 8, 17],
    22: [14, 16],
    23: [12, 17, 20],
    24: [17],
}

SWORD_ROOM  = 15
EXIT_ROOM   = 9
GUARD_ROOMS = {3}

COND_NAVIGATE   = 0
COND_FIGHT      = 1
COND_GET_SWORD  = 2
COND_HEAL       = 3
COND_LEVEL_EXIT = 4
N_COND          = 5

ORACLE_DIM     = 24 + N_COND + 2 + 1 + 1 + 1   # = 34
ORACLE_OBS_DIM = BASE_OBS_DIM + ORACLE_DIM       # = 571

# phase transition thresholds
PHASE2_ROOM7_VISITS = 100    # room 7 entered N times across all envs
PHASE3_FULL_RUNS    = 50     # sword → kill → exit completed N times

R_TARGET     =  10.0
R_ROOM_BASE  =   1.0   # per frontier connection: 1.0 × (1 + unvisited_neighbors)
R_RETURN_ROOM =  0.5
R_GET_SWORD  =  25.0
R_GUARD_HIT  =   3.0
R_KILL       =  40.0
R_DRINK_SM   =   8.0
R_DRINK_LG   =  18.0
R_LEVEL_EXIT = 100.0
R_HP_LOSS    =  -2.0


class OraclePoPEnv(gym.Wrapper):
    # class-level curriculum state — shared across all 30 envs
    _curriculum_phase   = 1
    _room7_visit_count  = 0
    _phase2_completions = 0

    def __init__(self, visual=False):
        super().__init__(PoPEnv(visual=visual))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ORACLE_OBS_DIM,), dtype=np.float32
        )
        self._oracle_buf   = np.zeros(ORACLE_DIM, dtype=np.float32)
        self._full_obs_buf = np.zeros(ORACLE_OBS_DIM, dtype=np.float32)

        self._visited_rooms          : set  = set()
        self._target_room            : int  = 1
        self._active_cond            : int  = COND_NAVIGATE
        self._prev_hp                : int  = 0
        self._prev_guard_hp          : int  = 0
        self._prev_room              : int  = 0
        self._sword_acquired_this_ep : bool = False
        self._guard_killed_this_ep   : bool = False
        self._rng = np.random.default_rng()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    @classmethod
    def _check_phase_transition(cls):
        if cls._curriculum_phase == 1 and cls._room7_visit_count >= PHASE2_ROOM7_VISITS:
            cls._curriculum_phase = 2
            print(f"[ORACLE] Phase 2 unlocked (room 7 visits: {cls._room7_visit_count})")
        if cls._curriculum_phase == 2 and cls._phase2_completions >= PHASE3_FULL_RUNS:
            cls._curriculum_phase = 3
            print(f"[ORACLE] Phase 3 unlocked (full runs: {cls._phase2_completions})")

    # --- neighbor selection (random same-layer) ---

    def _pick_navigate_target(self, current_room: int) -> int:
        phase = OraclePoPEnv._curriculum_phase
        env = self.env
        neighbors = list(LEVEL1_GRAPH.get(current_room, []))
        if not neighbors:
            return current_room

        pool = list(neighbors)

        # Phase 1 & 2: filter out visited rooms (forward only)
        if phase < 3:
            unvisited = [r for r in pool if r not in self._visited_rooms]
            if unvisited:
                pool = unvisited

        # guard safety: don't send unarmed worker into guard room
        if not env.have_sword:
            safe = [r for r in pool
                    if not (r in GUARD_ROOMS and env.g_alive == ALIVE and env.g_hpmax > 0)]
            if safe:
                pool = safe

        return int(self._rng.choice(pool))

    def _bfs_toward_exit(self, src: int) -> int:
        """BFS next hop toward EXIT_ROOM. Only used for LEVEL_EXIT condition."""
        env = self.env
        neighbors = LEVEL1_GRAPH.get(src, [])
        # filter guard rooms if unarmed (shouldn't happen for LEVEL_EXIT but safety)
        valid_first = [r for r in neighbors
                       if not (r in GUARD_ROOMS and env.g_alive == ALIVE and not env.have_sword)]
        if not valid_first:
            valid_first = list(neighbors)

        visited = {src}
        q = deque()
        for c in valid_first:
            if c not in visited:
                visited.add(c)
                q.append((c, c))
        while q:
            node, first_step = q.popleft()
            if node == EXIT_ROOM:
                return first_step
            for nb in LEVEL1_GRAPH.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, first_step))
        return valid_first[0] if valid_first else src

    # --- condition selection ---

    def _compute_condition(self, current_room: int) -> int:
        env = self.env
        phase = OraclePoPEnv._curriculum_phase

        guard_here = (env.g_room == current_room
                      and env.g_hpmax > 0
                      and env.g_alive == ALIVE)

        # P1: FIGHT — always fires when guard is here and we have sword
        if guard_here and env.have_sword:
            return COND_FIGHT

        # P2: GET_SWORD — in sword room without sword
        if current_room == SWORD_ROOM and not env.have_sword:
            return COND_GET_SWORD

        # P3: HEAL — low HP and potion in current room
        if env.hp < env.hp_max and self._potion_in_room(current_room):
            return COND_HEAL

        # P4: LEVEL_EXIT — guard dead, have sword
        if env.have_sword and env.g_alive != ALIVE:
            return COND_LEVEL_EXIT

        # In Phase 1, only NAVIGATE + incidental conditions above
        # In Phase 2+, oracle actively routes to sword/fight/exit
        # but those are handled by target_room selection, not condition override
        return COND_NAVIGATE

    def _get_target_room(self, condition: int, current_room: int) -> int:
        if condition == COND_FIGHT:
            return current_room
        if condition == COND_GET_SWORD:
            return SWORD_ROOM
        if condition == COND_HEAL:
            return current_room
        if condition == COND_LEVEL_EXIT:
            return self._bfs_toward_exit(current_room)
        # NAVIGATE — random same-layer
        return self._pick_navigate_target(current_room)

    # --- helpers ---

    def _potion_in_room(self, room: int) -> bool:
        if not hasattr(self.env, 'fg') or self.env.fg is None:
            return False
        off = (room - 1) * 30
        for i in range(30):
            if int(self.env.fg[off + i]) & 0x1F == T_POTION:
                return True
        return False

    def _potion_loc_in_room(self, room: int):
        if not hasattr(self.env, 'fg') or self.env.fg is None:
            return 0.0, 0.0
        off = (room - 1) * 30
        for i in range(30):
            if int(self.env.fg[off + i]) & 0x1F == T_POTION:
                return (i // 10) / 2.0, (i % 10) / 9.0
        return 0.0, 0.0

    # --- observation ---

    def _build_oracle_vec(self, condition: int, target_room: int) -> None:
        buf = self._oracle_buf; buf[:] = 0.0
        env = self.env

        # target room one-hot (24 bits)
        if 1 <= target_room <= 24:
            buf[target_room - 1] = 1.0

        # condition one-hot (5 bits, offset 24)
        buf[24 + condition] = 1.0

        # potion location (2 floats, offset 29)
        if condition == COND_HEAL:
            pr, pc = self._potion_loc_in_room(env.k_room)
            buf[29] = pr; buf[30] = pc

        # scalars (offset 31-33)
        buf[31] = 1.0 if env.have_sword else 0.0
        buf[32] = 1.0 if OraclePoPEnv._curriculum_phase >= 3 else 0.0
        buf[33] = 1.0 if (env.g_alive == ALIVE and env.g_hpmax > 0) else 0.0

    def _make_obs(self, base_obs: np.ndarray) -> np.ndarray:
        # live reference — caller must copy if storing (FrameStack does this)
        np.copyto(self._full_obs_buf[:BASE_OBS_DIM], base_obs)
        np.copyto(self._full_obs_buf[BASE_OBS_DIM:], self._oracle_buf)
        return self._full_obs_buf

    # --- reward ---

    def _oracle_reward(self, prev_hp: int, prev_guard_hp: int,
                       prev_room: int, condition: int, prev_target: int, is_new_room: bool, alive: bool) -> float:
        """Shaped bonus on top of base env reward (which already has -0.01 time + -5.0 death)."""
        env = self.env
        r = 0.0
        if not alive:
            return r  # base env already gave -5.0

        hp_delta = env.hp - prev_hp
        if hp_delta < 0 and condition != COND_FIGHT:
            r += hp_delta * abs(R_HP_LOSS)

        # target reached — agent arrived at the oracle's commanded room
        if env.k_room != prev_room and env.k_room == prev_target:
            r += R_TARGET

        if env.k_room != prev_room and 1 <= env.k_room <= 24:
            if is_new_room:
                # frontier reward: 1.0 × (1 + number of unvisited neighbors)
                neighbors = LEVEL1_GRAPH.get(env.k_room, [])
                frontier = sum(1 for nb in neighbors if nb not in self._visited_rooms)
                r += R_ROOM_BASE * (1 + frontier)
            elif OraclePoPEnv._curriculum_phase >= 3:
                r += R_RETURN_ROOM

        if env.have_sword and not self._sword_acquired_this_ep:
            r += R_GET_SWORD
            self._sword_acquired_this_ep = True

        if condition == COND_FIGHT:
            guard_delta = prev_guard_hp - env.g_hp
            if guard_delta > 0:
                r += guard_delta * R_GUARD_HIT
            if prev_guard_hp > 0 and env.g_alive != ALIVE:
                r += R_KILL
                self._guard_killed_this_ep = True

        if hp_delta > 0:
            r += R_DRINK_LG if hp_delta >= 2 else R_DRINK_SM

        if env.level > 1:
            r += R_LEVEL_EXIT
            # track full run completion for phase 2→3 transition
            if self._sword_acquired_this_ep and self._guard_killed_this_ep:
                OraclePoPEnv._phase2_completions += 1

        return r

    # --- gymnasium interface ---

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        base_obs, info = self.env.reset(seed=seed, options=options)

        self._visited_rooms          = set()
        self._sword_acquired_this_ep = bool(self.env.have_sword)
        self._guard_killed_this_ep   = False
        self._prev_hp                = self.env.hp
        self._prev_guard_hp          = self.env.g_hp
        self._prev_room              = self.env.k_room

        current_room = self.env.k_room
        self._visited_rooms.add(current_room)
        condition         = self._compute_condition(current_room)
        self._target_room = self._get_target_room(condition, current_room)
        self._active_cond = condition
        self._build_oracle_vec(condition, self._target_room)

        info.update({
            "oracle_condition": condition,
            "oracle_target":    self._target_room,
            "oracle_phase":     OraclePoPEnv._curriculum_phase,
        })
        return self._make_obs(base_obs), info

    def step(self, action):
        prev_hp       = self.env.hp
        prev_guard_hp = self.env.g_hp
        prev_room     = self.env.k_room
        prev_cond     = self._active_cond
        prev_target   = self._target_room

        base_obs, _env_rew, terminated, truncated, info = self.env.step(action)

        alive = not terminated
        current_room = self.env.k_room

        # update episode state + class-level counters
        if current_room != prev_room and 1 <= current_room <= 24:
            is_new_room = current_room not in self._visited_rooms
            self._visited_rooms.add(current_room)
            if current_room == 7:
                OraclePoPEnv._room7_visit_count += 1
        else:
            is_new_room = False

        # run oracle pipeline
        condition = self._compute_condition(current_room)

        # re-pick target when target reached, or condition changed
        if current_room == prev_target or condition != prev_cond:
            self._target_room = self._get_target_room(condition, current_room)

        self._active_cond = condition
        self._build_oracle_vec(condition, self._target_room)

        reward = _env_rew + self._oracle_reward(prev_hp, prev_guard_hp, prev_room, prev_cond, prev_target, is_new_room, alive)

        self._prev_hp       = self.env.hp
        self._prev_guard_hp = self.env.g_hp
        self._prev_room     = current_room

        # check phase transitions after state update
        OraclePoPEnv._check_phase_transition()

        info.update({
            "oracle_condition": condition,
            "oracle_target":    self._target_room,
            "oracle_phase":     OraclePoPEnv._curriculum_phase,
        })
        return self._make_obs(base_obs), reward, terminated, truncated, info


def make_oracle_env(seed: int = 0, rank: int = 0, visual: bool = False,
                    max_steps: int = 2048):
    def _thunk():
        env = OraclePoPEnv(visual=visual)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + rank)
        return env
    return _thunk


if __name__ == "__main__":
    env = OraclePoPEnv(visual=False)
    obs, info = env.reset(seed=42)
    print(f"OBS shape : {obs.shape}  (expected ({ORACLE_OBS_DIM},))")
    print(f"Room : {info['room']}  Target: {info['oracle_target']}  Phase: {info['oracle_phase']}")

    total_rew = 0.0
    for step_i in range(500):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        total_rew += rew
        if term or trunc:
            print(f"Episode ended step {step_i+1} | rew={total_rew:.2f}")
            break
    env.close()
    print("Smoke test passed.")

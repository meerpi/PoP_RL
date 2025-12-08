import os
import math
import heapq
import numpy as np
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces
from PoP_env.wrappers.pop_imports import SDLPoP_Interface
from PoP_env.wrappers.obs_builder import ObsBuilder, FLAT_OBS_SPACE

_SDLPOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SDLPoP"))
_SO_PATH    = os.path.join(_SDLPOP_DIR, "src", "libSDLPoP.so")

# PBRS constants
PBRS_SCALE = 5.0
W_RISKY    = 3.0
W_FATAL    = 10.0


class PoPAction(IntEnum):
    NONE              = 0
    FORWARD           = 1
    BACKWARD          = 2
    UP                = 3
    DOWN              = 4
    SHIFT             = 5
    FORWARD_UP        = 6   # jump right / running jump
    FORWARD_DOWN      = 7   # hop right
    BACKWARD_UP       = 8   # jump left
    BACKWARD_DOWN     = 9   # hop left
    FORWARD_SHIFT     = 10  # careful step right
    BACKWARD_SHIFT    = 13  # careful step left
    UP_SHIFT          = 16  # grab ledge while falling
    DOWN_SHIFT        = 17  # draw sword

# C engine action integers for the 14 valid inputs (preserves rl_glue.c case numbers)
VALID_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17]


class PoPEnv(gym.Env):
    """Prince of Persia gymnasium environment backed by the SDLPoP engine.
    """

    metadata = {"render_modes": []}

    def __init__(self, *args, **kwargs):
        self.room_table = np.zeros((24, 13), dtype=np.float32)
        # Initialize 24x13 fog-of-war room matrix(self, level=1, gamma=0.997):
        self.original_cwd = os.getcwd()
        self.level = level
        self.gamma = gamma

        # The C engine resolves data/ relative to cwd, so it must run from here.
        os.chdir(_SDLPOP_DIR)
        self.engine = SDLPoP_Interface(_SO_PATH)
        self.obs = ObsBuilder(self.engine)

        self.action_space = gym.spaces.MultiDiscrete([14, 7]) spaces.Discrete(len(VALID_ACTIONS))  # 14 valid inputs
        self.observation_space = FLAT_OBS_SPACE

        self.started = False
        self.subgoal = -1  # set externally by curriculum / wrapper

        # Persistent across episodes
        self.visit_counts = {}   # room -> times reached (global, drives novelty sqrt decay)
        self.visited_rooms = set()  # rooms seen globally (for logging)

    def _build_weighted_adj(self):
        """Build weighted adjacency from obs_builder edge data.
        Only includes traversable edges (edge_trav==1).
        Returns dict {src_0idx: [(dst_0idx, weight), ...]}.
        """
        adj = {}
        n = self.obs.n_edges
        for i in range(n):
            if not self.obs.edge_trav[i]:
                continue
            src = int(self.obs.edge_src[i])
            dst = int(self.obs.edge_dst[i])
            if self.obs.edge_fatal[i]:
                w = W_FATAL
            elif self.obs.edge_risky[i]:
                w = W_RISKY
            else:
                w = 1.0
            adj.setdefault(src, []).append((dst, w))
        return adj

    def _compute_potential(self, room_0idx):
        """Φ(s) = PBRS_SCALE / (min_weighted_dist_to_frontier + 1).
        Frontier = any room in [0,23] not in _ep_frontier_visited.
        Returns 0.0 if room invalid or no frontier reachable.
        """
        if not (0 <= room_0idx <= 23):
            return 0.0

        # Dijkstra from room_0idx
        dist = {room_0idx: 0.0}
        heap = [(0.0, room_0idx)]
        best = float('inf')

        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, float('inf')):
                continue
            if d >= best:
                break
            if node not in self._ep_frontier_visited:
                best = d
                continue
            for nb, w in self._wadj.get(node, []):
                nd = d + w
                if nd < dist.get(nb, float('inf')) and nd < best:
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, nb))

        if best == float('inf'):
            return 0.0
        return PBRS_SCALE / (best + 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.started:
            self.engine._start_level.value = self.level
            self.engine.lib.pop_main()
            self.started = True
        else:
            self.engine.lib.init_game(self.level)

        self.obs.build_map_graph()

        # Per-episode state
        self._current_room = int(self.engine._kid.room)
        self._novelty_seen = set()

        # PBRS: seed frontier with start room so Φ(s₀) doesn't treat it as distance-0 target
        self._ep_frontier_visited = {self._current_room - 1}  # 0-indexed
        self._wadj = self._build_weighted_adj()
        self._prev_phi = self._compute_potential(self._current_room - 1)

        # HP tracking
        self._prev_hp = int(self.engine._hitp_curr.value)

        # Milestone trackers
        self._had_sword = bool(self.engine._have_sword.value)
        self._prev_guard_hp = int(self.engine._guardhp_curr.value)
        self._prev_level = int(self.engine._current_level.value)

        # Curiosity tracking: seed with initial tile
        col = int(self.engine._kid.curr_col)
        row = int(self.engine._kid.curr_row)
        self._ep_curiosity_visited = {(self._current_room, col, row, self._had_sword)}

        return self.build_obs(), {}

    def step(self, action):
        self.engine._rl_action.value = VALID_ACTIONS[int(action)]

        reward = 0.0
        terminated = False
        for _ in range(2):
            # Check for time expiration in C engine to avoid calling expired() and segfaulting
            rem_min = int(self.engine._rem_min.value)
            rem_tick = int(self.engine._rem_tick.value)
            if rem_min == 0 or (rem_min == 1 and rem_tick <= 1):
                terminated = True
                break

            self.engine.lib.play_level_2()
            dead = self.engine._kid.alive >= 0

            reward -= 0.02  # time penalty per frame

            # ── Room tracking + novelty (fires even on death frame for partial credit) ──
            curr_room = int(self.engine._kid.room)
            if 1 <= curr_room <= 24:
                self.visited_rooms.add(curr_room)
                self._ep_frontier_visited.add(curr_room - 1)
                
                # Curiosity: tile-level visitation density (+0.1 per unique tile per episode)
                col = int(self.engine._kid.curr_col)
                row = int(self.engine._kid.curr_row)
                has_sword_cur = bool(self.engine._have_sword.value)
                cur_key = (curr_room, col, row, has_sword_cur)
                if cur_key not in self._ep_curiosity_visited:
                    self._ep_curiosity_visited.add(cur_key)
                    reward += 0.1

            if 1 <= curr_room <= 24 and curr_room != self._current_room:
                self._current_room = curr_room
                if curr_room not in self._novelty_seen:
                    self._novelty_seen.add(curr_room)
                    n = self.visit_counts.get(curr_room, 0) + 1
                    self.visit_counts[curr_room] = n
                    reward += 8.0 / (n ** 0.5)

            # ── Milestones (fire even on death frame) ──

            # Sword pickup (+15, once per episode)
            has_sword = bool(self.engine._have_sword.value)
            if has_sword and not self._had_sword:
                self._had_sword = True
                reward += 15.0

            # Guard kill (+10)
            guard_hp = int(self.engine._guardhp_curr.value)
            guard_dead = self.engine._guard.alive >= 0
            if self._prev_guard_hp > 0 and guard_hp == 0 and guard_dead:
                reward += 10.0
            self._prev_guard_hp = guard_hp

            # Level win (+50, terminates episode)
            curr_level = int(self.engine._current_level.value)
            if curr_level > self._prev_level:
                self._prev_level = curr_level
                reward += 50.0
                terminated = True
                break

            # ── Death (after room/milestones so discovery credit isn't lost) ──
            if dead:
                reward -= 10.0
                terminated = True
                break

            # HP delta (only on alive, non-win frames)
            curr_hp = int(self.engine._hitp_curr.value)
            reward += 0.5 * (curr_hp - self._prev_hp)
            self._prev_hp = curr_hp

        # ── Observation + PBRS (after frame-skip) ──
        # build_obs() calls map_graph() which delta-patches edge_trav/fatal/risky.
        # Rebuild _wadj from patched edges so Φ sees current traversability.
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            phi_next = 0.0  # MANDATORY for PBRS invariance
        else:
            obs = self.build_obs()
            self._wadj = self._build_weighted_adj()
            phi_next = self._compute_potential(self._current_room - 1)

        reward += self.gamma * phi_next - self._prev_phi
        self._prev_phi = phi_next

        info = {"rooms_visited": len(self.visited_rooms)}
        info["reward_int"] = getattr(self, "_last_reward_int", 0.0)
        return obs, reward, terminated, False, info

    def build_obs(self):
        return self.obs.build_flat(subgoal_room=self.subgoal)

    def close(self):
        if hasattr(self, "original_cwd"):
            os.chdir(self.original_cwd)

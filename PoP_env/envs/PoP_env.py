"""Prince of Persia Gymnasium environment.

Includes:
  - SMDP: exposes frames_elapsed (τ) in info for γ^τ discount correction
  - Room table: experience-based (24, 13) fog-of-war room observation
  - Room novelty: 10/√N (lifetime) + 5.0 (episodic first-visit) intrinsic reward
  - Curiosity state: +1 per unique (room,col,row,hp_loss,sword) tuple
  - Dual reward: extrinsic in reward, intrinsic in info['reward_int']
"""
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from wrappers.build_obs import ObsBuilder
from wrappers.discrete_actions import NUM_ACTIONS

REPEAT_CHOICES = [1, 2, 3, 4, 8, 13, 18]
N_REPEATS = len(REPEAT_CHOICES)
_PATH_STEP_REWARD = 15.0

NUM_CHANNELS = 12
ROOM_TABLE_FEATURES = 13
NUM_ROOMS = 24


class PoPEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, headless=True, visual_mode=False, max_steps=50000,
                 start_room=None, start_pos=0):
        super().__init__()
        self.headless = headless
        self.visual_mode = visual_mode
        self.max_steps = max_steps
        self.start_room = start_room
        self.start_pos = start_pos
        self.frame_count = 0
        self.step_count = 0
        self.last_tau = 0

        # C Engine Interface & Observation Builder
        self.obs_builder = ObsBuilder(headless=headless)

        # Gymnasium spaces
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, N_REPEATS])
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, (NUM_CHANNELS, 5, 12), dtype=np.uint8),
            "state": spaces.Box(-1.0, 1.0, (29,), dtype=np.float32),
            "room": spaces.Box(0, 24, (1,), dtype=np.int32),
            "action_history": spaces.Box(0, NUM_ACTIONS - 1, (5,), dtype=np.int32),
            "repeat_history": spaces.Box(0, N_REPEATS - 1, (5,), dtype=np.int32),
            "room_table": spaces.Box(0.0, 1.0, (NUM_ROOMS, ROOM_TABLE_FEATURES), dtype=np.float32),
            "have_sword": spaces.Box(0, 1, (1,), dtype=np.float32),
        })

        self.action_history = np.zeros(5, dtype=np.int32)
        self.repeat_history = np.zeros(5, dtype=np.int32)
        self.visited_rooms = set()
        self.sword_found = False
        self.prev_hp = None
        self.prev_guard_hp = None
        self.sword_drawn = False
        self.sword_draw_rewarded = False
        self.prev_level = 1
        self._last_reward_int = 0.0
        self._post_sword_baseline_set = False

        # Exploration state — room_visits_pre/post persist across episodes
        self.room_visits_pre = {}   # {room: count} before sword
        self.room_visits_post = {}  # {room: count} after sword
        self.visited_states = set()
        self.experienced_edges = {}         # {room_id: set(neighbour_ids)} for room_table connectivity
        self.room_table = np.zeros((NUM_ROOMS, ROOM_TABLE_FEATURES), dtype=np.float32)
        self.prev_room = -1

        # Memory-based post-sword return path state
        self.dead_guard_rooms = set()
        self._post_sword_paths = {}  # {guard_room: [sword_room, ..., guard_room]}
        self._post_sword_ptrs = {}   # {guard_room: next_idx}
        self._pbrs_hint = {}
        self.path_to_sword = []

    def set_pbrs_hint(self, hint):
        self._pbrs_hint = hint if isinstance(hint, dict) else {}

    def _build_return_paths(self):
        paths_by_guard = self._pbrs_hint.get("paths_by_guard", {})
        active = {gr: list(p) for gr, p in paths_by_guard.items()
                  if gr not in self.dead_guard_rooms and p}
        if active:
            return active
        fb = self._pbrs_hint.get("fallback", [])
        if not fb and len(self.path_to_sword) >= 2:
            fb = list(reversed(self.path_to_sword))
        return {"fallback": list(fb)} if len(fb) >= 2 else {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.obs_builder.initialized:
            self.obs_builder.init_engine()
            self.obs_builder.wait_until_alive()
        else:
            self.obs_builder.release_held_action()
            self.obs_builder.request_restart(level=1)
            self.obs_builder.wait_until_alive()

        # Wait until kid is standing (action 0, frame 15)
        for _ in range(60):
            self.obs_builder.sync_wait(1)
            self.obs_builder.refresh()
            if self.obs_builder.kid.action == 0 and self.obs_builder.kid.frame == 15:
                break

        if self.start_room is not None:
            self.obs_builder.set_start_room(self.start_room, self.start_pos, 0)
            self.obs_builder.sync_wait(3)

        self.obs_builder.refresh()
        self.frame_count = 0
        self.step_count = 0
        self.last_tau = 0
        self._last_reward_int = 0.0
        self.prev_hp = self.obs_builder.hitp_curr
        self.sword_found = self.obs_builder.have_sword
        self.prev_guard_hp = None
        self.sword_drawn = False
        self.sword_draw_rewarded = False
        self.prev_level = int(self.obs_builder.data.current_level)
        start_room = int(self.obs_builder.kid.room)
        self.visited_rooms = {start_room} if start_room >= 1 else set()
        self.visited_states = set()
        self.action_history = np.zeros(5, dtype=np.int32)
        self.repeat_history = np.zeros(5, dtype=np.int32)

        # Reset episode-local state (room_visits_pre/post and experienced_edges persist)
        self.room_table = np.zeros((NUM_ROOMS, ROOM_TABLE_FEATURES), dtype=np.float32)
        self.prev_room = start_room
        self.dead_guard_rooms = set()
        self._post_sword_paths = {}
        self._post_sword_ptrs = {}
        self.path_to_sword = [start_room] if start_room >= 1 else []

        # Discover starting room
        if start_room >= 1:
            self._discover_room(start_room)
            self._update_room_table_dynamic(start_room)

        obs = self._build_obs()
        return obs, self._get_info()

    def step(self, action):
        # FiGAR: hold action for REPEAT_CHOICES[k_idx] engine ticks
        action_id = int(action[0])
        k = REPEAT_CHOICES[int(action[1])]

        self.obs_builder.press_action(action_id)

        frames_elapsed = 0
        for _ in range(k):
            self.obs_builder.sync_wait(1)
            frames_elapsed += 1
            self.frame_count += 1
            self.obs_builder.refresh()
            if self.obs_builder.is_dead:
                break

        self.step_count += 1
        self.action_history = np.roll(self.action_history, -1)
        self.action_history[-1] = action_id
        self.repeat_history = np.roll(self.repeat_history, -1)
        self.repeat_history[-1] = int(action[1])
        self.last_tau = frames_elapsed

        reward = 0.0
        reward_int = 0.0
        terminated = False
        truncated = False

        kid = self.obs_builder.kid
        room = int(kid.room)
        hp = self.obs_builder.hitp_curr
        level = int(self.obs_builder.data.current_level)
        alive = not self.obs_builder.is_dead

        # Death penalty & auto-reset handling
        if not alive:
            terminated = True
            reward -= 5.0 if self.sword_found else 10.0
            self.obs_builder.release_held_action()

        # Damage penalty
        hp_loss = 0
        if self.prev_hp is not None and hp < self.prev_hp:
            hp_loss = 1
            reward -= 0.5 * (self.prev_hp - hp)
        self.prev_hp = hp

        # Curiosity: unique (room, col, row, hp_loss, have_sword) tuples → intrinsic
        curiosity_state = (room, int(kid.curr_col), int(kid.curr_row),
                           hp_loss, int(self.obs_builder.have_sword > 0))
        if curiosity_state not in self.visited_states:
            reward_int += 1.0
            self.visited_states.add(curiosity_state)

        # Sword pickup → extrinsic
        if self.obs_builder.have_sword and not self.sword_found:
            reward += 100.0
            self.sword_found = True
            if not self.path_to_sword or self.path_to_sword[-1] != room:
                self.path_to_sword.append(room)
            self._post_sword_paths = self._build_return_paths()
            self._post_sword_ptrs = {gr: (1 if len(p) > 1 else 0)
                                     for gr, p in self._post_sword_paths.items()}
            # Snapshot baseline once per training run (not per episode) so post-sword
            # novelty decays correctly over the lifetime of training.
            if not self._post_sword_baseline_set:
                self.room_visits_post = {
                    r: (0 if r in self.room_visits_pre else 1)
                    for r in range(1, 25)
                }
                self._post_sword_baseline_set = True

        # Guard combat rewards → extrinsic
        guard_hp = int(self.obs_builder.data.guardhp_curr)
        guard_in_room = (int(self.obs_builder.data.guard.room) == room and
                         int(self.obs_builder.data.guardhp_max) > 0)
        if guard_in_room:
            kid_sword_drawn = self.obs_builder.kid.sword == 2
            if kid_sword_drawn and not self.sword_draw_rewarded:
                reward += 15.0
                self.sword_draw_rewarded = True
            self.sword_drawn = kid_sword_drawn

            if self.prev_guard_hp is not None and self.prev_guard_hp > 0 and guard_hp < self.prev_guard_hp:
                reward += 10.0 * (self.prev_guard_hp - guard_hp)
            if self.prev_guard_hp is not None and self.prev_guard_hp > 0 and guard_hp == 0:
                reward += 300.0
                dead_room = int(self.obs_builder.data.guard.room)
                if dead_room > 0:
                    self.dead_guard_rooms.add(dead_room)
                if self.sword_found:
                    self._post_sword_paths = self._build_return_paths()
                    self._post_sword_ptrs = {
                        gr: (p.index(room) + 1 if room in p and p.index(room) + 1 < len(p) else 0)
                        for gr, p in self._post_sword_paths.items()
                    }
            self.prev_guard_hp = guard_hp
        else:
            self.prev_guard_hp = None
            self.sword_drawn = False

        # Level complete → extrinsic & terminate episode
        if level > self.prev_level:
            reward += 500.0
            self.prev_level = level
            self._post_sword_paths = {}
            self._post_sword_ptrs = {}
            terminated = True
            self.obs_builder.release_held_action()

        # Room transition rewards → intrinsic & memory path rewards
        if room != self.prev_room and alive:
            if not self.sword_found:
                if room in self.path_to_sword:
                    del self.path_to_sword[self.path_to_sword.index(room) + 1:]
                elif not self.path_to_sword or self.path_to_sword[-1] != room:
                    self.path_to_sword.append(room)

            # Update experienced edges (for room_table connectivity)
            if self.prev_room > 0 and room > 0:
                self.experienced_edges.setdefault(self.prev_room, set()).add(room)
                self.experienced_edges.setdefault(room, set()).add(self.prev_room)
                self._update_connectivity(self.prev_room, room)

            # Room novelty reward → intrinsic
            reward_int += self._room_novelty(room)

            # Post-sword: reward for following ANY active memorized return path
            # Progress-escalated step reward: grows as agent gets closer to the guard room
            if self.sword_found and self._post_sword_paths:
                for key, path in self._post_sword_paths.items():
                    ptr = self._post_sword_ptrs.get(key, 0)
                    if ptr < len(path) and room == path[ptr]:
                        progress = ptr / max(1, len(path) - 1)
                        step_reward = _PATH_STEP_REWARD * (1.0 + progress)
                        reward += step_reward
                        self._post_sword_ptrs[key] = ptr + 1

            if room != 0:
                self.prev_room = room

        # Update dynamic features in room table
        if room >= 1:
            self._update_room_table_dynamic(room)

        # Truncation on max steps
        if self.step_count >= self.max_steps:
            truncated = True

        self._last_reward_int = reward_int
        obs = self._build_obs()
        return obs, reward + reward_int, terminated, truncated, self._get_info()

    # ── Room Table ─────────────────────────────────────────────────────────────

    def _discover_room(self, room):
        # scan tiles, fill static room_table features (called once per room per episode)
        if room <= 0 or room > NUM_ROOMS:
            return
        idx = room - 1
        start = idx * 30
        fg = self.obs_builder.data.level.fg
        tiles = set(int(fg[start + i]) & 0x1f for i in range(30))
        self.room_table[idx, 0] = 1.0                                     # visited
        self.room_table[idx, 3] = float(4 in tiles)                       # has_gate
        self.room_table[idx, 5] = float(any(t in (5, 6, 15) for t in tiles))  # has_button
        self.room_table[idx, 6] = float(any(t in (2, 18) for t in tiles))     # has_danger
        self.room_table[idx, 7] = float(10 in tiles)                      # has_potion
        self.room_table[idx, 8] = float(16 in tiles or 17 in tiles)       # has_exit

    def _update_room_table_dynamic(self, current_room):
        # cols: 0=visited, 1=is_current, 2=visit_count, 3=has_gate, 4=gate_open, 5=has_button, 6=has_danger, 7=has_potion, 8=has_exit, 9=guard_present, 10=left, 11=right, 12=vert
        # Clear is_current for all rooms, set for current
        self.room_table[:, 1] = 0.0
        if 1 <= current_room <= NUM_ROOMS:
            self.room_table[current_room - 1, 1] = 1.0

        # Update visit count for visited rooms
        counts = self.room_visits_post if self.sword_found else self.room_visits_pre
        for r, count in counts.items():
            if 1 <= r <= NUM_ROOMS:
                self.room_table[r - 1, 2] = min(count, 20) / 20.0

        # Gate open state — only for visited rooms that have gates
        for idx in range(NUM_ROOMS):
            if self.room_table[idx, 0] == 0 or self.room_table[idx, 3] == 0:
                continue
            start = idx * 30
            gate_open = False
            for i in range(30):
                fg_val = int(self.obs_builder.data.level.fg[start + i])
                if (fg_val & 0x1f) == 4 and (fg_val >> 5) >= 2:
                    gate_open = True
                    break
            self.room_table[idx, 4] = float(gate_open)

        # Guard presence — live check for current room only
        guard = self.obs_builder.data.guard
        if 1 <= current_room <= NUM_ROOMS:
            guard_here = (int(guard.room) == current_room and
                          int(self.obs_builder.data.guardhp_max) > 0 and
                          guard.alive < 0)
            self.room_table[current_room - 1, 9] = float(guard_here)

    def _update_connectivity(self, prev_room, curr_room):
        if not (1 <= prev_room <= NUM_ROOMS and 1 <= curr_room <= NUM_ROOMS):
            return
        link = self.obs_builder.data.level.roomlinks[prev_room - 1]
        if link.right == curr_room:
            self.room_table[prev_room - 1, 11] = 1.0   # prev has right exit
            self.room_table[curr_room - 1, 10] = 1.0    # curr has left exit
        elif link.left == curr_room:
            self.room_table[prev_room - 1, 10] = 1.0
            self.room_table[curr_room - 1, 11] = 1.0
        else:  # up or down
            self.room_table[prev_room - 1, 12] = 1.0
            self.room_table[curr_room - 1, 12] = 1.0

    # ── Room Novelty ───────────────────────────────────────────────────────────

    def _room_novelty(self, room):
        # lifetime (every crossing, decays with √N) + episodic (first visit per ep)
        if self.sword_found:
            self.visited_rooms.add(room)
            return 0.0

        counts = self.room_visits_pre
        counts[room] = counts.get(room, 0) + 1

        # Lifetime component: fires every room crossing, decays with count
        bonus = 12.0 / (counts[room] ** 0.5)

        # Episodic component: fires only on first visit per episode
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)
            self._discover_room(room)
            bonus *= 5.0

        return bonus

    # ── Obs / Info ─────────────────────────────────────────────────────────────

    def _build_obs(self):
        obs = self.obs_builder.get_obs(
            self.obs_builder.hitp_max,
            self.action_history, self.repeat_history,
        )
        obs["room_table"] = self.room_table.copy()
        obs["have_sword"] = np.array([float(self.sword_found)], dtype=np.float32)
        return obs

    def _get_info(self):
        info = self.obs_builder.get_info()
        info["frame_count"] = self.frame_count
        info["step_count"] = self.step_count
        info["frames_elapsed"] = self.last_tau
        info["reward_int"] = self._last_reward_int
        return info

    def render(self):
        return self.obs_builder.get_rgb()

import ctypes
from ctypes import c_int, c_short, c_int8, c_int16, c_uint8, c_ushort
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

T_EMPTY=0; T_FLOOR=1; T_SPIKES=2; T_PILLAR=3; T_GATE=4
T_STUCK_BUTTON=5; T_DROP_BUTTON=6; T_TAPESTRY=7
T_BIGPILLAR_BOT=8; T_BIGPILLAR_TOP=9; T_POTION=10; T_LOOSE=11
T_DOORTOP=12; T_MIRROR=13; T_DEBRIS=14; T_RAISE_BUTTON=15
T_EXIT_LEFT=16; T_EXIT_RIGHT=17; T_CHOMPER=18; T_TORCH=19
T_WALL=20; T_SKELETON=21; T_SWORD=22

BUTTON_MAP = {T_STUCK_BUTTON: 1, T_DROP_BUTTON: 2, T_RAISE_BUTTON: 3}
POTION_MAP = {0: 1, 1: 2, 2: 4, 3: 5, 4: 3}
ALIVE = -1
SKELETON_CHARID = 5

CH_TILE=0; CH_GATE=1; CH_LOOSE=2; CH_PLATE=3
CH_POTION=4; CH_CHOMPER=5; CH_KID=6; CH_GUARD=7

NUM_CH = 8
GROWS = 5
GCOLS = 12

ACT_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 99:7}
N_ACT = 8
N_SW = 4       # 0=sheathed 1=drawing 2=drawn 3=putting-away
N_GTYPES = 7
G_SLOTS = 1

# kid: x y col row dir frame act(8) fx fy hp hpmax have_sw sw(4) start_room alive = 25
KID_DIM = 25
# guard slot: present same_room x y col row dir frame act(8) fx fy hp hpmax_abs skill type(7) sw(4) = 32
G_SLOT_DIM = 32
G_DIM = G_SLOTS * G_SLOT_DIM

GRID_FLAT = NUM_CH * GROWS * GCOLS
OBS_DIM = GRID_FLAT + KID_DIM + G_DIM
N_ACTIONS = 18


class PoPEnv(gym.Env):

    def __init__(self, visual=False):
        super().__init__()
        self.visual = visual
        self.SDLPoP_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP")
        os.chdir(self.SDLPoP_path)
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if not visual:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_RENDER_DRIVER"] = "software"

        self.lib = ctypes.CDLL(os.path.join(self.SDLPoP_path, "src", "libSDLPoP.so"))
        c_int.in_dll(self.lib, "rl_mode").value        = 1  # type: ignore
        c_int.in_dll(self.lib, "rl_visual_mode").value  = int(visual)  # type: ignore
        c_short.in_dll(self.lib, "start_level").value   = 1  # type: ignore

        self.lib.pop_main.argtypes     = []; self.lib.pop_main.restype = None
        self.lib.play_level_2.argtypes = []; self.lib.play_level_2.restype = c_int
        self.lib.init_game.argtypes    = [c_int]; self.lib.init_game.restype = None

        # RL checkpoint functions (in-memory save/load of full game state)
        self.lib.rl_save_checkpoint.argtypes  = []; self.lib.rl_save_checkpoint.restype = c_int
        self.lib.rl_load_checkpoint.argtypes  = []; self.lib.rl_load_checkpoint.restype = c_int
        self.lib.rl_checkpoint_is_valid.argtypes = []; self.lib.rl_checkpoint_is_valid.restype = c_int

        self.rl_action   = c_int.in_dll(self.lib, "rl_action")  # type: ignore
        self.rl_dead     = c_int.in_dll(self.lib, "rl_kid_dead")  # type: ignore

        self.grid     = np.zeros((NUM_CH, GROWS, GCOLS), dtype=np.float32)
        self.kid_v    = np.zeros(KID_DIM, dtype=np.float32)
        self.guard_v  = np.zeros(G_DIM, dtype=np.float32)
        self.obs_buf  = np.zeros(OBS_DIM, dtype=np.float32)

        self._blut = np.zeros(32, dtype=np.float32)
        for t, v in BUTTON_MAP.items(): self._blut[t] = v / 3.0
        self._plut = np.zeros(8, dtype=np.float32)
        for k, v in POTION_MAP.items():
            if k < 8: self._plut[k] = v / 7.0

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.steps = 0
        self.max_steps = 30_000
        self.initialized = False

        # Reward tracking state
        self.known_rooms      = set()     # persists across episodes
        self.guard_rooms_seen = {}        # room → True, cross-episode spawn map
        self.episode_rooms    = set()
        self.visited_states   = set()
        self.sword_found      = False
        self.sword_drawn      = False
        self._pending_room    = None
        self._guard_killed_this_ep = set()
        self._guard_kill_count     = 0
        self.room_visit_freq  = {}        # (level,room) → count, cross-episode
        self._phase_transition_fired = False  # True once per episode at sword pickup
        self._checkpoint_available = False

    def get_values(self):
        lib = self.lib
        self.hp     = c_short.in_dll(lib, "hitp_curr").value  # type: ignore
        self.hp_max = c_short.in_dll(lib, "hitp_max").value  # type: ignore
        self.level  = c_short.in_dll(lib, "current_level").value  # type: ignore
        self.have_sword = c_int16.in_dll(lib, "have_sword").value  # type: ignore

        raw = (c_uint8 * 2305).in_dll(lib, "level")  # type: ignore
        lv = np.frombuffer(raw, dtype=np.uint8)
        self.fg  = lv[:720]
        self.bg  = lv[720:1440]
        self.dl2 = lv[1696:1952]
        self.lv  = lv
        self.room_xs = lv[2049:2073]
        self.room_ys = lv[2073:2097]
        self.start_room = int(lv[2112])

        k = np.frombuffer((c_uint8 * 16).in_dll(lib, "Kid"), dtype=np.uint8)  # type: ignore
        self.k_frame = int(k[0]); self.k_x = int(k[1]); self.k_y = int(k[2])
        self.k_dir   = c_int8(k[3]).value
        self.k_col   = c_int8(k[4]).value; self.k_row = c_int8(k[5]).value
        self.k_act   = int(k[6])
        self.k_fx    = c_int8(k[7]).value; self.k_fy = c_int8(k[8]).value
        self.k_room  = int(k[9]); self.k_charid = int(k[11])
        self.k_sword = int(k[12]); self.k_alive = c_int8(k[13]).value

        g = np.frombuffer((c_uint8 * 16).in_dll(lib, "Guard"), dtype=np.uint8)  # type: ignore
        self.g_frame = int(g[0]); self.g_x = int(g[1]); self.g_y = int(g[2])
        self.g_dir   = c_int8(g[3]).value
        self.g_col   = c_int8(g[4]).value; self.g_row = c_int8(g[5]).value
        self.g_act   = int(g[6])
        self.g_fx    = c_int8(g[7]).value; self.g_fy = c_int8(g[8]).value
        self.g_room  = int(g[9]); self.g_charid = int(g[11])
        self.g_sword = int(g[12]); self.g_alive = c_int8(g[13]).value
        self.g_hp    = c_ushort.in_dll(lib, "guardhp_curr").value  # type: ignore
        self.g_hpmax = c_ushort.in_dll(lib, "guardhp_max").value  # type: ignore
        self.g_skill = c_ushort.in_dll(lib, "guard_skill").value  # type: ignore

    def _build_roomlinks(self):
        self.rl = self.lv[1952:1952+96].astype(np.int32).reshape(24, 4)

    def _tile(self, gr, gc, idx):
        bt = int(self.fg[idx]) & 0x1F
        bm = int(self.bg[idx])
        g = self.grid
        g[CH_TILE, gr, gc] = bt / 30.0
        if bt == T_GATE:
            g[CH_GATE, gr, gc] = 1.0 if bm >= 188 else (0.5 if bm > 0 else 0.0)
        elif bt == T_LOOSE:
            g[CH_LOOSE, gr, gc] = 0.0 if bm == 0 else (1.0 if bm & 0x80 else 0.5)
        elif bt in (T_STUCK_BUTTON, T_DROP_BUTTON, T_RAISE_BUTTON):
            timer = self.dl2[bm] & 0x1F if bt != T_STUCK_BUTTON else 0
            g[CH_PLATE, gr, gc] = -self._blut[bt] if timer > 1 else self._blut[bt]
        elif bt == T_CHOMPER:
            g[CH_CHOMPER, gr, gc] = (bm & 0x7F) / 127.0
        elif bt == T_POTION:
            g[CH_POTION, gr, gc] = self._plut[(bm >> 3) & 0x7]

    def _build_grid(self):
        g = self.grid; g[:] = 0.0
        room = self.k_room
        if room < 1 or room > 24: return

        off = (room - 1) * 30
        for i in range(30):
            r, c = divmod(i, 10)
            self._tile(r+1, c+1, off+i)

        if not hasattr(self, "rl"): return
        ri = room - 1
        if ri < 0 or ri >= 24: return
        lk = self.rl[ri]

        # left neighbour col 9 → grid col 0
        nb = int(lk[0])
        if 1 <= nb <= 24:
            noff = (nb-1)*30
            for r in range(3): self._tile(r+1, 0, noff + r*10 + 9)

        # right neighbour col 0 → grid col 11
        nb = int(lk[1])
        if 1 <= nb <= 24:
            noff = (nb-1)*30
            for r in range(3): self._tile(r+1, 11, noff + r*10)

        # up neighbour row 2 → grid row 0
        nb = int(lk[2])
        if 1 <= nb <= 24:
            noff = (nb-1)*30
            for c in range(10): self._tile(0, c+1, noff + 20 + c)

        # down neighbour row 0 → grid row 4
        nb = int(lk[3])
        if 1 <= nb <= 24:
            noff = (nb-1)*30
            for c in range(10): self._tile(4, c+1, noff + c)

    def _agents_on_grid(self):
        g = self.grid
        g[CH_KID] = 0.0; g[CH_GUARD] = 0.0

        if 1 <= self.k_room <= 24:
            r = int(np.clip(self.k_row, 0, 2)) + 1
            c = int(np.clip(self.k_col, 0, 9)) + 1
            g[CH_KID, r, c] = 1.0

        in_room = (self.g_room == self.k_room and self.g_hpmax > 0 and self.g_alive == ALIVE)
        if in_room:
            r = int(np.clip(self.g_row, 0, 2)) + 1
            c = int(np.clip(self.g_col, 0, 9)) + 1
            g[CH_GUARD, r, c] = 1.0
        elif self.g_hpmax > 0 and self.g_alive == ALIVE:
            self._boundary_guard()

    def _boundary_guard(self):
        if not hasattr(self, "rl"): return
        ri = self.k_room - 1
        if ri < 0 or ri >= 24: return
        lk = self.rl[ri]
        gr, gc, grow = self.g_room, self.g_col, self.g_row
        if gr == int(lk[1]) and gc == 0:
            self.grid[CH_GUARD, int(np.clip(grow,0,2))+1, 11] = 1.0
        elif gr == int(lk[0]) and gc == 9:
            self.grid[CH_GUARD, int(np.clip(grow,0,2))+1, 0] = 1.0
        elif gr == int(lk[2]) and grow == 2:
            self.grid[CH_GUARD, 0, int(np.clip(gc,0,9))+1] = 1.0
        elif gr == int(lk[3]) and grow == 0:
            self.grid[CH_GUARD, 4, int(np.clip(gc,0,9))+1] = 1.0

    def _kid_vec(self):
        v = self.kid_v; v[:] = 0.0
        i = 0
        v[i] = self.k_x / 255.0;      i += 1
        v[i] = self.k_y / 255.0;      i += 1
        v[i] = np.clip(self.k_col / 9.0, 0, 1);   i += 1
        v[i] = np.clip(self.k_row / 2.0, 0, 1);   i += 1
        v[i] = 0.0 if self.k_dir >= 0 else 1.0; i += 1
        v[i] = self.k_frame / 255.0;  i += 1
        a = ACT_MAP.get(self.k_act, 0); v[i+a] = 1.0; i += N_ACT
        v[i] = np.clip(self.k_fx / 15.0, -1, 1);  i += 1
        v[i] = np.clip(self.k_fy / 33.0, -1, 1);  i += 1
        v[i] = np.clip(self.hp / max(self.hp_max, 1), 0, 1);  i += 1
        v[i] = np.clip(self.hp_max / 10.0, 0, 1);             i += 1
        v[i] = 1.0 if self.have_sword else 0.0;    i += 1
        sw = min(self.k_sword, N_SW-1); v[i+sw] = 1.0; i += N_SW

        v[i] = 1.0 if self.k_room == self.start_room else 0.0; i += 1

        v[i] = 1.0 if self.k_alive == ALIVE else 0.0

    def _guard_vec(self):
        v = self.guard_v; v[:] = 0.0
        in_room = (self.g_room == self.k_room and self.g_hpmax > 0 and self.g_alive == ALIVE)
        at_border = False
        if not in_room and self.g_hpmax > 0 and self.g_alive == ALIVE:
            at_border = self._g_at_border()
        if not in_room and not at_border: return

        i = 0
        v[i] = 1.0; i += 1
        v[i] = 1.0 if in_room else 0.0; i += 1
        v[i] = self.g_x / 255.0;   i += 1
        v[i] = self.g_y / 255.0;   i += 1
        v[i] = (self.g_col - self.k_col) / 9.0;   i += 1
        v[i] = (self.g_row - self.k_row) / 2.0;   i += 1
        v[i] = 0.0 if self.g_dir >= 0 else 1.0; i += 1
        v[i] = self.g_frame / 255.0; i += 1
        a = ACT_MAP.get(self.g_act, 0); v[i+a] = 1.0; i += N_ACT
        v[i] = np.clip(self.g_fx / 15.0, -1, 1); i += 1
        v[i] = np.clip(self.g_fy / 33.0, -1, 1); i += 1
        v[i] = np.clip(self.g_hp / max(self.g_hpmax, 1), 0, 1);  i += 1
        v[i] = np.clip(self.g_hpmax / 5.0, 0, 1);                i += 1
        v[i] = min(self.g_skill, 11) / 11.0;      i += 1
        gt = min(self.g_charid, N_GTYPES-1); v[i+gt] = 1.0; i += N_GTYPES
        sw = min(self.g_sword, N_SW-1); v[i+sw] = 1.0

    def _g_at_border(self):
        if not hasattr(self, "rl"): return False
        ri = self.k_room - 1
        if ri < 0 or ri >= 24: return False
        lk = self.rl[ri]
        gr, gc, grow = self.g_room, self.g_col, self.g_row
        if gr == int(lk[1]) and gc == 0: return True
        if gr == int(lk[0]) and gc == 9: return True
        if gr == int(lk[2]) and grow == 2: return True
        if gr == int(lk[3]) and grow == 0: return True
        return False

    def _get_obs(self):
        self._build_grid()
        self._agents_on_grid()
        self._kid_vec()
        self._guard_vec()
        buf = self.obs_buf
        buf[:GRID_FLAT] = self.grid.ravel()
        buf[GRID_FLAT:GRID_FLAT+KID_DIM] = self.kid_v
        buf[GRID_FLAT+KID_DIM:] = self.guard_v
        return buf.copy()



    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        if not self.initialized:
            self.lib.pop_main()
            self.initialized = True
        else:
            self.lib.init_game(1)

        self.get_values()
        self._build_roomlinks()
        self.rl_dead.value = 0
        self.steps = 0

        # Reset per-episode reward state
        self.episode_rooms = set()
        self.episode_rooms.add((self.level, self.k_room))
        self.prev_level   = self.level
        self.prev_hp      = self.hp
        self.prev_hp_max  = self.hp_max
        self.prev_guard_hp = None
        self.sword_found  = self.have_sword > 0
        self.sword_drawn  = self.k_sword == 2
        self.prev_guard_room = self.g_room
        self.visited_states  = set()
        self._pending_room   = None
        self._guard_killed_this_ep = set()
        self._guard_kill_count     = 0
        self._phase_transition_fired = self.sword_found  # only fire on NEW sword pickups
        self._checkpoint_available = False

        info = {
            "level": self.level,
            "room":  self.k_room,
            "hp":    self.hp,
        }
        return self._get_obs(), info

    def step(self, action):
        self.rl_action.value = int(action)
        for _ in range(4):
            self.lib.play_level_2()
            if self.visual:
                time.sleep(1.0 / 15.0)
            if self.rl_dead.value == 1: break

        self.steps += 1
        prev_lv = self.level
        self.get_values()
        if self.level != prev_lv:
            self._build_roomlinks()

        alive = self.rl_dead.value != 1
        room  = self.k_room
        hp    = self.hp
        level = self.level

        # ── Reward ──
        reward = -0.05 if self.sword_found else -0.01  # larger P2 penalty for urgency

        if not alive:
            reward -= 5.0

        if self.prev_hp is not None and hp < self.prev_hp:
            reward -= 0.5 * (self.prev_hp - hp)
        self.prev_hp = hp

        if self.hp_max > self.prev_hp_max:
            reward += 20.0
        self.prev_hp_max = self.hp_max

        # Curiosity
        gate_key = np.packbits((self.grid[CH_GATE] > 0).flatten().astype(bool)).tobytes()
        curiosity_state = (level, room, self.k_col, self.k_row,
                           1 if self.have_sword > 0 else 0, gate_key)
        if curiosity_state not in self.visited_states:
            reward += 0.1
            self.visited_states.add(curiosity_state)

        guard_in_room = (self.g_room == room and self.g_hpmax > 0 and self.g_alive == ALIVE)

        # Sword pickup — single highest one-time reward
        if self.have_sword and not self.sword_found:
            reward += 25.0
            self.sword_found = True

        # Guard memory
        if guard_in_room and self.g_alive == ALIVE:
            self.guard_rooms_seen[room] = True

        # Combat rewards
        if guard_in_room:
            if self.prev_guard_hp is None:
                self.prev_guard_hp = self.g_hp

            kid_sword_drawn = self.k_sword == 2
            if kid_sword_drawn and not self.sword_drawn and self.g_charid != SKELETON_CHARID:
                reward += 15.0
            self.sword_drawn = kid_sword_drawn

            if self.g_charid != SKELETON_CHARID:
                if self.prev_guard_hp > 0 and self.g_hp < self.prev_guard_hp:
                    damage = self.prev_guard_hp - self.g_hp
                    reward += 10.0 * damage
                if self.prev_guard_hp is not None and self.prev_guard_hp > 0 and self.g_hp == 0:
                    reward += 300.0
                    self._guard_kill_count += 1
                    self._guard_killed_this_ep.add(room)
                    # Reset visit freq for all rooms — guard kill unlocks new paths
                    self.room_visit_freq.clear()

            self.prev_guard_hp = self.g_hp
        else:
            self.prev_guard_hp = None
            self.sword_drawn = False

        # Skeleton knockoff
        current_guard_room = self.g_room
        if (self.g_charid == SKELETON_CHARID and self.prev_guard_room == room and
            current_guard_room != room and self.g_alive == ALIVE and self.g_sword == 2):
            reward += 300.0
        self.prev_guard_room = current_guard_room

        # Level up
        if level > self.prev_level:
            reward += 500.0
            self.room_visit_freq.clear()  # reset decay on level change
            self.prev_level = level
            self._pending_room = None

        # Frontier / room exploration
        frontier_connections = 0
        if alive and self._pending_room is not None:
            pending_key = self._pending_room
            pr = pending_key[1]

            if pending_key not in self.known_rooms:
                self.known_rooms.add(pending_key)
                self.episode_rooms.add(pending_key)
                if 1 <= pr <= 24 and hasattr(self, "rl"):
                    lk = self.rl[pr - 1]
                    for d in range(4):
                        nb = int(lk[d])
                        if 1 <= nb <= 24 and (level, nb) not in self.known_rooms:
                            frontier_connections += 1
                reward += 25.0 * (1 + frontier_connections)

            elif pending_key not in self.episode_rooms:
                self.episode_rooms.add(pending_key)
                has_unexplored = False
                has_guard = False
                if 1 <= pr <= 24 and hasattr(self, "rl"):
                    lk = self.rl[pr - 1]
                    has_unexplored = any(
                        1 <= int(lk[d]) <= 24 and (level, int(lk[d])) not in self.known_rooms
                        for d in range(4)
                    )
                    active_guard_rooms = {
                        r for r in self.guard_rooms_seen
                        if r not in self._guard_killed_this_ep
                    }
                    has_guard = pr in active_guard_rooms and bool(self.have_sword)
                base = 15.0 if has_guard else (10.0 if has_unexplored else (8.0 if self.sword_found else 5.0))
                # Decay only for P2 (post-sword) — P1 needs full rewards for permadeath navigation
                if self.sword_found:
                    freq = self.room_visit_freq.get(pending_key, 0)
                    decay = max(0.3, 0.95 ** freq)
                    reward += base * decay
                    self.room_visit_freq[pending_key] = freq + 1
                else:
                    reward += base

            self._pending_room = None

        if alive:
            room_key = (level, room)
            if room_key not in self.episode_rooms:
                self._pending_room = room_key
        else:
            self._pending_room = None

        obs = self._get_obs()
        terminated = not alive
        truncated  = self.steps >= self.max_steps

        guard_fought = 1 if (guard_in_room and self.k_sword == 2) else 0
        active_guard_rooms_info = {
            r for r in self.guard_rooms_seen if r not in self._guard_killed_this_ep
        }
        # Detect phase transition: sword picked up this step
        phase_transition = False
        if not self._phase_transition_fired and self.sword_found:
            phase_transition = True
            self._phase_transition_fired = True
            # Save checkpoint at sword pickup for curriculum training
            self.save_checkpoint()

        info = {
            "level":                  level,
            "room":                   room,
            "hp":                     hp,
            "steps":                  self.steps,
            "dead":                   not alive,
            "frontier_connections":   frontier_connections,
            "episode_rooms":          len(self.episode_rooms),
            "guard_hp":               self.g_hp if guard_in_room else -1,
            "guard_hp_max":           self.g_hpmax if guard_in_room else 0,
            "kid_sword_drawn":        1 if self.sword_drawn else 0,
            "visited_tiles_count":    len(self.visited_states),
            "sword_found":            1 if self.sword_found else 0,
            "guard_fought":           guard_fought,
            "guard_killed":           self._guard_kill_count,
            "guard_rooms_seen_count": len(active_guard_rooms_info),
            "phase_transition":       phase_transition,
        }
        return obs, reward, terminated, truncated, info

    def get_known_rooms(self):
        return set(self.known_rooms)

    def set_known_rooms(self, rooms):
        self.known_rooms = set(rooms)

    def get_guard_rooms_seen(self):
        return dict(self.guard_rooms_seen)

    def set_guard_rooms_seen(self, rooms):
        self.guard_rooms_seen = dict(rooms)

    # ---- Checkpoint API for curriculum training ----
    def save_checkpoint(self):
        """Save current game state to in-memory buffer in C."""
        ok = self.lib.rl_save_checkpoint()
        if ok:
            self._checkpoint_available = True
        return ok

    def load_checkpoint(self):
        """Restore game state from in-memory checkpoint buffer."""
        ok = self.lib.rl_load_checkpoint()
        if ok:
            self.get_values()  # Re-read all Python-side state from C
            self.sword_found = bool(self.have_sword)
            self._phase_transition_fired = self.sword_found
        return ok

    def has_checkpoint(self):
        """Check if a valid checkpoint exists."""
        return bool(self.lib.rl_checkpoint_is_valid())

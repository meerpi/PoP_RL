import ctypes
from ctypes import c_int, c_short
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


# ── Tile type constants ──
ALIVE           = -1
T_EMPTY         = 0
T_FLOOR         = 1
T_SPIKES        = 2
T_PILLAR        = 3
T_GATE          = 4
T_STUCK_BUTTON  = 5
T_DROP_BUTTON   = 6
T_TAPESTRY      = 7
T_BIGPILLAR_BOT = 8
T_BIGPILLAR_TOP = 9
T_POTION        = 10
T_LOOSE         = 11
T_DOORTOP       = 12
T_MIRROR        = 13
T_DEBRIS        = 14
T_RAISE_BUTTON  = 15
T_EXIT_LEFT     = 16
T_EXIT_RIGHT    = 17
T_CHOMPER       = 18
T_TORCH         = 19
T_WALL          = 20
T_SKELETON      = 21
T_SWORD         = 22

BUTTON_MAP   = {T_STUCK_BUTTON: 1, T_DROP_BUTTON: 2, T_RAISE_BUTTON: 3}
POTION_MAP   = {0: 1, 1: 2, 2: 4, 3: 5, 4: 3}
SKELETON_CHARID = 5

# --- Channel Constants ---
CH_TILE_TYPE    = 0
CH_LOOSE_STATE  = 1
CH_GATE         = 2
CH_BUTTON       = 3
CH_CHOMPER      = 4
CH_ITEM         = 5
CH_SPIKE_STATE  = 6
CH_POTION_HARM  = 7

NUM_CHANNELS    = 8
ROWS, COLS      = 3, 10

class PoPEnv(gym.Env):

    def __init__(self, visual=False):
        self.visual = visual
        self.SDLPoP_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP")
        self.so_path = os.path.join(self.SDLPoP_path, "src", "libSDLPoP.so")
        os.chdir(self.SDLPoP_path)
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if not self.visual:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_RENDER_DRIVER"] = "software"
        self.lib = ctypes.CDLL(self.so_path)
        ctypes.c_int.in_dll(self.lib, "rl_step_mode").value = 1
        ctypes.c_int.in_dll(self.lib, "rl_visual_mode").value = 1 if self.visual else 0
        ctypes.c_short.in_dll(self.lib, "start_level").value = 1
        STATE_DIM  = 29
        ACTION_COUNT = 18
        act_hist   = 5

        self.grid         = np.zeros((NUM_CHANNELS, ROWS, COLS), dtype=np.float32)
        self.state        = np.zeros(STATE_DIM, dtype=np.float32)
        self.action_history = np.zeros(act_hist, dtype=np.int64)

        self.flat_obs_size = (NUM_CHANNELS * ROWS * COLS) + STATE_DIM + act_hist
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=-1.0, high=ACTION_COUNT, shape=(self.flat_obs_size,), dtype=np.float32)

        self.obs_buf = np.zeros(self.flat_obs_size, dtype=np.float32)

        self.step_count  = 0
        self.max_steps   = 30_000
        self.initialized = False
        
        self.room_xs = np.zeros(24, dtype=np.uint8)
        self.room_ys = np.zeros(24, dtype=np.uint8)
        self.episode_rooms = set()
        self.known_rooms = set()    # persists across episodes — never cleared
        self.visited_states = set()
        self.sword_found = False
        self.sword_drawn = False
        self._pending_room = None
        self._sword_phase_step = 0          # steps elapsed since sword pickup
        self.guard_rooms_seen = {}          # room → True, CROSS-EPISODE spawn map; never cleared
        self._guard_killed_this_ep = set()  # rooms where guard was killed THIS episode
        self._guard_kill_count = 0          # kill counter this episode

        self._button_lut = np.zeros(32, dtype=np.float32)
        for t, v in BUTTON_MAP.items():
            self._button_lut[t] = v / 3.0

        self._potion_lut = np.zeros(8, dtype=np.float32)
        for k, v in POTION_MAP.items():
            if k < 8:
                self._potion_lut[k] = v / 7.0

        self.lib.pop_main.argtypes    = []
        self.lib.pop_main.restype     = None
        self.lib.play_level_2.argtypes = []
        self.lib.play_level_2.restype  = ctypes.c_int

        self.lib.init_game.argtypes = [ctypes.c_int]
        self.lib.init_game.restype  = None

        self.is_restart_level = ctypes.c_int.in_dll(self.lib, "is_restart_level")
        self.rl_kid_dead_ptr = ctypes.c_int.in_dll(self.lib, "rl_kid_dead")

    def _build_roomlinks(self):
        self.roomlinks = []
        for r in range(24):
            b = 1952 + r * 4
            self.roomlinks.append({
                "left":  int(self.level_np[b]),
                "right": int(self.level_np[b + 1]),
                "up":    int(self.level_np[b + 2]),
                "down":  int(self.level_np[b + 3]),
            })

    def _load_room_coords(self):
        dat_name = f"res{2000 + self.current_level - 1}.bin"
        dat_path = os.path.join(self.SDLPoP_path, "data", "LEVELS", dat_name)
        if os.path.exists(dat_path):
            with open(dat_path, "rb") as f:
                dat_buf = f.read()
            if len(dat_buf) >= 2305:
                self.room_xs = np.array(list(dat_buf[2049:2073]), dtype=np.uint8)
                self.room_ys = np.array(list(dat_buf[2073:2097]), dtype=np.uint8)
            else:
                self.room_xs = np.zeros(24, dtype=np.uint8)
                self.room_ys = np.zeros(24, dtype=np.uint8)
        else:
            self.room_xs = np.zeros(24, dtype=np.uint8)
            self.room_ys = np.zeros(24, dtype=np.uint8)

    def get_values(self):
        self.hitp_curr = c_short.in_dll(self.lib, "hitp_curr").value
        self.hitp_max = c_short.in_dll(self.lib, "hitp_max").value
        self.current_level = c_short.in_dll(self.lib, "current_level").value

        self.action = c_int.in_dll(self.lib, "rl_action").value
        self.rl_step_mode = c_int.in_dll(self.lib, "rl_step_mode").value
        self.start_level = c_short.in_dll(self.lib, "start_level").value
        
        self.raw_level = (ctypes.c_uint8 * 2305).in_dll(self.lib, "level")

        level_np = np.frombuffer(self.raw_level, dtype=np.uint8)

        self.fg = level_np[:720]
        self.bg = level_np[720:1440]
        self.doorlinks1 = level_np[1440:1696]
        self.doorlinks2 = level_np[1696:1952]

        self.level_np = level_np

        self.start_pos = int(level_np[2113])
        self.start_room = int(level_np[2112])
        self.start_dir = ctypes.c_int8(level_np[2114]).value

        self.guards_tile = level_np[2119:2143]
        self.guards_dir = level_np[2143:2167]
        self.guards_x = level_np[2167:2191]
        self.guards_skill = level_np[2215:2239]


        self.kid_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Kid")
        self.guard_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Guard")

        kid_np = np.frombuffer(self.kid_raw, dtype=np.uint8)
        guard_np = np.frombuffer(self.guard_raw, dtype=np.uint8)


        self.kid_frame = int(kid_np[0])
        self.kid_x = int(kid_np[1])
        self.kid_y = int(kid_np[2])
        self.kid_direction = ctypes.c_int8(kid_np[3]).value
        self.kid_curr_col = ctypes.c_int8(kid_np[4]).value
        self.kid_curr_row = ctypes.c_int8(kid_np[5]).value
        self.kid_action = int(kid_np[6])
        self.kid_fall_x = ctypes.c_int8(kid_np[7]).value
        self.kid_fall_y = ctypes.c_int8(kid_np[8]).value
        self.kid_room = int(kid_np[9])
        self.kid_repeat = int(kid_np[10])
        self.kid_charid = int(kid_np[11])
        self.kid_sword = int(kid_np[12])
        self.have_sword = ctypes.c_int16.in_dll(self.lib, "have_sword").value
        self.kid_alive = ctypes.c_int8(kid_np[13]).value
        self.kid_curr_seq = int(kid_np[14]) | (int(kid_np[15]) << 8)


        self.guard_frame = int(guard_np[0])
        self.guard_x = int(guard_np[1])
        self.guard_y = int(guard_np[2])
        self.guard_direction = ctypes.c_int8(guard_np[3]).value
        self.guard_curr_col = ctypes.c_int8(guard_np[4]).value
        self.guard_curr_row = ctypes.c_int8(guard_np[5]).value
        self.guard_action = int(guard_np[6])
        self.guard_fall_x = ctypes.c_int8(guard_np[7]).value
        self.guard_fall_y = ctypes.c_int8(guard_np[8]).value
        self.guard_room = int(guard_np[9])
        self.guard_repeat = int(guard_np[10])
        self.guard_charid = int(guard_np[11])
        self.guard_sword = int(guard_np[12])
        self.guard_alive = ctypes.c_int8(guard_np[13]).value
        self.guard_curr_seq = int(guard_np[14]) | (int(guard_np[15]) << 8)
        self.guardhp_curr = ctypes.c_ushort.in_dll(self.lib, "guardhp_curr").value
        self.guardhp_max = ctypes.c_ushort.in_dll(self.lib, "guardhp_max").value



    def create_grid(self, room=None):
        if room is None:
            room = self.kid_room
        if room < 1 or room > 24:
            self.grid[:] = 0
            return self.grid

        room_offset = (room - 1) * 30
        room_fg = self.fg[room_offset : room_offset + 30]
        room_bg = self.bg[room_offset : room_offset + 30]

        t_arr = (room_fg.astype(np.uint8) & 0x1F)
        m_arr = room_bg.astype(np.uint8)

        gate_raw  = np.where(t_arr == T_GATE,
                     np.where(m_arr == 0, 1, np.where(m_arr >= 188, 3, 2)),
                     0).astype(np.float32)
        loose_raw = np.where(t_arr == T_LOOSE,
                     np.where(m_arr != 0, 2, 1), 0).astype(np.float32)
        chomp_raw = np.where(t_arr == T_CHOMPER, (m_arr & 0x7F).astype(np.float32), 0.0)
        spike_raw = np.where(t_arr == T_SPIKES,  (m_arr & 0x0F).astype(np.float32), 0.0)

        t = t_arr.reshape(3, 10)
        m = m_arr.reshape(3, 10)

        grid = self.grid
        grid[:] = 0

        grid[CH_TILE_TYPE]   = np.clip(t, 0, 30) / 30.0
        grid[CH_LOOSE_STATE] = loose_raw.reshape(3, 10) / 2.0
        grid[CH_GATE]        = gate_raw.reshape(3, 10) / 3.0
        grid[CH_BUTTON]      = self._button_lut[t]
        grid[CH_CHOMPER]     = chomp_raw.reshape(3, 10) / 127.0
        grid[CH_SPIKE_STATE] = spike_raw.reshape(3, 10) / 15.0

        item = np.zeros((3, 10), dtype=np.float32)
        item = np.where(t == T_POTION, self._potion_lut[(m >> 3) & 0x7], item)
        item = np.where(t == T_SWORD,  6.0 / 7.0, item)
        item = np.where((t == T_EXIT_LEFT) | (t == T_EXIT_RIGHT), 1.0, item)
        grid[CH_ITEM] = item

        harm = np.zeros((3, 10), dtype=np.float32)
        harm = np.where((t == T_POTION) & (((m >> 3) & 0x7) == 3), 1.0, harm)
        grid[CH_POTION_HARM] = harm


        room_idx = room - 1
        if 0 <= room_idx < 24 and hasattr(self, "roomlinks"):
            links = self.roomlinks[room_idx]
            for nb_room, grid_col, nb_col, grid_row, nb_row in [
                (links["left"],  0, 9, None, None),
                (links["right"], 9, 0, None, None),
                (links["up"],   None, None, 0, 2),
                (links["down"], None, None, 2, 0),
            ]:
                if nb_room < 1 or nb_room > 24:
                    continue
                nb_offset = (nb_room - 1) * 30
                rows_to_check = range(ROWS) if grid_row is None else [grid_row]
                cols_to_check = range(COLS) if grid_col is None else [grid_col]
                for r in rows_to_check:
                    for c in cols_to_check:
                        n_r = r if nb_row is None else nb_row
                        n_c = c if nb_col is None else nb_col
                        nb_idx = nb_offset + n_r * COLS + n_c
                        bt = int(self.fg[nb_idx]) & 0x1F
                        bm = int(self.bg[nb_idx])
                        if bt == T_GATE:
                            gs = 1.0 if bm == 0 else (3.0 if bm >= 188 else 2.0)
                            grid[CH_GATE, r, c] = max(grid[CH_GATE, r, c], gs / 3.0)
                        if bt == T_CHOMPER:
                            grid[CH_CHOMPER, r, c] = max(grid[CH_CHOMPER, r, c], (bm & 0x7F) / 127.0)
                        if bt == T_SPIKES:
                            grid[CH_SPIKE_STATE, r, c] = max(grid[CH_SPIKE_STATE, r, c], (bm & 0x0F) / 15.0)
                        if bt == T_POTION:
                            grid[CH_ITEM, r, c] = POTION_MAP.get((bm >> 3) & 0x7, 1) / 7.0
                            if ((bm >> 3) & 0x7) == 3:
                                grid[CH_POTION_HARM, r, c] = 1.0
                        elif bt == T_SWORD:
                            grid[CH_ITEM, r, c] = 6.0 / 7.0
                        elif bt in (T_EXIT_LEFT, T_EXIT_RIGHT):
                            grid[CH_ITEM, r, c] = 1.0

        return grid

    def get_obs_state(self):
        guard_in_room = (self.guard_room == self.kid_room and
                         self.guardhp_max > 0 and
                         self.guard_alive == ALIVE)

        self.state[0]  = self.hitp_curr / max(self.hitp_max, 1.0)
        self.state[1]  = self.hitp_max / 10.0
        self.state[2]  = self.current_level / 14.0
        self.state[3]  = np.clip(self.kid_fall_y / 10.0, -1.0, 1.0)
        self.state[4]  = np.clip(self.kid_fall_x / 10.0, -1.0, 1.0)
        self.state[5]  = 1.0 if self.have_sword != 0 else 0.0
        self.state[6]  = 1.0 if self.kid_room == self.start_room else 0.0
        room_idx = max(0, min(self.kid_room - 1, 23))
        bx = int(self.room_xs[room_idx]) if self.room_xs[room_idx] != 255 else 0
        by = int(self.room_ys[room_idx]) if self.room_ys[room_idx] != 255 else 0
        col = max(0, min(9, self.kid_curr_col))
        row = max(0, min(2, self.kid_curr_row))
        self.state[7]  = ((bx * 10) + col) / 250.0
        self.state[8]  = ((by * 3)  + row) / 100.0
        self.state[9]  = self.kid_x / 255.0
        self.state[10] = self.kid_y / 255.0
        self.state[11] = col / 9.0
        self.state[12] = row / 2.0
        self.state[13] = 1.0 if self.kid_direction < 0 else 0.0
        self.state[14] = float(min(self.kid_sword, 2)) / 2.0
        self.state[15] = self.kid_action / 255.0
        self.state[16] = self.kid_frame / 255.0
        # ── Guard entity scalars (17-23) ──
        self.state[17] = 1.0 if guard_in_room else 0.0
        if guard_in_room:
            g_col = max(0, min(9, self.guard_curr_col))
            self.state[18] = (g_col - col) / 9.0
            self.state[19] = max(0, min(2, self.guard_curr_row)) / 2.0
            self.state[20] = self.guardhp_curr / max(self.guardhp_max, 1.0)
            self.state[21] = 1.0 if self.guard_direction < 0 else 0.0
            self.state[22] = float(min(self.guard_charid, 5)) / 5.0
            self.state[23] = self.guard_action / 255.0
        else:
            self.state[18:24] = 0.0
        # ── Exploration memory (24-27) ──
        self.state[24] = 1.0 if (self.current_level, self.kid_room) in self.episode_rooms else 0.0
        self.state[25] = min(len(self.episode_rooms), 24) / 24.0
        self.state[26] = float(np.any(np.abs(self.grid[CH_ITEM] - 6.0/7.0) < 0.01))
        if 1 <= self.kid_room <= 24 and hasattr(self, "roomlinks"):
            links = self.roomlinks[self.kid_room - 1]
            self.state[27] = sum(1 for nb in links.values() if 1 <= nb <= 24) / 4.0
        else:
            self.state[27] = 0.0
        # ── Guard memory (28) — nearby active-guard room signal ──
        # Excludes rooms where the guard was killed this episode
        room = self.kid_room
        if self.have_sword and 1 <= room <= 24 and hasattr(self, "roomlinks"):
            active_gr = {r for r in self.guard_rooms_seen if r not in self._guard_killed_this_ep}
            has_nearby_guard = any(
                nb in active_gr
                for nb in self.roomlinks[room - 1].values()
                if 1 <= nb <= 24
            )
            self.state[28] = 1.0 if (room in active_gr or has_nearby_guard) else 0.0
        else:
            self.state[28] = 0.0

        return self.state

    def _get_obs(self):
        self.get_obs_state()  # updates self.state in-place

        idx = 0
        grid_flat = self.grid.ravel()   # (240,)
        self.obs_buf[idx : idx + 240] = grid_flat
        idx += 240
        self.obs_buf[idx : idx + len(self.state)] = self.state
        idx += len(self.state)
        self.obs_buf[idx : idx + len(self.action_history)] = self.action_history
        return self.obs_buf

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.initialized:
            self.lib.pop_main()
            self.initialized = True
        else:
            self.lib.init_game(1)


        self.get_values()
        self._load_room_coords()
        self._build_roomlinks()
        self.rl_kid_dead_ptr.value = 0

        self.step_count    = 0
        self.action_history[:] = 0
        self.episode_rooms = set()
        self.episode_rooms.add((self.current_level, self.kid_room))

        self.prev_level = self.current_level
        self.prev_hp = self.hitp_curr
        self.prev_hitp_max = self.hitp_max
        self.prev_guard_hp = None

        self.sword_found = self.have_sword > 0
        self.sword_drawn = self.kid_sword == 2
        self.prev_guard_room = self.guard_room
        self.visited_states = set()
        self._pending_room = None
        self._sword_phase_step = 0
        # Note: guard_rooms_seen is NOT reset here — it's cross-episode like known_rooms
        self._guard_killed_this_ep = set()
        self._guard_kill_count = 0

        self.create_grid()

        info = {
            "level": self.current_level,
            "room": self.kid_room,
            "hp": self.hitp_curr,
            "steps_alive": self.step_count,
            "deaths": 1 if self.rl_kid_dead_ptr.value == 1 else 0,
        }
        return self._get_obs(), info

    def apply_action(self, action_idx):
        ctypes.c_int.in_dll(self.lib, "rl_action").value = action_idx

    def step(self, action):
        self.apply_action(action)
        
        reward = -0.01
        terminated = False

        for _ in range(4):
            self.lib.play_level_2()
            if self.visual:
                time.sleep(1.0 / 15.0)

            alive = self.rl_kid_dead_ptr.value != 1

            if not alive:
                terminated = True
                break

        self.step_count += 1
        self.get_values()
        self.create_grid()
        
        room = self.kid_room
        hp = self.hitp_curr
        level = self.current_level
        
        if not alive:
            reward -= 5.0
        
        if self.prev_hp is not None and hp < self.prev_hp:
            reward -= 0.5 * (self.prev_hp - hp)
        self.prev_hp = hp
        
        if self.hitp_max > self.prev_hitp_max:
            reward += 20.0
        self.prev_hitp_max = self.hitp_max

        gate_key = np.packbits((self.grid[CH_GATE] > 0).flatten().astype(bool)).tobytes()
        curiosity_state = (self.current_level, room, self.kid_curr_col, self.kid_curr_row,
                           1 if self.have_sword > 0 else 0, gate_key)
        if curiosity_state not in self.visited_states:
            reward += 0.1
            self.visited_states.add(curiosity_state)

        # Hoist guard state so guard memory + sword-phase logic can use it
        guard_hp = self.guardhp_curr
        guard_in_room = (self.guard_room == room and self.guardhp_max > 0 and self.guard_alive == ALIVE)

        if self.have_sword and not self.sword_found:
            reward += 70.0
            self.sword_found = True
            # ── Phase 2 begins: treat return journey as fresh exploration ──
            self.episode_rooms = {(self.current_level, room)}
            self._sword_phase_step = 0

        if self.have_sword:
            self._sword_phase_step += 1

        # ── Guard memory: permanent spawn map + within-episode kill tracking ──
        if guard_in_room and self.guard_alive == ALIVE:
            self.guard_rooms_seen[room] = True      # permanent — guard spawns here
        # Note: kill detection happens below in the combat block (hp transition 0→0)

        # guard_hp and guard_in_room already computed above
        
        if guard_in_room:
            if self.prev_guard_hp is None:
                self.prev_guard_hp = guard_hp
                
            kid_sword_drawn = self.kid_sword == 2
            if kid_sword_drawn and not self.sword_drawn and self.guard_charid != SKELETON_CHARID:
                reward += 15.0
            self.sword_drawn = kid_sword_drawn

            if self.guard_charid != SKELETON_CHARID:
                if self.prev_guard_hp > 0 and guard_hp < self.prev_guard_hp:
                    damage = self.prev_guard_hp - guard_hp
                    reward += 10.0 * damage
                if self.prev_guard_hp is not None and self.prev_guard_hp > 0 and guard_hp == 0:
                    reward += 300.0
                    self._guard_kill_count += 1
                    self._guard_killed_this_ep.add(room)  # mark as dead this episode
            
            self.prev_guard_hp = guard_hp
        else:
            self.prev_guard_hp = None
            self.sword_drawn = False

        current_guard_room = self.guard_room
        if (self.guard_charid == SKELETON_CHARID and self.prev_guard_room == room and 
            current_guard_room != room and self.guard_alive == ALIVE and self.guard_sword == 2):
            reward += 300.0  # Successfully knocked the skeleton off the ledge
        self.prev_guard_room = current_guard_room

        if level > self.prev_level:
            reward += 500.0
            self.prev_level = level
            self._pending_room = None   # discard stale previous-level room; new level queued below
            self._load_room_coords()
            self._build_roomlinks()


        frontier_connections = 0
        if not terminated and self._pending_room is not None:
            pending_key = self._pending_room
            pr = pending_key[1]

            if pending_key not in self.known_rooms:
                # Tier 1: new discovery
                self.known_rooms.add(pending_key)
                self.episode_rooms.add(pending_key)
                if 1 <= pr <= 24 and hasattr(self, "roomlinks"):
                    links = self.roomlinks[pr - 1]
                    for nb in [links["left"], links["right"], links["up"], links["down"]]:
                        if 1 <= nb <= 24 and (self.current_level, nb) not in self.known_rooms:
                            frontier_connections += 1
                reward += 25.0 * (1 + frontier_connections)

            elif pending_key not in self.episode_rooms:
                # Tier 2: known room, first visit this episode — weighted by frontier value.
                # In sword phase (phase 2), base rewards are tripled so GAE can bridge
                # the return journey in ~20-step hops rather than one 200-step leap.
                self.episode_rooms.add(pending_key)
                has_unexplored = False
                has_guard = False
                if 1 <= pr <= 24 and hasattr(self, "roomlinks"):
                    links = self.roomlinks[pr - 1]
                    has_unexplored = any(
                        1 <= nb <= 24 and (self.current_level, nb) not in self.known_rooms
                        for nb in links.values()
                    )
                    active_guard_rooms = {
                        r for r in self.guard_rooms_seen
                        if r not in self._guard_killed_this_ep
                    }
                    has_guard = pr in active_guard_rooms and bool(self.have_sword)
                base = 4.0 if has_unexplored else (3.0 if has_guard else 1.0)
                if self.have_sword:
                    base *= 3.0   # +3 / +9 near guard / +12 near unknown — strong return gradient
                reward += base

            self._pending_room = None


        if not terminated:
            room_key = (self.current_level, room)
            if room_key not in self.episode_rooms:
                self._pending_room = room_key
        else:
            self._pending_room = None

        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = action

        truncated = self.step_count >= self.max_steps

        guard_fought = 1 if (guard_in_room and self.kid_sword == 2) else 0
        # Compute active guard rooms for state[28] — also used in info
        active_guard_rooms_info = {
            r for r in self.guard_rooms_seen if r not in self._guard_killed_this_ep
        }
        info = {
            "level":                    level,
            "room":                     room,
            "hp":                       hp,
            "steps_alive":              self.step_count,
            "deaths":                   1 if self.rl_kid_dead_ptr.value == 1 else 0,
            "frontier_connections":     frontier_connections,
            "episode_rooms":            len(self.episode_rooms),
            "guard_hp":                 self.guardhp_curr if guard_in_room else -1,
            "guard_hp_max":             self.guardhp_max if guard_in_room else 0,
            "kid_sword_drawn":          1 if self.sword_drawn else 0,
            "visited_tiles_count":      len(self.visited_states),
            "sword_found":              1 if self.sword_found else 0,
            # PEIT events
            "guard_fought":             guard_fought,
            "guard_killed":             self._guard_kill_count,  # total kills this episode
            "sword_phase_step":         self._sword_phase_step,
            "guard_rooms_seen_count":   len(active_guard_rooms_info),  # active (non-killed) guards
        }
        return self._get_obs(), reward, terminated, truncated, info

    def get_known_rooms(self):
        return set(self.known_rooms)

    def set_known_rooms(self, rooms):
        self.known_rooms = set(rooms)

    def get_guard_rooms_seen(self):
        return dict(self.guard_rooms_seen)

    def set_guard_rooms_seen(self, rooms):
        self.guard_rooms_seen = dict(rooms)

import ctypes
from ctypes import c_int, c_short, c_char_p, c_void_p, cast, create_string_buffer, c_uint8
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Tile type constants ──
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

WALKABLE_TYPES = {
    T_FLOOR, T_BIGPILLAR_TOP, T_LOOSE, T_DOORTOP, T_DEBRIS,
    T_RAISE_BUTTON, T_EXIT_LEFT, T_EXIT_RIGHT,
    T_TORCH, T_SKELETON, T_SWORD,
    T_STUCK_BUTTON, T_DROP_BUTTON,
}
WALL_TYPES   = {T_WALL, T_PILLAR, T_BIGPILLAR_BOT, T_TAPESTRY}
BUTTON_MAP   = {T_STUCK_BUTTON: 1, T_DROP_BUTTON: 2, T_RAISE_BUTTON: 3}
POTION_MAP   = {0: 1, 1: 2, 2: 4, 4: 3}

# --- Channel Constants ---
CH_TILE_TYPE    = 0
CH_WALKABLE     = 1
CH_WALL         = 2
CH_FLOOR_EDGE   = 3
CH_LOOSE_STATE  = 4
CH_GATE         = 5
CH_BUTTON       = 6
CH_CHOMPER      = 7
CH_PLAYER_POS   = 8
CH_PLAYER_ACT   = 9
CH_ENEMY_POS    = 10
CH_ENEMY_DIR    = 11
CH_ITEM         = 12
CH_PLAYER_DIR   = 13
CH_PLAYER_SWORD = 14
CH_PLAYER_FRAME = 15
CH_ENEMY_ACT    = 16
CH_ENEMY_SWORD  = 17
CH_ENEMY_TYPE   = 18
CH_ENEMY_FRAME  = 19

NUM_CHANNELS    = 20
ROWS, COLS      = 3, 10

# Precomputed row index array for floor_edge channel
_ROW_IDX = np.arange(3, dtype=np.int32)[:, None]  # shape (3,1)


class PoPEnv(gym.Env):
    def _set_g_argv(self, lib, argv_list):
        argv_buffers = []
        argv = (c_char_p * len(argv_list))()
        for i, s in enumerate(argv_list):
            buf = create_string_buffer(s.encode("utf-8"))
            argv_buffers.append(buf)
            argv[i] = cast(buf, c_char_p)
        c_int.in_dll(lib, "g_argc").value = len(argv_list)
        c_void_p.in_dll(lib, "g_argv").value = cast(argv, c_void_p).value
        return argv_buffers, argv 

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
        self._set_g_argv(self.lib, ["prince", "megahit"])
        ctypes.c_int.in_dll(self.lib, "RL_Mode").value = 1
        ctypes.c_int.in_dll(self.lib, "RL_Visual").value = 1 if self.visual else 0
        ctypes.c_short.in_dll(self.lib, "start_level").value = 1
        
        NUM_CHANNELS, GRID_H, GRID_W = 20, 3, 10
        STATE_DIM  = 9
        ACTION_COUNT = 18
        act_hist   = 5

        self.grid        = np.zeros((NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        self.state      = np.zeros(STATE_DIM, dtype=np.float32)
        self.action_history  = np.zeros(act_hist, dtype=np.int32)

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Dict({
            "grid":  spaces.Box(low=0.0, high=1.0, shape=(NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32),
            "action_history": spaces.Box(low=0, high=ACTION_COUNT - 1, shape=(act_hist,), dtype=np.int32),
        })

        self.step_count  = 0
        self.max_steps   = 100_000
        self.initialized = False
        
        self.room_xs = np.zeros(24, dtype=np.uint8)
        self.room_ys = np.zeros(24, dtype=np.uint8)
        self.guard_permanently_dead = False
        self.episode_rooms = set()

        # ── Precomputed lookup tables for vectorized create_grid ──
        self._walkable_lut = np.zeros(32, dtype=np.float32)
        for t in WALKABLE_TYPES:
            self._walkable_lut[t] = 1.0

        self._wall_lut = np.zeros(32, dtype=np.float32)
        for t in WALL_TYPES:
            self._wall_lut[t] = 1.0

        self._button_lut = np.zeros(32, dtype=np.float32)
        for t, v in BUTTON_MAP.items():
            self._button_lut[t] = v / 3.0

        self._potion_lut = np.zeros(8, dtype=np.float32)
        for k, v in POTION_MAP.items():
            if k < 8:
                self._potion_lut[k] = v / 6.0

        self.lib.pop_main.argtypes    = []
        self.lib.pop_main.restype     = None
        self.lib.play_level_2.argtypes = []
        self.lib.play_level_2.restype  = ctypes.c_int

        self.lib.init_game.argtypes = [ctypes.c_int]
        self.lib.init_game.restype  = None

        self.is_restart_level = ctypes.c_int.in_dll(self.lib, "is_restart_level")
        self.prince_death_ptr = ctypes.c_int.in_dll(self.lib, "prince_death")

        # ── Cache ctypes pointers for _update_reward_state ──
        self._ptr_hitp_curr     = c_short.in_dll(self.lib, "hitp_curr")
        self._ptr_have_sword    = ctypes.c_int16.in_dll(self.lib, "have_sword")
        self._ptr_guardhp_curr  = ctypes.c_ushort.in_dll(self.lib, "guardhp_curr")
        self._ptr_guardhp_max   = ctypes.c_ushort.in_dll(self.lib, "guardhp_max")
        self._ptr_current_level = c_short.in_dll(self.lib, "current_level")

    def _build_roomlinks(self):
        self.roomlinks = []
        for r in range(24):
            b = 1952 + r * 4
            self.roomlinks.append({
                "left":  int(self._level_view[b]),
                "right": int(self._level_view[b + 1]),
                "up":    int(self._level_view[b + 2]),
                "down":  int(self._level_view[b + 3]),
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

    def _cache_ctypes_views(self):
        """Cache numpy views of ctypes buffers. Call once after first get_values()."""
        self._level_view = np.frombuffer(self.raw_level, dtype=np.uint8)
        self._kid_view   = np.frombuffer(self.kid_raw,   dtype=np.uint8)
        self._guard_view = np.frombuffer(self.guard_raw,  dtype=np.uint8)

    def _update_reward_state(self):
        """Light read — only the 5 values needed for reward shaping in the 4-frame loop."""
        self.hitp_curr     = self._ptr_hitp_curr.value
        self.have_sword    = self._ptr_have_sword.value
        self.guardhp_curr  = self._ptr_guardhp_curr.value
        self.guardhp_max   = self._ptr_guardhp_max.value
        self.current_level = self._ptr_current_level.value

    def get_values(self):
        """Full read — all game state for observation building. Call once per step."""
        self.hitp_curr = c_short.in_dll(self.lib, "hitp_curr").value
        self.hitp_max = c_short.in_dll(self.lib, "hitp_max").value
        self.current_level = c_short.in_dll(self.lib, "current_level").value

        self.action = c_int.in_dll(self.lib, "action").value
        self.RL_Mode = c_int.in_dll(self.lib, "RL_Mode").value
        self.RL_Visual = c_int.in_dll(self.lib, "RL_Visual").value
        self.start_level = c_short.in_dll(self.lib, "start_level").value
        
        self.raw_level = (ctypes.c_uint8 * 2305).in_dll(self.lib, "level")

        # Use cached view if available, else create
        if hasattr(self, '_level_view'):
            level_np = self._level_view
        else:
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

        # kid and guard Info
        self.kid_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Kid")
        self.guard_raw = (ctypes.c_uint8 * 16).in_dll(self.lib, "Guard")

        if hasattr(self, '_kid_view'):
            kid_np = self._kid_view
            guard_np = self._guard_view
        else:
            kid_np = np.frombuffer(self.kid_raw, dtype=np.uint8)
            guard_np = np.frombuffer(self.guard_raw, dtype=np.uint8)

        # kid's Info 
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

        # Guard's Info
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
        self.rem_min = ctypes.c_ushort.in_dll(self.lib, "rem_min").value
        self.rem_tick = ctypes.c_ushort.in_dll(self.lib, "rem_tick").value

    def create_grid(self, room=None):
        if room is None:
            room = self.kid_room
        if room < 1 or room > 24:
            return np.zeros((20, 3, 10), dtype=np.float32)

        room_offset = (room - 1) * 30
        room_fg = self.fg[room_offset : room_offset + 30]
        room_bg = self.bg[room_offset : room_offset + 30]

        # ── Vectorized tile processing (replaces two Python loops) ──
        t_arr = (room_fg.astype(np.uint8) & 0x1F)
        m_arr = room_bg.astype(np.uint8)

        # State arrays
        gate_raw  = np.where(t_arr == T_GATE,
                     np.where(m_arr == 0, 1, np.where(m_arr >= 188, 3, 2)),
                     0).astype(np.float32)
        loose_raw = np.where(t_arr == T_LOOSE,
                     np.where(m_arr != 0, 2, 1),
                     0).astype(np.float32)
        chomp_raw = np.where(t_arr == T_CHOMPER,
                     (m_arr & 0x7F).astype(np.float32),
                     0.0)

        # Reshape to grid (3×10)
        t = t_arr.reshape(3, 10)
        m = m_arr.reshape(3, 10)
        gate  = gate_raw.reshape(3, 10)
        loose = loose_raw.reshape(3, 10)
        chomp = chomp_raw.reshape(3, 10)

        # Derived masks
        gate_open   = (t == T_GATE) & (gate >= 2)
        gate_closed = (t == T_GATE) & (gate < 2)
        chomp_open  = (t == T_CHOMPER) & (chomp == 0)

        grid = self.grid
        grid[:] = 0

        # Fill channels vectorized
        grid[CH_TILE_TYPE]   = t / 30.0
        grid[CH_WALKABLE]    = np.where(gate_open, 1.0,
                                np.where(gate_closed, 0.0,
                                 np.where(chomp_open, 1.0, self._walkable_lut[t])))
        grid[CH_WALL]        = np.where(gate_closed, 1.0, self._wall_lut[t])
        grid[CH_FLOOR_EDGE]  = np.where((_ROW_IDX == 2) & (t == T_EMPTY), 1.0, 0.0)
        grid[CH_LOOSE_STATE] = loose / 2.0
        grid[CH_GATE]        = gate / 3.0
        grid[CH_BUTTON]      = self._button_lut[t]
        grid[CH_CHOMPER]     = chomp / 7.0

        # Items
        item = np.zeros((3, 10), dtype=np.float32)
        item = np.where(t == T_POTION, self._potion_lut[(m >> 3) & 0x7], item)
        item = np.where(t == T_SWORD, 5.0 / 6.0, item)
        item = np.where((t == T_EXIT_LEFT) | (t == T_EXIT_RIGHT), 1.0, item)
        grid[CH_ITEM] = item

        # ── Neighbor room border tiles (left/right edges) ──
        room_idx = room - 1
        if 0 <= room_idx < 24:
            links = self.roomlinks[room_idx]
            for nb_room, grid_col, nb_col in [
                (links["left"],  0, 9),
                (links["right"], 9, 0),
            ]:
                if nb_room < 1 or nb_room > 24:
                    continue
                nb_offset = (nb_room - 1) * 30
                for row in range(ROWS):
                    nb_idx = nb_offset + row * COLS + nb_col
                    bt = int(self.fg[nb_idx]) & 0x1F
                    bm = int(self.bg[nb_idx])

                    if bt == T_GATE:
                        gs = 1.0 if bm == 0 else (3.0 if bm >= 188 else 2.0)
                        grid[CH_GATE, row, grid_col] = max(grid[CH_GATE, row, grid_col], gs / 3.0)
                        if gs < 2:
                            grid[CH_WALL, row, grid_col] = 1.0
                            grid[CH_WALKABLE, row, grid_col] = 0.0
                        else:
                            grid[CH_WALKABLE, row, grid_col] = 1.0

                    if bt in WALL_TYPES and grid[CH_WALKABLE, row, grid_col] == 0:
                        grid[CH_WALL, row, grid_col] = 1.0

                    if bt == T_CHOMPER:
                        phase = bm & 0x7F
                        grid[CH_CHOMPER, row, grid_col] = max(grid[CH_CHOMPER, row, grid_col], float(phase) / 7.0)

                    if bt == T_POTION:
                        grid[CH_ITEM, row, grid_col] = POTION_MAP.get(bm >> 3, 1) / 6.0
                    elif bt == T_SWORD:
                        grid[CH_ITEM, row, grid_col] = 5.0 / 6.0
                    elif bt in (T_EXIT_LEFT, T_EXIT_RIGHT):
                        grid[CH_ITEM, row, grid_col] = 6.0 / 6.0

        # ── Player & enemy overlays ──
        kid_col   = max(0, min(9, self.kid_curr_col))
        kid_row   = max(0, min(2, self.kid_curr_row))
        guard_col = max(0, min(9, self.guard_curr_col))
        guard_row = max(0, min(2, self.guard_curr_row))

        grid[CH_PLAYER_POS, kid_row, kid_col] = 1.0
        grid[CH_PLAYER_ACT, kid_row, kid_col] = min(float(self.kid_action), 7.0) / 7.0
        grid[CH_PLAYER_DIR, kid_row, kid_col] = 1.0 if self.kid_direction < 0 else 0.0
        grid[CH_PLAYER_SWORD, kid_row, kid_col] = float(min(self.kid_sword, 2) + 1) / 3.0
        grid[CH_PLAYER_FRAME, kid_row, kid_col] = self.kid_frame / 255.0

        if self.guard_room == self.kid_room and self.guard_alive < 0:
            grid[CH_ENEMY_POS,   guard_row, guard_col] = float(min(4, max(1, self.guardhp_curr))) / 4.0
            grid[CH_ENEMY_DIR,   guard_row, guard_col] = (1.0 if self.guard_direction < 0 else 2.0) / 2.0
            grid[CH_ENEMY_ACT,   guard_row, guard_col] = min(float(self.guard_action), 7.0) / 7.0
            grid[CH_ENEMY_SWORD, guard_row, guard_col] = float(min(self.guard_sword, 2) + 1) / 3.0
            grid[CH_ENEMY_TYPE,  guard_row, guard_col] = float(min(self.guard_charid, 4) + 1) / 5.0
            grid[CH_ENEMY_FRAME, guard_row, guard_col] = self.guard_frame / 255.0

        return grid

    def get_obs_state(self):
        hp_norm      = self.hitp_curr / max(self.hitp_max, 1.0)
        hp_max_norm  = self.hitp_max  / 10.0
        level_norm   = self.current_level / 14.0
        fall_y_norm  = np.clip(self.kid_fall_y / 10.0, -1.0, 1.0)
        death        = 1.0 if self.prince_death_ptr.value == 1 else 0.0
        have_sword   = 1.0 if self.have_sword != 0 else 0.0

        room_idx = max(0, min(self.kid_room - 1, 23))
        bx = int(self.room_xs[room_idx]) if self.room_xs[room_idx] != 255 else 0
        by = int(self.room_ys[room_idx]) if self.room_ys[room_idx] != 255 else 0
        col = max(0, min(9, self.kid_curr_col))
        row = max(0, min(2, self.kid_curr_row))
        gx = (bx * 10) + col
        gy = (by * 3)  + row

        self.state[0] = hp_norm
        self.state[1] = hp_max_norm
        self.state[2] = level_norm
        self.state[3] = fall_y_norm
        self.state[4] = death
        self.state[5] = have_sword
        self.state[6] = self.kid_room / 24.0
        self.state[7] = gx / 250.0
        self.state[8] = gy / 100.0

        return self.state

    def _get_obs(self):
        return {
            "grid": self.create_grid().copy(),
            "state": self.get_obs_state().copy(),
            "action_history": self.action_history.copy(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.initialized:
            self.lib.pop_main()
            self.initialized = True
        else:
            self.lib.init_game(self.current_level)

        for _ in range(25):
            self.lib.play_level_2()

        self.get_values()
        self._cache_ctypes_views()   # cache numpy views after first get_values
        self._load_room_coords()
        self._build_roomlinks()
        self.prince_death_ptr.value = 0

        self.step_count    = 0
        self.action_history[:] = 0
        self.guard_permanently_dead = False
        self.episode_rooms = set()
        self.episode_rooms.add((self.current_level, self.kid_room))
        
        self.prev_level = self.current_level

        info = {
            "level": self.current_level,
            "room": self.kid_room,
            "hp": self.hitp_curr,
            "steps_alive": self.step_count,
            "deaths": 1 if self.prince_death_ptr.value == 1 else 0,
        }
        return self._get_obs(), info

    def apply_action(self, action_idx):
        ctypes.c_int.in_dll(self.lib, "action").value = action_idx

    def step(self, action):
        self.apply_action(action)
        
        reward = 0.0
        prev_hp         = self.hitp_curr
        prev_have_sword = self.have_sword
        prev_guardhp    = self.guardhp_curr
        terminated = False

        for _ in range(4):
            self.lib.play_level_2()
            self._update_reward_state()   # light read (5 values)
            self.step_count += 1

            # ── Step penalty: 0.002 per game frame (4 frames/step = 0.008/step) ──
            reward -= 0.002

            alive = self.prince_death_ptr.value != 1

            # ── HP change: ±0.5 per HP bar (only while alive) ──
            if alive and self.hitp_curr != prev_hp:
                reward += 0.5 * (self.hitp_curr - prev_hp)
                prev_hp = self.hitp_curr

            if self.have_sword != 0 and prev_have_sword == 0:
                reward += 10.0
                prev_have_sword = self.have_sword
            if self.guardhp_curr == 0 and prev_guardhp > 0:
                reward += 50.0
                prev_guardhp = 0
            if self.current_level > getattr(self, "prev_level", self.current_level):
                reward += 50.0
                self.get_values()   # full read needed for level transition
                self._build_roomlinks()
                self._load_room_coords()
                self.prev_level = self.current_level
                self.guard_permanently_dead = False

            if self.guardhp_curr == 0 and self.guardhp_max > 0:
                self.guard_permanently_dead = True

            terminated = not alive
            if terminated:
                break

        # Full read once for observation building
        self.get_values()

        # ── Episodic frontier reward ─────────────────────────────
        frontier_connections = 0
        if not terminated:
            room_key = (int(self.current_level), int(self.kid_room))

            if room_key not in self.episode_rooms:
                self.episode_rooms.add(room_key)

                # Count neighbors not yet visited THIS episode
                if 1 <= self.kid_room <= 24 and hasattr(self, "roomlinks"):
                    links = self.roomlinks[self.kid_room - 1]
                    for nb in [links["left"], links["right"], links["up"], links["down"]]:
                        if 1 <= nb <= 24:
                            if (int(self.current_level), nb) not in self.episode_rooms:
                                frontier_connections += 1

            reward += 5.0 * frontier_connections
        else:
            reward -= 5.0

        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = action

        truncated = self.step_count >= self.max_steps

        info = {
            "level":                self.current_level,
            "room":                 self.kid_room,
            "hp":                   self.hitp_curr,
            "steps_alive":          self.step_count,
            "deaths":               1 if self.prince_death_ptr.value == 1 else 0,
            "frontier_connections": frontier_connections,
            "episode_rooms":        len(self.episode_rooms),
        }
        return self._get_obs(), reward, terminated, truncated, info

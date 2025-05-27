import ctypes
from ctypes import c_int, c_short, c_int8, c_int16, c_uint8, c_ushort
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

T_EMPTY=0; T_FLOOR=1; T_SPIKES=2; T_PILLAR=3; T_GATE=4
T_STUCK_BUTTON=5; T_DROP_BUTTON=6; T_TAPESTRY=7
T_BIGPILLAR_BOT=8; T_BIGPILLAR_TOP=9; T_POTION=10; T_LOOSE=11
T_DOORTOP=12; T_MIRROR=13; T_DEBRIS=14; T_RAISE_BUTTON=15
T_EXIT_LEFT=16; T_EXIT_RIGHT=17; T_CHOMPER=18; T_TORCH=19
T_WALL=20; T_SKELETON=21; T_SWORD=22

BUTTON_MAP = {T_STUCK_BUTTON: 1, T_DROP_BUTTON: 2, T_RAISE_BUTTON: 3}
POTION_MAP = {0: 1, 1: 2, 2: 4, 3: 5, 4: 3}
ALIVE = -1

# human-recorded directed traversal graph for level 1
# one-way falls are directed edges; unreachable rooms have no entry
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

LEVEL1_SWORD_ROOM  = 15
LEVEL1_EXIT_ROOM   = 9
LEVEL1_GUARD_ROOMS = {3}

SUBGOAL_NAVIGATE  = 0
SUBGOAL_GET_SWORD = 1
SUBGOAL_KILL      = 2
SUBGOAL_GET_HP    = 3
SUBGOAL_LEVEL_UP  = 4
N_SUBGOALS        = 5

CH_TILE_TYPE=0;   CH_LOOSE_STATE=1; CH_GATE=2;       CH_BUTTON=3
CH_CHOMPER=4;     CH_SPIKE_STATE=5; CH_ITEM=6;       CH_POTION_HARM=7
CH_KID_PRES=8;    CH_KID_ACTION=9;  CH_KID_FRAME=10
CH_KID_FACING=11; CH_KID_FALL=12;   CH_KID_SWORD=13
CH_GUARD_PRES=14; CH_GUARD_ACTION=15; CH_GUARD_FRAME=16
CH_GUARD_HP=17;   CH_GUARD_FACING=18

NUM_CH = 19; ROWS = 3; COLS = 10

GRID_DIM    = NUM_CH * ROWS * COLS   # 570
STATE_DIM   = 10
GOAL_DIM    = 24
SUBGOAL_DIM = N_SUBGOALS             # 5
ARRIVED_DIM = 1
OBS_DIM     = GRID_DIM + STATE_DIM + GOAL_DIM + SUBGOAL_DIM + ARRIVED_DIM  # 610

_GR = slice(0,         GRID_DIM)
_ST = slice(_GR.stop,  _GR.stop + STATE_DIM)
_GL = slice(_ST.stop,  _ST.stop + GOAL_DIM)
_SG = slice(_GL.stop,  _GL.stop + SUBGOAL_DIM)
_AR = slice(_SG.stop,  _SG.stop + ARRIVED_DIM)

N_ACTIONS = 18

R_DEATH        = -10.0
R_TIME_TAX     =  -0.01
R_NAVIGATE     =  30.0
R_GET_SWORD    = 100.0
R_KILL         = 100.0
R_GET_HP       =  50.0
R_LEVEL_UP     = 150.0
R_STEP_CLOSER  =   0.2
R_NEW_ROOM     =   1.0   # phase-2 room novelty bonus
R_ARRIVE       =   0.0   # added to goal-specific rewards
R_WRONG_ROOM_PENALTY = -0.1

# per-env step thresholds for curiosity phase transitions
CURIOSITY_TILE_STEPS  = 50_000
CURIOSITY_ROOM_STEPS  = 200_000


class PoPEnv(gym.Env):

    def __init__(self, visual=False):
        super().__init__()
        self.SDLPoP_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP")
        os.chdir(self.SDLPoP_path)
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if not visual:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_RENDER_DRIVER"] = "software"

        self.lib = ctypes.CDLL(os.path.join(self.SDLPoP_path, "src", "libSDLPoP.so"))
        c_int.in_dll(self.lib, "rl_mode").value        = 1
        c_int.in_dll(self.lib, "rl_visual_mode").value = int(visual)
        c_short.in_dll(self.lib, "start_level").value  = 1

        self.lib.pop_main.argtypes     = []; self.lib.pop_main.restype = None
        self.lib.play_level_2.argtypes = []; self.lib.play_level_2.restype = c_int
        self.lib.init_game.argtypes    = [c_int]; self.lib.init_game.restype = None

        self.rl_action_ptr   = c_int.in_dll(self.lib, "rl_action")
        self.rl_kid_dead_ptr = c_int.in_dll(self.lib, "rl_kid_dead")

        self.grid         = np.zeros((NUM_CH, ROWS, COLS), dtype=np.float32)
        self.state        = np.zeros(STATE_DIM,   dtype=np.float32)
        self._goal_buf    = np.zeros(GOAL_DIM,    dtype=np.float32)
        self._subgoal_buf = np.zeros(SUBGOAL_DIM, dtype=np.float32)
        self.obs_buf      = np.zeros(OBS_DIM,     dtype=np.float32)

        self.target_room           = 0
        self.subgoal               = SUBGOAL_NAVIGATE
        self.step_count            = 0
        self.max_steps             = 30_000
        self._prev_kid_room        = 0
        self.micro_visited_tiles   = set()
        self.visited_rooms         = set()    # phase-2: cleared per subgoal
        self.total_steps           = 0        # never reset — training progress clock
        self._subgoal_steps        = 0
        self._ever_had_sword       = False
        self._last_arrived         = 0.0

        self._hp_at_subgoal_start    = 0
        self._level_at_subgoal_start = 0
        self._base_seed              = 0
        self._reset_count            = 0

        self._button_lut = np.zeros(32, dtype=np.float32)
        for t, v in BUTTON_MAP.items():
            self._button_lut[t] = v / 3.0
        self._potion_lut = np.zeros(8, dtype=np.float32)
        for k, v in POTION_MAP.items():
            self._potion_lut[k] = v / 7.0

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.initialized = False

    def get_values(self):
        lib = self.lib
        self.hitp_curr     = c_short.in_dll(lib, "hitp_curr").value
        self.hitp_max      = c_short.in_dll(lib, "hitp_max").value
        self.current_level = c_short.in_dll(lib, "current_level").value
        self.have_sword    = c_int16.in_dll(lib, "have_sword").value

        raw = (c_uint8 * 2305).in_dll(lib, "level")
        lv  = np.frombuffer(raw, dtype=np.uint8)
        self.fg         = lv[:720]
        self.bg         = lv[720:1440]
        self.doorlinks2 = lv[1696:1952]   # 256-byte doorlink timer array
        self.level_np   = lv

        kid = np.frombuffer((c_uint8 * 16).in_dll(lib, "Kid"), dtype=np.uint8)
        self.kid_frame     = int(kid[0])
        self.kid_x         = int(kid[1])
        self.kid_y         = int(kid[2])
        self.kid_direction = c_int8(kid[3]).value
        self.kid_curr_col  = c_int8(kid[4]).value
        self.kid_curr_row  = c_int8(kid[5]).value
        self.kid_action    = int(kid[6])
        self.kid_fall_x    = c_int8(kid[7]).value
        self.kid_fall_y    = c_int8(kid[8]).value
        self.kid_room      = int(kid[9])
        self.kid_sword     = int(kid[12])
        self.kid_curr_seq  = int(kid[14]) | (int(kid[15]) << 8)

        g = np.frombuffer((c_uint8 * 16).in_dll(lib, "Guard"), dtype=np.uint8)
        self.guard_frame     = int(g[0])
        self.guard_direction = c_int8(g[3]).value
        self.guard_curr_col  = c_int8(g[4]).value
        self.guard_curr_row  = c_int8(g[5]).value
        self.guard_action    = int(g[6])
        self.guard_room      = int(g[9])
        self.guard_alive     = c_int8(g[13]).value
        self.guard_curr_seq  = int(g[14]) | (int(g[15]) << 8)
        self.guardhp_curr    = c_ushort.in_dll(lib, "guardhp_curr").value
        self.guardhp_max     = c_ushort.in_dll(lib, "guardhp_max").value

    # roomlink direction indices
    _RL_LEFT = 0; _RL_RIGHT = 1; _RL_UP = 2; _RL_DOWN = 3

    def _build_roomlinks(self):
        if not hasattr(self, "level_np"):
            raise RuntimeError("_build_roomlinks called before get_values()")
        self.roomlinks = self.level_np[1952:1952 + 96].astype(np.int32).reshape(24, 4)

    def _rebuild_tile_channels(self, room):
        grid = self.grid
        if room < 1 or room > 24:
            grid[:] = 0.0
            return
            
        # Extremely fast cache guard based on underlying bytes hash to avoid looping arrays and bounds checks when room is truly static
        room_offset = (room - 1) * 30
        room_fg = self.fg[room_offset : room_offset + 30]
        room_bg = self.bg[room_offset : room_offset + 30]
        door_links = self.doorlinks2
        cur_hash = hash(room_fg.tobytes() + room_bg.tobytes() + door_links.tobytes())
        if getattr(self, '_cached_grid_room', -1) == room and getattr(self, '_cached_grid_hash', None) == cur_hash:
            return
        
        self._cached_grid_room = room
        self._cached_grid_hash = cur_hash
        
        grid[:] = 0

        off = (room - 1) * 30
        fg = self.fg; bg = self.bg; dl2 = self.doorlinks2
        blut = self._button_lut; plut = self._potion_lut

        # single pass — fills all tile channels at once
        for i in range(30):
            r = i // 10; c = i % 10
            bt = int(fg[off + i]) & 0x1F
            bm = int(bg[off + i])
            grid[CH_TILE_TYPE, r, c] = bt * 0.033333333  # bt / 30
            if bt == T_LOOSE:
                grid[CH_LOOSE_STATE, r, c] = (
                    0.333333333 if bm == 0 else        # 1/3 normal
                    0.666666667 if bm & 0x80 else      # 2/3 shaking
                    1.0)                                # 3/3 about-to-fall
            elif bt == T_GATE:
                grid[CH_GATE, r, c] = (
                    0.333333333 if bm == 0 else        # closed
                    1.0 if bm >= 188 else               # open
                    0.666666667)                         # partial
            elif bt == T_DROP_BUTTON or bt == T_RAISE_BUTTON:
                timer = dl2[bm] & 0x1F
                grid[CH_BUTTON, r, c] = -blut[bt] if timer > 1 else blut[bt]
            elif bt == T_STUCK_BUTTON:
                grid[CH_BUTTON, r, c] = blut[bt]
            elif bt == T_CHOMPER:
                grid[CH_CHOMPER, r, c] = (bm & 0x7F) / 127.0
            elif bt == T_SPIKES:
                grid[CH_SPIKE_STATE, r, c] = (bm & 0x0F) / 15.0
            elif bt == T_POTION:
                ptype = (bm >> 3) & 0x7
                grid[CH_ITEM, r, c] = plut[ptype]
                if ptype == 3:
                    grid[CH_POTION_HARM, r, c] = 1.0
            elif bt == T_SWORD:
                grid[CH_ITEM, r, c] = 0.857142857  # 6/7
            elif bt == T_EXIT_LEFT or bt == T_EXIT_RIGHT:
                grid[CH_ITEM, r, c] = 1.0

        ri = room - 1
        if 0 <= ri < 24 and hasattr(self, "roomlinks"):
            lk = self.roomlinks[ri]
            for nb, gc, nc, gr, nr in [
                (lk[self._RL_LEFT],  0,    9,    None, None),
                (lk[self._RL_RIGHT], 9,    0,    None, None),
                (lk[self._RL_UP],    None, None, 0,    2   ),
                (lk[self._RL_DOWN],  None, None, 2,    0   ),
            ]:
                if nb < 1 or nb > 24: continue
                noff = (nb - 1) * 30
                for r in (range(ROWS) if gr is None else [gr]):
                    for c in (range(COLS) if gc is None else [gc]):
                        nr2 = r if nr is None else nr
                        nc2 = c if nc is None else nc
                        idx = noff + nr2 * COLS + nc2
                        bt  = int(self.fg[idx]) & 0x1F
                        bm  = int(self.bg[idx])
                        if bt == T_GATE:
                            gs = 1.0 if bm == 0 else (3.0 if bm >= 188 else 2.0)
                            grid[CH_GATE, r, c] = max(grid[CH_GATE, r, c], gs / 3)
                        if bt == T_CHOMPER:
                            grid[CH_CHOMPER, r, c] = max(grid[CH_CHOMPER, r, c], (bm & 0x7F) / 127)
                        if bt == T_SPIKES:
                            grid[CH_SPIKE_STATE, r, c] = max(grid[CH_SPIKE_STATE, r, c], (bm & 0xF) / 15)
                        if bt == T_POTION:
                            ptype = (bm >> 3) & 0x7
                            grid[CH_ITEM, r, c] = self._potion_lut[ptype] if ptype < 8 else 0.0
                            if ptype == 3:
                                grid[CH_POTION_HARM, r, c] = 1.0
                        elif bt == T_SWORD:
                            grid[CH_ITEM, r, c] = 6 / 7
                        elif bt in (T_EXIT_LEFT, T_EXIT_RIGHT):
                            grid[CH_ITEM, r, c] = 1.0

    def _add_agent_channels(self):
        grid       = self.grid
        for ch in (CH_KID_PRES, CH_KID_ACTION, CH_KID_FRAME, CH_KID_FALL, CH_KID_SWORD,
                   CH_KID_FACING,
                   CH_GUARD_PRES, CH_GUARD_ACTION, CH_GUARD_FRAME, CH_GUARD_HP, CH_GUARD_FACING):
            grid[ch] = 0.0

        guard_here = (
            self.guard_room  == self.kid_room
            and self.guardhp_max  > 0
            and self.guard_alive == ALIVE
        )

        if 1 <= self.kid_room <= 24:
            r = int(np.clip(self.kid_curr_row, 0, 2))
            c = int(np.clip(self.kid_curr_col, 0, 9))
            grid[CH_KID_PRES,   r, c] = 1.0
            grid[CH_KID_ACTION, r, c] = min(self.kid_action / 128.0, 1.0)
            grid[CH_KID_FRAME,  r, c] = self.kid_frame  / 255.0
            grid[CH_KID_FALL,   r, c] = float(np.clip(self.kid_fall_y / 30.0, -1, 1))
            grid[CH_KID_SWORD,  r, c] = self.kid_sword  / 2.0
        grid[CH_KID_FACING, :, :] = 1.0 if self.kid_direction >= 0 else 0.0

        if guard_here:
            gr = int(np.clip(self.guard_curr_row, 0, 2))
            gc = int(np.clip(self.guard_curr_col, 0, 9))
            grid[CH_GUARD_PRES,   gr, gc] = 1.0
            grid[CH_GUARD_ACTION, gr, gc] = min(self.guard_action / 128.0, 1.0)
            grid[CH_GUARD_FRAME,  gr, gc] = self.guard_frame  / 255.0
            grid[CH_GUARD_HP,     gr, gc] = self.guardhp_curr / max(self.guardhp_max, 1)
            grid[CH_GUARD_FACING, :,  :] = 1.0 if self.guard_direction >= 0 else 0.0

    def _fill_state(self):
        s = self.state
        s[0] = (self.hitp_curr / 10.0) * 2.0 - 1.0
        s[1] = (self.hitp_max / 10.0) * 2.0 - 1.0
        s[2] = (self.current_level / 15.0) * 2.0 - 1.0
        s[3] = float(np.clip(self.kid_fall_y  / 30.0, -1, 1))
        s[4] = float(np.clip(self.kid_fall_x  / 10.0, -1, 1))
        s[5] = 1.0 if self.have_sword else -1.0
        if self.kid_room < 1 or self.kid_room > 24:
            s[6:] = 0.0
            return
            
        kid_x_norm = (self.kid_x / 320.0) * 2.0 - 1.0
        kid_y_norm = (self.kid_y / 200.0) * 2.0 - 1.0
        s[6] = kid_x_norm
        s[7] = kid_y_norm
        s[8] = (min(self.kid_curr_seq,   200) / 200.0) * 2.0 - 1.0
        s[9] = (min(self.guard_curr_seq, 200) / 200.0) * 2.0 - 1.0

    def _fill_goal(self):
        g = self._goal_buf; g[:] = 0.0
        if 1 <= self.target_room <= 24: g[self.target_room - 1] = 1.0

    def _fill_subgoal(self):
        s = self._subgoal_buf; s[:] = 0.0
        if 0 <= self.subgoal < N_SUBGOALS: s[self.subgoal] = 1.0

    def assign_goal(self, target_room, subgoal):
        self.target_room = target_room
        self.subgoal     = subgoal
        self._hp_at_subgoal_start    = self.hitp_curr
        self._level_at_subgoal_start = self.current_level
        self.micro_visited_tiles     = set()
        self.visited_rooms           = set()
        self._subgoal_steps          = 0

    def _check_arrived(self):
        sg = self.subgoal
        if sg == SUBGOAL_NAVIGATE:
            return float(self.kid_room == self.target_room)
        if sg == SUBGOAL_GET_SWORD:
            return float(self.have_sword > 0)
        if sg == SUBGOAL_KILL:
            return float(
                self.kid_room   == self.target_room
                and self.guard_room  == self.target_room
                and self.guard_alive != ALIVE
                and self.guardhp_max  > 0
            )
        if sg == SUBGOAL_GET_HP:
            # If player takes damage between assignment and arrival, original target HP becomes unreachable.
            # We lower the bar dynamically so any HP gain from current damaged state satisfies it, preventing softlogs.
            return float(self.kid_room == self.target_room and self.hitp_curr > self._hp_at_subgoal_start)
        if sg == SUBGOAL_LEVEL_UP:
            return float(self.current_level > self._level_at_subgoal_start)
        return 0.0

    def _get_obs(self):
        # Always rebuild since dynamic tile states (gates, etc.) change mid-room
        self._rebuild_tile_channels(self.kid_room)
        self._add_agent_channels()

        self._fill_state()
        self._fill_goal()
        self._fill_subgoal()
        self.obs_buf[_GR] = self.grid.ravel()
        self.obs_buf[_ST] = self.state
        self.obs_buf[_GL] = self._goal_buf
        self.obs_buf[_SG] = self._subgoal_buf
        self.obs_buf[_AR] = self._last_arrived
        return self.obs_buf.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self._base_seed = seed
        self._reset_count += 1
        self._np_rng = np.random.default_rng(seed if seed is not None else self._base_seed + self._reset_count)

        if not self.initialized:
            self.lib.pop_main()
            self.initialized = True
        else:
            self.lib.init_game(1)
        self.get_values()
        self._build_roomlinks()
        self.rl_kid_dead_ptr.value = 0
        self.step_count = 0
        self._ever_had_sword = False
        self._prev_kid_room = self.kid_room  # reset before _sample_next_goal to avoid cross-episode bleed
        first_room, first_subgoal = self._sample_next_goal()
        self.assign_goal(first_room, first_subgoal)
        self._last_arrived = 0.0
        obs = self._get_obs()
        return obs, {"level": self.current_level, "room": self.kid_room, "hp": self.hitp_curr}

    def _sample_next_goal(self):
        neighbors = LEVEL1_GRAPH.get(self.kid_room, [])
        if not neighbors:
            return self.kid_room, SUBGOAL_NAVIGATE
        # prevent backtracking until the sword has been obtained at least once
        if not self._ever_had_sword and self._prev_kid_room in neighbors and len(neighbors) > 1:
            neighbors = [r for r in neighbors if r != self._prev_kid_room]
        target_room = int(self._np_rng.choice(neighbors))
        if target_room == LEVEL1_SWORD_ROOM and not self.have_sword:
            return target_room, SUBGOAL_GET_SWORD
        # note: guard respawns after lib.init_game on reset, get_values picks it up before this
        if target_room in LEVEL1_GUARD_ROOMS and self.have_sword and self.guard_alive == ALIVE:
            return target_room, SUBGOAL_KILL
        if target_room == LEVEL1_EXIT_ROOM and self.have_sword:
            # Note: After arriving in Room 9 and level change, logic might loop agent back to Room 3.
            # Room 3's guard re-triggers SUBGOAL_KILL until dead again.
            if self.current_level < 14: return target_room, SUBGOAL_LEVEL_UP
        off   = (target_room - 1) * 30
        t_arr = (self.fg[off:off+30].astype(np.uint8) & 0x1F)
        if T_POTION in t_arr and self.hitp_curr < self.hitp_max and self._np_rng.random() > 0.5:
            return target_room, SUBGOAL_GET_HP
        return target_room, SUBGOAL_NAVIGATE

    def step(self, action):
        self._last_arrived = 0.0
        self.rl_action_ptr.value = int(action)

        for _ in range(4):
            self.lib.play_level_2()
            if self.rl_kid_dead_ptr.value == 1:
                break

        self.step_count     += 1
        self._subgoal_steps += 1
        self.total_steps    += 1

        prev_level = self.current_level
        self.get_values()
        if self.have_sword and not self._ever_had_sword:
            self._ever_had_sword = True
        if self.current_level != prev_level:
            self._build_roomlinks()

        # Reset tile set on room change so every room entry gives fresh bonuses
        if self.kid_room != self._prev_kid_room:
            self.micro_visited_tiles = set()

        alive   = (self.rl_kid_dead_ptr.value != 1)
        if alive and self.subgoal == SUBGOAL_GET_HP and self.hitp_curr < self._hp_at_subgoal_start:
            self._hp_at_subgoal_start = self.hitp_curr
            
        arrived = self._check_arrived() == 1.0
        # self._last_arrived = float(arrived) # Moved below to prevent premature overwrite

        info = {
            "room":         self.kid_room,
            "target_room":  self.target_room,
            "subgoal":      self.subgoal,
            "hp":           self.hitp_curr,
            "have_sword":   self.have_sword,
            "goal_event":   False,
            "goal_success": False,
            "goal_subgoal": self.subgoal,
            "goal_steps":   self._subgoal_steps,
        }

        is_wrong_room = (self.kid_room != self.target_room)

        if arrived and alive:
            self._last_arrived = 1.0
            payout = {
                SUBGOAL_NAVIGATE:  R_NAVIGATE,
                SUBGOAL_GET_SWORD: R_GET_SWORD,
                SUBGOAL_KILL:      R_KILL,
                SUBGOAL_GET_HP:    R_GET_HP,
                SUBGOAL_LEVEL_UP:  R_LEVEL_UP,
            }.get(self.subgoal, R_NAVIGATE)
            reward = payout + R_TIME_TAX + R_ARRIVE
            info["goal_event"]   = True
            info["goal_success"] = True
            
            # Subgoal specific handling...
            if self.subgoal == SUBGOAL_GET_HP:
                self._hp_at_subgoal_start = self.hitp_curr  # Ensure newly gained HP applies permanently
            
            next_room, next_subgoal = self._sample_next_goal()
            self.assign_goal(next_room, next_subgoal)
            
        elif not alive:
            reward = R_DEATH
            info["goal_event"]   = True
            info["goal_success"] = False

        else:
            reward = R_TIME_TAX

            if self.total_steps < CURIOSITY_TILE_STEPS:
                # Phase 1 — tile curiosity
                # (col, row, room) fixes aliasing: same grid coord in
                # different rooms is now a distinct state
                tile_state = (self.kid_curr_col, self.kid_curr_row, self.kid_room)
                if tile_state not in self.micro_visited_tiles:
                    self.micro_visited_tiles.add(tile_state)
                    reward += R_STEP_CLOSER

            elif self.total_steps < CURIOSITY_ROOM_STEPS:
                # Phase 2 — room novelty only
                # visited_rooms cleared on each new subgoal so it never
                # permanently exhausts; max 24 bonuses per subgoal
                if self.kid_room not in self.visited_rooms:
                    self.visited_rooms.add(self.kid_room)
                    reward += R_NEW_ROOM

            # Phase 3 — no curiosity, pure time tax + goal rewards

            # Wrong-room penalty only after curiosity phases end.
            if self.total_steps >= CURIOSITY_ROOM_STEPS:
                if self.kid_room != self._prev_kid_room and is_wrong_room:
                    reward -= 1.0

        self._prev_kid_room = self.kid_room

        obs = self._get_obs()
        return obs, reward, not alive, self.step_count >= self.max_steps, info
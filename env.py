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

CH_TILE=0; CH_GATE=1; CH_LOOSE=2; CH_PLATE=3
CH_POTION=4; CH_CHOMPER=5; CH_KID=6; CH_GUARD=7

NUM_CH = 8
GROWS = 5
GCOLS = 12

ACT_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 99:7}
N_ACT = LEVEL1_GRAPH = {
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
8
N_SW = 4       # 0=sheathed 1=drawing 2=drawn 3=putting-away
N_GTYPES = 7
G_SLOTS = 1      # Level 1 has one guard; second slot was always zeros

# kid: x y col row dir frame act(8) fx fy hp hpmax have_sw sw(4) room alive = 25
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
        self.SDLPoP_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP")
        os.chdir(self.SDLPoP_path)
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if not visual:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_RENDER_DRIVER"] = "software"

        self.lib = ctypes.CDLL(os.path.join(self.SDLPoP_path, "src", "libSDLPoP.so"))
        c_int.in_dll(self.lib, "rl_mode").value        = 1
        c_int.in_dll(self.lib, "rl_visual_mode").value  = int(visual)
        c_short.in_dll(self.lib, "start_level").value   = 1

        self.lib.pop_main.argtypes     = []; self.lib.pop_main.restype = None
        self.lib.play_level_2.argtypes = []; self.lib.play_level_2.restype = c_int
        self.lib.init_game.argtypes    = [c_int]; self.lib.init_game.restype = None

        self.rl_action   = c_int.in_dll(self.lib, "rl_action")
        self.rl_dead     = c_int.in_dll(self.lib, "rl_kid_dead")

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

    def get_values(self):
        lib = self.lib
        self.hp     = c_short.in_dll(lib, "hitp_curr").value
        self.hp_max = c_short.in_dll(lib, "hitp_max").value
        self.level  = c_short.in_dll(lib, "current_level").value
        self.have_sword = c_int16.in_dll(lib, "have_sword").value

        raw = (c_uint8 * 2305).in_dll(lib, "level")
        lv = np.frombuffer(raw, dtype=np.uint8)
        self.fg  = lv[:720]
        self.bg  = lv[720:1440]
        self.dl2 = lv[1696:1952]
        self.lv  = lv

        k = np.frombuffer((c_uint8 * 16).in_dll(lib, "Kid"), dtype=np.uint8)
        self.k_frame = int(k[0]); self.k_x = int(k[1]); self.k_y = int(k[2])
        self.k_dir   = c_int8(k[3]).value
        self.k_col   = c_int8(k[4]).value; self.k_row = c_int8(k[5]).value
        self.k_act   = int(k[6])
        self.k_fx    = c_int8(k[7]).value; self.k_fy = c_int8(k[8]).value
        self.k_room  = int(k[9]); self.k_charid = int(k[11])
        self.k_sword = int(k[12]); self.k_alive = c_int8(k[13]).value

        g = np.frombuffer((c_uint8 * 16).in_dll(lib, "Guard"), dtype=np.uint8)
        self.g_frame = int(g[0]); self.g_x = int(g[1]); self.g_y = int(g[2])
        self.g_dir   = c_int8(g[3]).value
        self.g_col   = c_int8(g[4]).value; self.g_row = c_int8(g[5]).value
        self.g_act   = int(g[6])
        self.g_fx    = c_int8(g[7]).value; self.g_fy = c_int8(g[8]).value
        self.g_room  = int(g[9]); self.g_charid = int(g[11])
        self.g_sword = int(g[12]); self.g_alive = c_int8(g[13]).value
        self.g_hp    = c_ushort.in_dll(lib, "guardhp_curr").value
        self.g_hpmax = c_ushort.in_dll(lib, "guardhp_max").value
        self.g_skill = c_ushort.in_dll(lib, "guard_skill").value

    def _build_roomlinks(self):
        self.rl = self.lv[1952:1952+96].astype(np.int32).reshape(24, 4)

    def _tile(self, gr, gc, idx):
        bt = int(self.fg[idx]) & 0x1F
        bm = int(self.bg[idx])
        g = self.grid
        g[CH_TILE, gr, gc] = bt / 30.0
        if bt == T_GATE:
            # bm=0 closed, 1-187 closing countdown, >=188 open (GATE_OPEN_TIMER)
            g[CH_GATE, gr, gc] = 1.0 if bm >= 188 else (0.5 if bm > 0 else 0.0)
        elif bt == T_LOOSE:
            # bm=0 intact, bm>0 without 0x80 = shaking frames, 0x80 set = about to fall
            # (confirmed: get_loose_frame in seg008.c checks modifier & 0x80)
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
SG_GET_SWORD, SG_KILL_GUARD, SG_REACH_EXIT
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
        v[i] = self.k_room / 24.0;    i += 1
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
        v[i] = np.clip(self.g_col / 9.0, 0, 1);   i += 1
        v[i] = np.clip(self.g_row / 2.0, 0, 1);   i += 1
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
        return buf

    def reset(self, seed=None, options=None):
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
        obs = self._get_obs()
        return obs, {"level": self.level, "room": self.k_room, "hp": self.hp}

    def step(self, action):
        self.rl_action.value = int(action)
        for _ in range(4):
            self.lib.play_level_2()
            if self.rl_dead.value == 1: break

        self.steps += 1
        prev_lv = self.level
        self.get_values()
        if self.level != prev_lv: self._build_roomlinks()

        alive = self.rl_dead.value != 1
        reward = -0.01 if alive else -5.0

        obs = self._get_obs()
        info = {"level": self.level, "room": self.k_room, "hp": self.hp,
                "steps": self.steps, "dead": not alive}
        return obs, reward, not alive, self.steps >= self.max_steps, info

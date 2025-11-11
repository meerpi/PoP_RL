The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
from collections import deque, OrderedDict
import numpy as np
from gymnasium import spaces

# engine sentinel: Guard.direction == 0x56 means no active guard in this room
DIR_56_NONE = 0x56

MAX_ADJ  = 96   # 24 rooms × 4 directions — hard ceiling for padded edge arrays
MAX_TRIG = 128  # safe upper bound for trigger edges per relation per level

# Shared gymnasium space for kid and guard char fields.
# guard adds "present" on top when constructing its Dict.
_CHAR_FIELDS = OrderedDict([
    ("curr_seq",  spaces.Box(0,     65535, shape=(), dtype=np.int64)),
    ("action",    spaces.Box(0,     99,    shape=(), dtype=np.int64)),
    ("frame",     spaces.Box(0,     255,   shape=(), dtype=np.int64)),
    ("repeat",    spaces.Box(0,     1,     shape=(), dtype=np.int64)),
    ("direction", spaces.Box(-128,  127,   shape=(), dtype=np.int64)),
    ("sword",     spaces.Box(0,     255,   shape=(), dtype=np.int64)),
    ("charid",    spaces.Box(0,     255,   shape=(), dtype=np.int64)),
    ("curr_row",  spaces.Box(-128,  127,   shape=(), dtype=np.int64)),
    ("curr_col",  spaces.Box(-128,  127,   shape=(), dtype=np.int64)),
    ("x",         spaces.Box(0.0,   1.0,   shape=(), dtype=np.float32)),
    ("y",         spaces.Box(0.0,   1.0,   shape=(), dtype=np.float32)),
    ("fall_x",    spaces.Box(-1.0,  1.0,   shape=(), dtype=np.float32)),
    ("fall_y",    spaces.Box(-1.0,  1.0,   shape=(), dtype=np.float32)),
    ("room",      spaces.Box(0,     24,    shape=(), dtype=np.int64)),
    ("is_alive",  spaces.Box(0.0,   1.0,   shape=(), dtype=np.float32)),
    ("death_frame_norm", spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)),
])

OBS_SPATIAL_SIZE = 240
OBS_KID_SIZE     = len(_CHAR_FIELDS)
OBS_GUARD_SIZE   = len(_CHAR_FIELDS) + 1

_TRIGGER_SPACE = spaces.Dict(OrderedDict([
    ("src",      spaces.Box(0, 23,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("dst",      spaces.Box(0, 23,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("sw_pos",   spaces.Box(0, 29,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("gate_pos", spaces.Box(0, 29,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("timer",    spaces.Box(0, 31,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("chain",    spaces.Box(0, 255, shape=(MAX_TRIG,), dtype=np.int64)),
    ("mask",     spaces.Box(0, 1,   shape=(MAX_TRIG,), dtype=np.uint8)),
]))

SPATIAL_SPACE = spaces.Dict(OrderedDict([
    ("grid_fg",       spaces.Box(0, 32,  shape=(5, 12), dtype=np.uint8)),
    ("grid_fg_flags", spaces.Box(0, 7,   shape=(5, 12), dtype=np.uint8)),
    ("grid_bg",       spaces.Box(0, 256, shape=(5, 12), dtype=np.int16)),
    ("grid_valid",    spaces.Box(0, 1,   shape=(5, 12), dtype=np.uint8)),
]))

KID_SPACE = spaces.Dict(_CHAR_FIELDS)

GUARD_SPACE = spaces.Dict(OrderedDict([*_CHAR_FIELDS.items(), ("present", spaces.Box(0, 1, shape=(), dtype=np.uint8))]))

GRAPH_SPACE = spaces.Dict(OrderedDict([
    ("skill",         spaces.Box(0, 15,   shape=(24,),      dtype=np.int64)),
    ("color",         spaces.Box(0, 255,  shape=(24,),      dtype=np.int64)),
    ("rx",            spaces.Box(0.0, 1.0, shape=(24,),     dtype=np.float32)),
    ("ry",            spaces.Box(0.0, 1.0, shape=(24,),     dtype=np.float32)),
    ("has_guard",     spaces.Box(0, 1,    shape=(24,),      dtype=np.uint8)),
    ("is_subgoal",    spaces.Box(0, 1,    shape=(24,),      dtype=np.uint8)),
    ("visited",       spaces.Box(0, 1,    shape=(24,),      dtype=np.uint8)),
    # per-room tile counts for 5 critical types: spike(2), loose(5), gate(4), drop_btn(6), raise_btn(15)
    ("room_fg_counts", spaces.Box(0, 30,  shape=(24, 5),    dtype=np.uint8)),
    ("edge_src",      spaces.Box(0, 23,   shape=(MAX_ADJ,), dtype=np.int64)),
    ("edge_dst",      spaces.Box(0, 23,   shape=(MAX_ADJ,), dtype=np.int64)),
    ("edge_fatal",    spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_risky",    spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_trav",     spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_irr",      spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_hop",      spaces.Box(0, 9999, shape=(MAX_ADJ,), dtype=np.int64)),
    ("edge_mask",     spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("trigger_open",  _TRIGGER_SPACE),
    ("trigger_close", _TRIGGER_SPACE),
    ("subgoal_hops",  spaces.Box(0, 9999, shape=(), dtype=np.int64)),
]))

MISC_SPACE = spaces.Dict(OrderedDict([
    ("hitp_curr",         spaces.Box(0.0, 1.0,  shape=(), dtype=np.float32)),
    ("hitp_max",          spaces.Box(0.0, 10.0, shape=(), dtype=np.float32)),
    ("guardhp_curr",      spaces.Box(0.0, 1.0,  shape=(), dtype=np.float32)),
    ("guardhp_max",       spaces.Box(0.0, 10.0, shape=(), dtype=np.float32)),
    ("have_sword",        spaces.Box(0, 1,       shape=(), dtype=np.int64)),
    ("current_level",     spaces.Box(0, 31,      shape=(), dtype=np.int64)),
    ("hop_norm",          spaces.Box(0.0, 1.0,   shape=(), dtype=np.float32)),
    ("subgoal_reachable", spaces.Box(0, 1,       shape=(), dtype=np.int64)),
    ("subgoal_room",      spaces.Box(-1, 23,     shape=(), dtype=np.int64)),
]))


def classify_fall(src, dst, lv, fg):
    """Check if src→dst is a downward transition with a fatal or risky drop.
    The fall starts at src's bottom row (row 2). We find which columns are
    empty there, then trace through dst (and further down-links) counting
    contiguous empty rows. ≥3 total empty rows = fatal, 2 = risky.
    """
    # only down-links can be falls
    if lv.roomlinks[src].down != dst + 1:
        return 0, 0
    # which columns are empty at src's bottom row?
    src_base = src * 30
    empty_cols = [c for c in range(10) if (fg[src_base + 20 + c] & 0x1F) == 0]
    if not empty_cols:
        return 0, 0
    rows = 1  # src row 2 itself

    cur = dst
    while rows < 10:
        for row_idx in range(3):
            still_empty = any((fg[cur * 30 + row_idx * 10 + c] & 0x1F) == 0 for c in empty_cols)
            if still_empty:
                rows += 1
            else:
                if rows >= 3:
                    return 1, 0
                if rows == 2:
                    return 0, 1
                return 0, 0
        else:
            # fell through all 3 rows of cur — keep going down
            nxt = lv.roomlinks[cur].down
            if not nxt:
                break
            cur = nxt - 1
            continue
        break

    if rows >= 3:
        return 1, 0
    if rows == 2:
        return 0, 1
    return 0, 0


def bfs_dist(start, goal, adj):
    """Shortest directed path from start to goal. Returns -1 if unreachable."""
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        node, dist = q.popleft()
        for nb in adj.get(node, []):
            if nb == goal:
                return dist + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))
    return -1


def _pad(arr, size, fill=0):
    out = np.full(size, fill, dtype=arr.dtype)
    out[:len(arr)] = arr
    return out


class ObsBuilder:
    """Builds all observation components from the SDLPoP interface.
    Owns the map-graph state (visited, edge arrays, subgoal cache) so the
    env doesn't have to. Call build_map_graph() once per level load.
    """

    def __init__(self, engine):
        self.engine = engine

    # ------------------------------------------------------------------ #
    # Component 1: spatial grid                                           #
    # ------------------------------------------------------------------ #

    def spatial_tensor(self):
        """Builds a padded (5,12) view of the current room and its four neighbors.

        The halo ring lets the network see tiles across doorways without needing
        a separate side-channel. Returns four arrays — fg tile ids (lower 5 bits),
        fg flags (upper 3 bits — bit 0x20 is loose-floor armed/stable per seg007.c,
        others retained cheaply), bg bytes, and a valid mask.
        """
        fg_flat = np.frombuffer(self.engine._level.fg, dtype=np.uint8)
        bg_flat = np.frombuffer(self.engine._level.bg, dtype=np.uint8)
        room  = int(self.engine._kid.room)
        links = self.engine._level.roomlinks[room - 1]

        grid_fg       = np.zeros((5, 12), dtype=np.uint8)   # max value 32 (sentinel)
        grid_fg_flags = np.zeros((5, 12), dtype=np.uint8)   # max value 7
        grid_bg       = np.zeros((5, 12), dtype=np.int16)   # int16: real 0-255, sentinel 256
        grid_valid    = np.zeros((5, 12), dtype=np.uint8)

        base = (room - 1) * 30
        grid_fg[1:4, 1:11]       = (fg_flat[base:base+30] & 0x1F).reshape(3, 10)
        grid_fg_flags[1:4, 1:11] = (fg_flat[base:base+30] >> 5).reshape(3, 10)
        grid_bg[1:4, 1:11]       = bg_flat[base:base+30].reshape(3, 10)
        grid_valid[1:4, 1:11]    = 1

        left = links.left
        if left:
            nb = (left - 1) * 30
            grid_fg[1:4, 0]       = fg_flat[[nb+9, nb+19, nb+29]] & 0x1F
            grid_fg_flags[1:4, 0] = fg_flat[[nb+9, nb+19, nb+29]] >> 5
            grid_bg[1:4, 0]       = bg_flat[[nb+9, nb+19, nb+29]]
            grid_valid[1:4, 0]    = 1

        right = links.right
        if right:
            nb = (right - 1) * 30
            grid_fg[1:4, 11]       = fg_flat[[nb, nb+10, nb+20]] & 0x1F
            grid_fg_flags[1:4, 11] = fg_flat[[nb, nb+10, nb+20]] >> 5
            grid_bg[1:4, 11]       = bg_flat[[nb, nb+10, nb+20]]
            grid_valid[1:4, 11]    = 1

        up = links.up
        if up:
            nb = (up - 1) * 30
            grid_fg[0, 1:11]       = fg_flat[nb+20:nb+30] & 0x1F
            grid_fg_flags[0, 1:11] = fg_flat[nb+20:nb+30] >> 5
            grid_bg[0, 1:11]       = bg_flat[nb+20:nb+30]
            grid_valid[0, 1:11]    = 1

        down = links.down
        if down:
            nb = (down - 1) * 30
            grid_fg[4, 1:11]       = fg_flat[nb:nb+10] & 0x1F
            grid_fg_flags[4, 1:11] = fg_flat[nb:nb+10] >> 5
            grid_bg[4, 1:11]       = bg_flat[nb:nb+10]
            grid_valid[4, 1:11]    = 1

        # Padding sentinels — written after all halo fills so only truly
        # unoccupied cells get them. Real fg ids are 0-31; 32 means "no room".
        # Real bg values are 0-255; 256 (only reachable in int16) means "no room".
        invalid = grid_valid == 0
        grid_fg[invalid] = 32
        grid_bg[invalid] = 256

        return grid_fg, grid_fg_flags, grid_bg, grid_valid

    # ------------------------------------------------------------------ #
    # Components 2a/2b: entity tensors                                   #
    # ------------------------------------------------------------------ #

    def kid_tensor(self):
        """All 16 raw char_type fields for Kid. No embedding — that's the model's job.
        This is the single source of truth; scatter payload and bypass vector both slice from it.
        """
        k = self.engine._kid
        return {
            # animation phase
            "curr_seq":  np.int64(k.curr_seq),
            "action":    np.int64(k.action),    # vocab {0-7,99} — full enum actions in types.h:402-412
            "frame":     np.int64(k.frame),
            "repeat":    np.int64(k.repeat),    # {0,1}: safe_step() sets 1 while stepping to edge (seg005.c:609), clears to 0 at edge unless edge_type==WALL (seg005.c:611-612)
            # stance / orientation
            "direction": np.int64(k.direction),
            "sword":     np.int64(k.sword),
            "charid":    np.int64(k.charid),
            # position / momentum
            "curr_row":  np.int64(k.curr_row),
            "curr_col":  np.int64(k.curr_col),
            "x":         np.float32(k.x) / 255.0,
            "y":         np.float32(k.y) / 255.0,
            "fall_x":    np.float32(k.fall_x) / 128.0,
            "fall_y":    np.float32(k.fall_y) / 128.0,
            "room":      np.int64(k.room),
            "is_alive":  np.float32(1.0 if k.alive == -1 else 0.0),
            "death_frame_norm": np.float32(max(0, k.alive) / 8.0),
        }

    def guard_tensor(self):
        """Same 16 raw fields as kid_tensor, from Guard's struct.
        Presence uses dir_56_none (the engine's own sentinel) as primary check,
        with a room-match guard against a stale-but-technically-active struct.
        """
        g = self.engine._guard
        present = (g.direction != DIR_56_NONE) and (g.room == self.engine._kid.room)
        if not present:
            return self.empty_guard()
        return {
            "curr_seq":  np.int64(g.curr_seq),
            "action":    np.int64(g.action),    # vocab {0-7,99} — full enum actions in types.h:402-412
            "frame":     np.int64(g.frame),
            "repeat":    np.int64(g.repeat),    # {0,1}: safe_step() sets 1 while stepping to edge (seg005.c:609), clears to 0 at edge unless edge_type==WALL (seg005.c:611-612)
            "direction": np.int64(g.direction),
            "sword":     np.int64(g.sword),
            "charid":    np.int64(g.charid),    # 0=kid, 1=shadow, 2=guard, 4=skeleton, 5=princess, 6=vizier, 24=mouse (types.h:323-332)
            "curr_row":  np.int64(g.curr_row),
            "curr_col":  np.int64(g.curr_col),
            "x":         np.float32(g.x) / 255.0,
            "y":         np.float32(g.y) / 255.0,
            "fall_x":    np.float32(g.fall_x) / 128.0,
            "fall_y":    np.float32(g.fall_y) / 128.0,
            "room":      np.int64(g.room),
            "is_alive":  np.float32(1.0 if g.alive == -1 else 0.0),
            "death_frame_norm": np.float32(max(0, g.alive) / 8.0),
            "present":   np.uint8(1),
        }

    def empty_guard(self):
        zero_int = np.int64(0)
        zero_float = np.float32(0.0)
        return {
            "curr_seq":  zero_int,
            "action":    zero_int,
            "frame":     zero_int,
            "repeat":    zero_int,
            "direction": zero_int,
            "sword":     zero_int,
            "charid":    zero_int,
            "curr_row":  zero_int,
            "curr_col":  zero_int,
            "x":         zero_float,
            "y":         zero_float,
            "fall_x":    zero_float,
            "fall_y":    zero_float,
            "room":      zero_int,
            "is_alive":  zero_float,
            "death_frame_norm": zero_float,
            "present":   np.uint8(0),
        }

    # ------------------------------------------------------------------ #
    # Component 3: room graph                                             #
    # ------------------------------------------------------------------ #

    def build_map_graph(self):
        """Precompute the static room graph for the current level.
        Call once per level load. Only visited and subgoal_hops change after this.
        """
        lv = self.engine._level
        fg = np.frombuffer(lv.fg, dtype=np.uint8)

        # node features (24 rooms, 0-indexed)
        skill = np.frombuffer(lv.guards_skill, dtype=np.uint8).copy()
        color = np.frombuffer(lv.guards_color, dtype=np.uint8).copy()
        gtile = np.frombuffer(lv.guards_tile, dtype=np.uint8)
        rxs   = np.frombuffer(lv.roomxs, dtype=np.uint8)
        rys   = np.frombuffer(lv.roomys, dtype=np.uint8)

        self.room_nodes = {
            "skill":     skill.astype(np.int64),    # vocab 16
            "color":     color.astype(np.int64),    # vocab 256
            "rx":        rxs.astype(np.float32) / 255.0,
            "ry":        rys.astype(np.float32) / 255.0,
            # guards_tile < 30 is the engine's own sentinel (enter_guard seg002:115,
            # pos_guards seg003:834). guards_skill==15 is NOT a no-guard marker:
            # on level 1, 7 of 24 rooms have tile>=30 (no guard) but skill in {0,1}.
            "has_guard": (gtile < 30).astype(np.uint8),
        }
        # per-room counts of 5 critical tile types (computed once per level)
        # columns: spike(2), loose_floor(5), gate(4), drop_btn(6), raise_btn(15)
        _CRITICAL = np.array([2, 5, 4, 6, 15], dtype=np.uint8)
        tile_ids = fg & 0x1F  # lower 5 bits are tile type
        counts = np.zeros((24, 5), dtype=np.uint8)
        for r in range(24):
            room_tiles = tile_ids[r * 30 : r * 30 + 30]
            for ci, t in enumerate(_CRITICAL):
                counts[r, ci] = np.uint8((room_tiles == t).sum())
        self.room_fg_counts = counts
        self.visited = np.zeros(24, dtype=np.uint8)

        # adjacency edges — build raw first, pad after all BFS is done
        adj_src, adj_dst, adj_is_up = [], [], []
        adj_fatal, adj_risky = [], []
        for r in range(24):
            links = lv.roomlinks[r]
            for is_up, nb in [(0, links.left), (0, links.right), (1, links.up), (0, links.down)]:
                if nb == 0:
                    continue
                adj_src.append(r)
                adj_dst.append(nb - 1)
                adj_is_up.append(is_up)
                f, ri = classify_fall(r, nb - 1, lv, fg)
                adj_fatal.append(f)
                adj_risky.append(ri)

        n = len(adj_src)
        raw_src   = np.array(adj_src,   dtype=np.int64)
        raw_dst   = np.array(adj_dst,   dtype=np.int64)
        raw_fatal = np.array(adj_fatal, dtype=np.uint8)
        raw_risky = np.array(adj_risky, dtype=np.uint8)

        # traversable: down/left/right always true; up defaults false and only
        # flips to true if the reverse down-link EXISTS and is safe — positive
        # proof of climbability. "Not proven dangerous" isn't enough: a missing
        # reciprocal down-edge (e.g. rooms 13/18/24's broken links) would pass
        # a negative check vacuously.
        safe_down = set()
        for i in range(n):
            if adj_is_up[i] == 0 and not raw_fatal[i] and not raw_risky[i]:
                safe_down.add((raw_src[i], raw_dst[i]))

        trav = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            if adj_is_up[i] == 0:
                trav[i] = 1  # down/left/right always traversable
            elif (raw_dst[i], raw_src[i]) in safe_down:
                trav[i] = 1  # up: only if reverse-down is proven safe

        # irreversible / return_hop: BFS from dst→src over safe+traversable edges
        safe = {}
        for i in range(n):
            if trav[i] and not raw_fatal[i] and not raw_risky[i]:
                safe.setdefault(int(raw_src[i]), []).append(int(raw_dst[i]))

        irr = np.zeros(n, dtype=np.uint8)
        hop = np.full(n, 9999, dtype=np.int64)
        for i in range(n):
            d = bfs_dist(int(raw_dst[i]), int(raw_src[i]), safe)
            if d < 0:
                irr[i] = 1
            else:
                hop[i] = d

        # Pad all edge arrays to MAX_ADJ; real edges are in slots [0, n)
        self.n_edges  = n
        edge_mask     = np.zeros(MAX_ADJ, dtype=np.uint8)
        edge_mask[:n] = 1
        self.edge_mask  = edge_mask
        self.edge_src   = _pad(raw_src,   MAX_ADJ)
        self.edge_dst   = _pad(raw_dst,   MAX_ADJ)
        self.edge_fatal = _pad(raw_fatal, MAX_ADJ)
        self.edge_risky = _pad(raw_risky, MAX_ADJ)
        self.edge_trav  = _pad(trav,      MAX_ADJ)
        self.edge_irr   = _pad(irr,       MAX_ADJ)
        self.edge_hop   = _pad(hop,       MAX_ADJ, fill=9999)

        # trigger edges — opener (raise_btn=15) and closer (drop_btn=6) split
        # into two relations; bg value of the button tile is the doorlink index
        dl1 = np.frombuffer(lv.doorlinks1, dtype=np.uint8)
        dl2 = np.frombuffer(lv.doorlinks2, dtype=np.uint8)
        open_edges, close_edges = [], []

        for r in range(24):
            base = r * 30
            for pos in range(30):
                tile = fg[base + pos] & 0x1F
                if tile == 15:
                    bucket = open_edges
                elif tile == 6:
                    bucket = close_edges
                else:
                    continue
                idx = int(lv.bg[base + pos])
                chain_pos = 0
                while True:
                    dst_room = ((dl1[idx] & 0x60) >> 5) + ((dl2[idx] & 0xE0) >> 3)
                    dst_tile = dl1[idx] & 0x1F
                    timer    = dl2[idx] & 0x1F
                    has_next = not (dl1[idx] & 0x80)
                    if dst_room > 0:
                        bucket.append((r, dst_room - 1, pos, dst_tile, timer, chain_pos))
                    chain_pos += 1
                    if not has_next:
                        break
                    idx += 1

        self.trigger_open  = pack_triggers(open_edges)
        self.trigger_close = pack_triggers(close_edges)
        self.subgoal       = -1
        self.subgoal_hops  = 9999

    def map_graph(self, subgoal_room=-1):
        """Returns the current room graph state. Flips visited for kid's room.
        Recomputes subgoal_hops only when the subgoal changes.
        """
        room = int(self.engine._kid.room)
        self.visited[room - 1] = 1

        if subgoal_room != self.subgoal:
            self.subgoal = subgoal_room
            if subgoal_room < 0:
                self.subgoal_hops = 9999
            else:
                # BFS on traversable, non-fatal edges (risky allowed)
                safe = {}
                for i in range(self.n_edges):
                    if self.edge_trav[i] and not self.edge_fatal[i]:
                        safe.setdefault(int(self.edge_src[i]), []).append(int(self.edge_dst[i]))
                d = bfs_dist(room - 1, subgoal_room, safe)
                self.subgoal_hops = d if d >= 0 else 9999

        is_subgoal = np.zeros(24, dtype=np.uint8)
        if subgoal_room >= 0:
            is_subgoal[subgoal_room] = 1

        return {
            # nodes (24,)
            "skill":         self.room_nodes["skill"],
            "color":         self.room_nodes["color"],
            "rx":            self.room_nodes["rx"],
            "ry":            self.room_nodes["ry"],
            "has_guard":     self.room_nodes["has_guard"],
            "is_subgoal":    is_subgoal,
            "visited":       self.visited.copy(),
            "room_fg_counts": self.room_fg_counts,
            # adjacency edges — all padded to MAX_ADJ=96; use edge_mask to find real ones
            "edge_src":   self.edge_src,
            "edge_dst":   self.edge_dst,
            "edge_fatal": self.edge_fatal,
            "edge_risky": self.edge_risky,
            "edge_trav":  self.edge_trav,
            "edge_irr":   self.edge_irr,
            "edge_hop":   self.edge_hop,
            "edge_mask":  self.edge_mask,
            # trigger edges (two relations, padded to MAX_TRIG=128)
            "trigger_open":  self.trigger_open,
            "trigger_close": self.trigger_close,
            # global
            "subgoal_hops": np.int64(self.subgoal_hops),
        }

    def misc_tensor(self, subgoal_room=-1):
        """Returns the component 4 (misc) observation fields."""
        room = int(self.engine._kid.room)
        if subgoal_room != self.subgoal:
            self.subgoal = subgoal_room
            if subgoal_room < 0:
                self.subgoal_hops = 9999
            else:
                safe = {}
                for i in range(self.n_edges):
                    if self.edge_trav[i] and not self.edge_fatal[i]:
                        safe.setdefault(int(self.edge_src[i]), []).append(int(self.edge_dst[i]))
                d = bfs_dist(room - 1, subgoal_room, safe)
                self.subgoal_hops = d if d >= 0 else 9999

        h_max = max(int(self.engine._hitp_max.value), 1)
        h_curr = float(self.engine._hitp_curr.value)
        hitp_curr = np.float32(h_curr / h_max)
        hitp_max = np.float32(float(h_max) / 10.0)

        g = self.engine._guard
        present = (g.direction != DIR_56_NONE) and (g.room == self.engine._kid.room)
        if present:
            gh_max = max(int(self.engine._guardhp_max.value), 1)
            gh_curr = float(self.engine._guardhp_curr.value)
            guardhp_curr = np.float32(gh_curr / gh_max)
            guardhp_max = np.float32(float(gh_max) / 10.0)
        else:
            guardhp_curr = np.float32(0.0)
            guardhp_max = np.float32(0.0)

        hops = self.subgoal_hops
        hop_norm = np.float32(min(hops, 24) / 24.0)
        subgoal_reachable = np.int64(1 if hops < 9999 else 0)

        return {
            "hitp_curr":         hitp_curr,
            "hitp_max":          hitp_max,
            "guardhp_curr":      guardhp_curr,
            "guardhp_max":       guardhp_max,
            "have_sword":        np.int64(1 if self.engine._have_sword.value else 0),
            "current_level":     np.int64(self.engine._current_level.value),
            "hop_norm":          hop_norm,
            "subgoal_reachable": subgoal_reachable,
            "subgoal_room":      np.int64(subgoal_room),
        }

    def build(self, subgoal_room=-1):
        """Single call that returns the complete obs dict.
        map_graph() and misc_tensor() both need the BFS result for subgoal_hops;
        calling map_graph() first lets misc_tensor() reuse the cached value.
        """
        fg, fg_flags, bg, valid = self.spatial_tensor()
        return {
            "spatial": {"grid_fg": fg, "grid_fg_flags": fg_flags, "grid_bg": bg, "grid_valid": valid},
            "kid":     self.kid_tensor(),
            "guard":   self.guard_tensor(),
            "graph":   self.map_graph(subgoal_room=subgoal_room),
            "misc":    self.misc_tensor(subgoal_room=subgoal_room),
        }


def pack_triggers(edges):
    """Pack raw trigger tuples into fixed-size arrays padded to MAX_TRIG.
    Real entries are in slots [0, n); use 'mask' to find them.
    """
    assert len(edges) <= MAX_TRIG, f"trigger edges ({len(edges)}) exceed MAX_TRIG ({MAX_TRIG})"
    n = len(edges)
    mask     = np.zeros(MAX_TRIG, dtype=np.uint8)
    src      = np.zeros(MAX_TRIG, dtype=np.int64)
    dst      = np.zeros(MAX_TRIG, dtype=np.int64)
    sw_pos   = np.zeros(MAX_TRIG, dtype=np.int64)
    gate_pos = np.zeros(MAX_TRIG, dtype=np.int64)
    timer    = np.zeros(MAX_TRIG, dtype=np.int64)
    chain    = np.zeros(MAX_TRIG, dtype=np.int64)
    if n > 0:
        mask[:n] = 1
        s, d, sw, gt, tm, ch = zip(*edges[:n])
        src[:n]      = s
        dst[:n]      = d
        sw_pos[:n]   = sw
        gate_pos[:n] = gt
        timer[:n]    = tm
        chain[:n]    = ch
    return {
        "src":      src,
        "dst":      dst,
        "sw_pos":   sw_pos,
        "gate_pos": gate_pos,
        "timer":    timer,
        "chain":    chain,
        "mask":     mask,
    }

The above content shows the entire, complete file contents of the requested file.

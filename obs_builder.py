from collections import deque, OrderedDict
import numpy as np
from gymnasium import spaces

MAX_ADJ  = 96   # 24 rooms × 4 directions — hard ceiling for padded edge arrays
MAX_TRIG = 128  # safe upper bound for trigger edges per relation per level
MAX_EDGES = MAX_ADJ + MAX_TRIG + MAX_TRIG  # 352 — total padded edge slots (adj + open + close)

_TRIGGER_SPACE = spaces.Dict(OrderedDict([
    ("src",      spaces.Box(0, 23,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("dst",      spaces.Box(0, 23,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("sw_pos",   spaces.Box(0, 29,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("gate_pos", spaces.Box(0, 29,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("timer",    spaces.Box(0, 31,  shape=(MAX_TRIG,), dtype=np.int64)),
    ("chain",    spaces.Box(0, 255, shape=(MAX_TRIG,), dtype=np.int64)),
    ("mask",     spaces.Box(0, 1,   shape=(MAX_TRIG,), dtype=np.uint8)),
]))

GRAPH_SPACE = spaces.Dict(OrderedDict([
    ("edge_src",   spaces.Box(0, 23,   shape=(MAX_ADJ,), dtype=np.int64)),
    ("edge_dst",   spaces.Box(0, 23,   shape=(MAX_ADJ,), dtype=np.int64)),
    ("edge_fatal", spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_risky", spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_trav",  spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("edge_mask",  spaces.Box(0, 1,    shape=(MAX_ADJ,), dtype=np.uint8)),
    ("subgoal_hops", spaces.Box(0, 9999, shape=(), dtype=np.int64)),
]))


def classify_fall(drop_height: int) -> int:
    """Classify drop height: 0 = safe, 1 = risky (hp loss), 2 = fatal."""
    if drop_height <= 1: return 0
    elif drop_height <= 2: return 1
    return 2

def _tile_is_open(tile, bg_val):
    """True if this tile position doesn't block a falling body.
    A gate (tile 4) is passable when its modifier >= 112 — that's the
    engine's own character-traversal check (seg003.c:746).
    """
    if tile == 0:
        return True
    if tile == 4:
        return bg_val >= 112
    return False


def classify_fall(src, dst, lv, fg, bg):
    """Check if src->dst is a downward transition with a fatal or risky drop.
    The fall starts at src's bottom row (row 2). We find which columns are
    open there (empty or passable gate), then trace through dst (and further
    down-links) counting contiguous open rows. >=3 = fatal, 2 = risky.
    bg must be a flat uint8/int array parallel to fg (0-indexed, same shape).
    """
    # only down-links can be falls
    if lv.roomlinks[src].down != dst + 1:
        return 0, 0
    # which columns are open at src's bottom row?
    src_base = src * 30
    empty_cols = [
        c for c in range(10)
        if _tile_is_open(fg[src_base + 20 + c] & 0x1F, int(bg[src_base + 20 + c]))
    ]
    if not empty_cols:
        return 0, 0

    max_rows = 1
    for c in empty_cols:
        rows = 1
        cur = dst
        while rows < 10:
            blocked = False
            for row_idx in range(3):
                pos = cur * 30 + row_idx * 10 + c
                if not _tile_is_open(fg[pos] & 0x1F, int(bg[pos])):
                    blocked = True
                    break
                rows += 1
            if blocked:
                break
            nxt = lv.roomlinks[cur].down
            if not nxt:
                break
            cur = nxt - 1
        if rows > max_rows:
            max_rows = rows

    if max_rows >= 3:
        return 1, 0
    if max_rows == 2:
        return 0, 1
    return 0, 0


# ────────────────────────────────────────────────────────────────────────────
# Level1Static — process-level singleton for truly level-constant data
# Built once per worker process; shared across all episodes on that worker.
# ────────────────────────────────────────────────────────────────────────────

_LEVEL1_STATIC = None


def get_level1_static(lv):
    """Return the Level1Static singleton, building it on first call."""
    global _LEVEL1_STATIC
    if _LEVEL1_STATIC is None:
        _LEVEL1_STATIC = _build_level1_static(lv)
    return _LEVEL1_STATIC


def _build_level1_static(lv):
    """Build the process-level cache of level-design constants.

    Includes:
      room_nodes   — skill/color/rx/ry/has_guard (level data, never changes)
      trigger_open / trigger_close — doorlink wiring
      edge_src/dst — list of (src, dst) pairs for every adjacency edge
      fg_reverse_index — tile_pos -> [edge_indices] for fg-driven classify_fall cells
      bg_reverse_index — tile_pos -> [edge_indices] for bg-driven (gate) cells
    """
    skill = np.frombuffer(lv.guards_skill, dtype=np.uint8).copy()
    color = np.frombuffer(lv.guards_color, dtype=np.uint8).copy()
    gtile = np.frombuffer(lv.guards_tile,  dtype=np.uint8)
    rxs   = np.frombuffer(lv.roomxs,       dtype=np.uint8)
    rys   = np.frombuffer(lv.roomys,       dtype=np.uint8)

    room_nodes = {
        "skill":     skill.astype(np.int64),
        "color":     color.astype(np.int64),
        "rx":        rxs.astype(np.float32) / 255.0,
        "ry":        rys.astype(np.float32) / 255.0,
        # has_guard reads guards_tile from level data — this is the room's *assigned*
        # guard slot (level design constant), not the guard's current runtime position.
        "has_guard": (gtile < 30).astype(np.uint8),
    }

    # Trigger edges (doorlink wiring is purely level-constant)
    fg_flat = np.frombuffer(lv.fg, dtype=np.uint8)
    dl1 = np.frombuffer(lv.doorlinks1, dtype=np.uint8)
    dl2 = np.frombuffer(lv.doorlinks2, dtype=np.uint8)
    open_edges, close_edges = [], []
    for r in range(24):
        base = r * 30
        for pos in range(30):
            tile = fg_flat[base + pos] & 0x1F
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
                    bucket.append((r, dst_room - 1, base + pos, dst_tile, timer, chain_pos))
                chain_pos += 1
                if not has_next:
                    break
                idx += 1

    trigger_open  = pack_triggers(open_edges)
    trigger_close = pack_triggers(close_edges)

    # Build adjacency edge list (src, dst pairs only — used for reverse index)
    adj_pairs = []
    for r in range(24):
        links = lv.roomlinks[r]
        for nb in [links.left, links.right, links.up, links.down]:
            if nb == 0:
                continue
            adj_pairs.append((r, nb - 1))

    # Build reverse indices: which edge indices does classify_fall touch for a given tile pos?
    fg_rev = {}  # pos -> [edge_idx, ...]
    bg_rev = {}  # pos -> [edge_idx, ...]

    fg_cur = np.frombuffer(lv.fg, dtype=np.uint8)  # level initial fg

    for e_idx, (src, dst) in enumerate(adj_pairs):
        # Only down-links can be falls
        if lv.roomlinks[src].down != dst + 1:
            continue
        src_base = src * 30
        # src bottom row positions — any that are open (or gate) contribute
        for c in range(10):
            pos = src_base + 20 + c
            tile = fg_cur[pos] & 0x1F
            if tile == 4:
                bg_rev.setdefault(pos, []).append(e_idx)
            elif tile == 0:
                fg_rev.setdefault(pos, []).append(e_idx)
            else:
                # currently solid — register on fg so we notice if it collapses
                fg_rev.setdefault(pos, []).append(e_idx)

        # trace downward scan cells — same logic as classify_fall, geometry only
        for c in range(10):
            cur = dst
            visited_rooms = set()
            while cur is not None and cur not in visited_rooms:
                visited_rooms.add(cur)
                for row_idx in range(3):
                    pos = cur * 30 + row_idx * 10 + c
                    tile = fg_cur[pos] & 0x1F
                    if tile == 4:
                        bg_rev.setdefault(pos, []).append(e_idx)
                    else:
                        fg_rev.setdefault(pos, []).append(e_idx)
                nxt = lv.roomlinks[cur].down
                cur = (nxt - 1) if nxt else None

    # Deduplicate
    fg_reverse_index = {k: list(dict.fromkeys(v)) for k, v in fg_rev.items()}
    bg_reverse_index = {k: list(dict.fromkeys(v)) for k, v in bg_rev.items()}

    class _Static:
        pass
    s = _Static()
    s.room_nodes       = room_nodes
    s.trigger_open     = trigger_open
    s.trigger_close    = trigger_close
    s.adj_pairs        = adj_pairs          # list of (src, dst)
    s.fg_reverse_index = fg_reverse_index
    s.bg_reverse_index = bg_reverse_index
    return s


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


class ObsBuilder:
    """Graph-only subset of ObsBuilder: owns map-graph state (visited, edge arrays,
    subgoal cache). Call build_map_graph() once per level load / episode reset.
    """

    def __init__(self, target):
        self.target = target

        # Tier 1 singleton (built once per worker process)
        self.static = None
        # Tier 2 per-episode caches — populated in build_map_graph()
        self._cached_fg = None
        self._cached_bg = None
        self._subgoal_kid_room = -1
        # SPS instrumentation
        self._graph_skip_steps = 0
        self._graph_recompute_steps = 0

    @property
    def level(self):
        if hasattr(self.target, "level"):
            return self.target.level
        return getattr(self.target, "_level", None)

    @property
    def kid(self):
        if hasattr(self.target, "kid"):
            return self.target.kid
        return getattr(self.target, "_kid", None)

    @property
    def have_sword(self):
        if hasattr(self.target, "have_sword"):
            val = self.target.have_sword
            return bool(val.value if hasattr(val, "value") else val)
        if hasattr(self.target, "_have_sword"):
            val = self.target._have_sword
            return bool(val.value if hasattr(val, "value") else val)
        return False

    def build_map_graph(self):
        """Initialize the room graph for the current episode.
        Tier 1 (level-constant) data is pulled from the process singleton.
        Tier 2 (episode-scoped) data is computed fresh here and patched lazily
        each step in map_graph().
        """
        lv = self.level
        fg = np.frombuffer(lv.fg, dtype=np.uint8)
        bg = np.frombuffer(lv.bg, dtype=np.uint8)

        # ── Tier 1: pull from process singleton ──────────────────────────────
        self.static = get_level1_static(lv)
        self.room_nodes     = self.static.room_nodes
        self.trigger_open   = self.static.trigger_open
        self.trigger_close  = self.static.trigger_close
        adj_pairs           = self.static.adj_pairs  # list of (src, dst)

        # ── Tier 2: snapshot fg/bg for delta detection ────────────────────────
        self._cached_fg = fg.copy()
        self._cached_bg = bg.copy().astype(np.int16)
        self._cached_have_sword = self.have_sword

        # ── room_fg_counts — computed from fresh fg snapshot ──────────────────
        _CRITICAL = np.array([2, 11, 4, 6, 15], dtype=np.uint8)
        tile_ids = fg & 0x1F
        self.room_fg_counts = (tile_ids.reshape(24, 30, 1) == _CRITICAL).sum(axis=1).astype(np.uint8)
        self.visited = np.zeros(24, dtype=np.uint8)

        # ── adjacency edges — classify_fall for every down-link ───────────────
        n = len(adj_pairs)
        adj_is_up = []
        adj_fatal = []
        adj_risky = []
        raw_src   = np.empty(n, dtype=np.int64)
        raw_dst   = np.empty(n, dtype=np.int64)
        for i, (src, dst) in enumerate(adj_pairs):
            raw_src[i] = src
            raw_dst[i] = dst
            lnk = lv.roomlinks[src]
            is_up = int(lnk.up == dst + 1 and lnk.down != dst + 1)
            adj_is_up.append(is_up)
            f, ri = classify_fall(src, dst, lv, fg, bg)
            adj_fatal.append(f)
            adj_risky.append(ri)

        raw_fatal = np.array(adj_fatal, dtype=np.uint8)
        raw_risky = np.array(adj_risky, dtype=np.uint8)

        # Pad all edge arrays to MAX_ADJ; real edges are in slots [0, n)
        self.n_edges  = n
        edge_mask     = np.zeros(MAX_ADJ, dtype=np.uint8)
        edge_mask[:n] = 1
        self.edge_mask  = edge_mask
        self.edge_src   = _pad(raw_src,   MAX_ADJ)
        self.edge_dst   = _pad(raw_dst,   MAX_ADJ)
        self.edge_fatal = _pad(raw_fatal, MAX_ADJ)
        self.edge_risky = _pad(raw_risky, MAX_ADJ)
        self.edge_trav  = np.zeros(MAX_ADJ, dtype=np.uint8)
        self.edge_irr   = np.zeros(MAX_ADJ, dtype=np.int64)
        self.edge_hop   = np.full(MAX_ADJ, 9999, dtype=np.int64)

        # Run traversability + BFS pass once at reset
        self._run_bfs_pass(adj_is_up)

        self.subgoal      = -1
        self.subgoal_hops = 9999

    def _run_bfs_pass(self, adj_is_up=None):
        """Recompute traversability, irr, hop for all edges from current fatal/risky.
        Called once at reset and whenever any edge's fatal/risky changes.
        adj_is_up: list of 0/1 per real edge (len = n_edges). If None, infer from
        edge arrays (slower, only used after a lazy patch).
        """
        n = self.n_edges
        if adj_is_up is None:
            lv = self.level
            adj_is_up = []
            for i in range(n):
                src = int(self.edge_src[i])
                dst = int(self.edge_dst[i])
                lnk = lv.roomlinks[src]
                adj_is_up.append(int(lnk.up == dst + 1 and lnk.down != dst + 1))

        raw_fatal = self.edge_fatal[:n]
        raw_risky = self.edge_risky[:n]
        raw_src   = self.edge_src[:n]
        raw_dst   = self.edge_dst[:n]

        # traversable
        safe_down = set()
        for i in range(n):
            if adj_is_up[i] == 0 and not raw_fatal[i] and not raw_risky[i]:
                safe_down.add((int(raw_src[i]), int(raw_dst[i])))

        trav = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            if adj_is_up[i] == 0:
                trav[i] = 1
            elif (int(raw_dst[i]), int(raw_src[i])) in safe_down:
                trav[i] = 1

        # Room 9 (0-indexed 8) exception: edges into Room 9 are non-traversable until sword is obtained
        if not self.have_sword:
            for i in range(n):
                if raw_dst[i] == 8:
                    trav[i] = 0

        self.edge_trav[:n] = trav
        self.edge_trav[n:] = 0

        # irr + hop via BFS
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
        self.edge_irr[:n] = irr
        self.edge_irr[n:] = 0
        self.edge_hop[:n] = hop
        self.edge_hop[n:] = 9999

        # Invalidate cached subgoal so next map_graph() call reruns BFS
        self.subgoal = -2  # sentinel that != any valid room

    def _refresh_subgoal(self, subgoal_room):
        """Recompute subgoal_hops from kid's current room to subgoal_room.
        No-op only when BOTH the subgoal AND the kid's room are unchanged.
        """
        kid_room = int(self.kid.room)
        if subgoal_room == self.subgoal and kid_room == self._subgoal_kid_room:
            return
        self.subgoal = subgoal_room
        self._subgoal_kid_room = kid_room
        if subgoal_room < 0:
            self.subgoal_hops = 9999
            return
        n = self.n_edges
        safe = {}
        for i in range(n):
            if self.edge_trav[i] and not self.edge_fatal[i]:
                safe.setdefault(int(self.edge_src[i]), []).append(int(self.edge_dst[i]))
        d = bfs_dist(kid_room - 1, subgoal_room, safe)
        self.subgoal_hops = d if d >= 0 else 9999

    def map_graph(self, subgoal_room=-1):
        """Returns the current room graph state. Flips visited for kid's room.
        Delta-patches edge_fatal/risky for any changed fg/bg tiles, then reruns
        the full (cheap) BFS pass if any edge changed.
        """
        # ── Tier 2 delta patch ───────────────────────────────────────
        lv  = self.level
        fg  = np.frombuffer(lv.fg, dtype=np.uint8)
        bg  = np.frombuffer(lv.bg, dtype=np.uint8).astype(np.int16)
        sword_state = self.have_sword

        changed_fg = np.where(fg != self._cached_fg)[0]
        changed_bg = np.where(bg != self._cached_bg)[0]
        sword_changed = (sword_state != self._cached_have_sword)

        if changed_fg.size > 0 or changed_bg.size > 0 or sword_changed:
            self._graph_recompute_steps += 1
            edges_to_redo = set()
            for pos in changed_fg:
                edges_to_redo.update(self.static.fg_reverse_index.get(int(pos), ()))
            for pos in changed_bg:
                edges_to_redo.update(self.static.bg_reverse_index.get(int(pos), ()))

            for e in edges_to_redo:
                src = int(self.edge_src[e])
                dst = int(self.edge_dst[e])
                f, ri = classify_fall(src, dst, lv, fg, bg)
                self.edge_fatal[e] = f
                self.edge_risky[e] = ri

            if edges_to_redo or sword_changed:
                self._run_bfs_pass()

            # Decrement/increment room_fg_counts for changed fg cells only
            _CRITICAL = np.array([2, 11, 4, 6, 15], dtype=np.uint8)
            for pos in changed_fg:
                room_idx = int(pos) // 30
                old_t = int(self._cached_fg[pos]) & 0x1F
                new_t = int(fg[pos]) & 0x1F
                for ci, t in enumerate(_CRITICAL):
                    if old_t == t:
                        self.room_fg_counts[room_idx, ci] -= 1
                    if new_t == t:
                        self.room_fg_counts[room_idx, ci] += 1

            self._cached_fg[changed_fg] = fg[changed_fg]
            self._cached_bg[changed_bg] = bg[changed_bg]
            self._cached_have_sword = sword_state
        else:
            self._graph_skip_steps += 1

        room = int(self.kid.room)
        if 1 <= room <= 24:
            self.visited[room - 1] = 1

        self._refresh_subgoal(subgoal_room)

        return {
            "edge_src":      self.edge_src,
            "edge_dst":      self.edge_dst,
            "edge_fatal":    self.edge_fatal,
            "edge_risky":    self.edge_risky,
            "edge_trav":     self.edge_trav,
            "edge_mask":     self.edge_mask,
            "subgoal_hops": np.int64(self.subgoal_hops),
        }

    def build_graph_obs(self, subgoal_room=-1):
        """Stack the 6 edge arrays into a (6, MAX_ADJ) uint8 observation array."""
        g = self.map_graph(subgoal_room=subgoal_room)
        return np.stack([g["edge_src"], g["edge_dst"], g["edge_fatal"], g["edge_risky"],
                         g["edge_trav"], g["edge_mask"]], axis=0).astype(np.uint8)


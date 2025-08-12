import env1
from obs_builder import bfs_dist

e = env1.PoPEnv(headless=True)
e.reset()

lv = e.data.level

def _tile_is_fall_open(tile, bg_val):
    return tile in (0, 4)

def _tile_is_hwall_open(tile, bg_val):
    return tile not in (20, 21, 23, 24, 9)

import obs_builder
obs_builder._tile_is_open = _tile_is_hwall_open # default for static rebuild

# Monkeypatch classify_fall and classify_hwall to use the correct helper
def classify_fall_fixed(src, dst, lv, fg, bg):
    if lv.roomlinks[src].down != dst + 1:
        return 0, 0, 0
    src_base = src * 30
    empty_cols = [
        c for c in range(10)
        if _tile_is_fall_open(fg[src_base + 20 + c] & 0x1F, int(bg[src_base + 20 + c]))
    ]
    if not empty_cols:
        return 0, 0, 1

    max_rows = 1
    for c in empty_cols:
        rows = 1
        cur = dst
        while rows < 10:
            blocked = False
            for row_idx in range(3):
                pos = cur * 30 + row_idx * 10 + c
                if not _tile_is_fall_open(fg[pos] & 0x1F, int(bg[pos])):
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

    if max_rows >= 4:
        return 1, 0, 0
    if max_rows == 3:
        return 0, 1, 0
    return 0, 0, 0

def classify_hwall_fixed(src, dst, lv, fg, bg):
    lnk = lv.roomlinks[src]
    if lnk.right == dst + 1:
        src_col, dst_col = 9, 0
    elif lnk.left == dst + 1:
        src_col, dst_col = 0, 9
    else:
        return 0
    def _col_open(room, col):
        for row in range(3):
            pos = room * 30 + row * 10 + col
            if _tile_is_hwall_open(fg[pos] & 0x1F, int(bg[pos])):
                return True
        return False
    if not _col_open(src, src_col) or not _col_open(dst, dst_col):
        return 1
    return 0

obs_builder.classify_fall = classify_fall_fixed
obs_builder.classify_hwall = classify_hwall_fixed

obs_builder._LEVEL1_STATIC = None
g = e.obs_builder.build_map_graph()

n = e.obs_builder.n_edges
safe = {}
for i in range(n):
    if e.obs_builder.edge_trav[i] and not e.obs_builder.edge_fatal[i]:
        src = int(e.obs_builder.edge_src[i])
        dst = int(e.obs_builder.edge_dst[i])
        safe.setdefault(src, []).append(dst)

print("Split Logic Graph Adjacency:")
for k in sorted(safe.keys()):
    print(f"Room {k+1} -> {[x+1 for x in safe[k]]}")

print("\nDistances from Room 1 (Start):")
for target in range(24):
    d = bfs_dist(0, target, safe)
    print(f"Room 1 -> Room {target+1}: {d}")


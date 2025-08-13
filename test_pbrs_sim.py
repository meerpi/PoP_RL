"""Standalone PBRS simulation — path-based, no BFS.

Simulates merge_pbrs_hints + _build_pbrs_map exactly as the live code does,
using the real memory.json. Injects synthetic paths (as if agent reached
sword room 15 and guard room 3) to show the PBRS gradient is correct.

Usage:
    .venv/bin/python PrincipiaDev/test_pbrs_sim.py
"""
import json

MEMORY_PATH = "PrincipiaDev/runs/PoP_Grid__agent1__32__1785097842/memory.json"
_PBRS_LAMBDA = 10.0
GAMMA = 0.9993

# ── replicate agent1.merge_pbrs_hints ────────────────────────────────────────

def merge_pbrs_hints(memory_list):
    best = None
    for mem in memory_list:
        paths = mem.get("paths", {})
        for key in ("to_guard", "to_sword_reversed"):
            p = paths.get(key)
            if p and (best is None or len(p) < len(best)):
                best = p
    return {"path": best if best is not None else []}

# ── replicate env1._build_pbrs_map ───────────────────────────────────────────

def build_pbrs_map(hint):
    path = hint.get("path", [])
    if len(path) < 2:
        return {}
    n = len(path)
    return {r: -_PBRS_LAMBDA * (n - 1 - i) for i, r in enumerate(path)}

def pbrs_delta(phi, prev_phi, next_room, k):
    phi_new = phi.get(next_room, prev_phi)
    return GAMMA**k * phi_new - prev_phi, phi_new

# ── load memory and inject synthetic paths ────────────────────────────────────

with open(MEMORY_PATH) as f:
    raw = json.load(f)
mem = raw[0] if isinstance(raw, list) else raw
mem.setdefault("paths", {"to_sword": None, "to_guard": None, "to_sword_reversed": None})

SWORD_ROOM = 15
GUARD_ROOM = 3

# Simulate: agent walked 1->2->6->8->7->20->12->15 to get sword
path_to_sword = [1, 2, 6, 8, 7, 20, 12, 15]
mem["paths"]["to_sword"] = path_to_sword
mem["paths"]["to_sword_reversed"] = list(reversed(path_to_sword))

# No guard path yet — agent hasn't reached guard with sword in this run
mem["paths"]["to_guard"] = None

print("=" * 60)
print("PHASE 1 — No path_to_guard yet, using reversed path_to_sword")
print("=" * 60)

hint = merge_pbrs_hints([mem])
phi  = build_pbrs_map(hint)
print(f"\nPBRS path used: {hint['path']}")
print(f"\nPotential map:")
for r in sorted(phi):
    hops = int(-phi[r] / _PBRS_LAMBDA)
    print(f"  room {r:2d}  Phi={phi[r]:6.1f}  ({hops} steps from guard)")

prev_phi = phi.get(SWORD_ROOM, 0.0)
print(f"\nSword pickup room {SWORD_ROOM}: prev_phi = {prev_phi:.1f}")

K = 9

def simulate_path(label, rooms, phi, prev_phi_start, k=K):
    prev = prev_phi_start
    print(f"\n  --- {label} ---")
    for room in rooms:
        delta, phi_new = pbrs_delta(phi, prev, room, k)
        tag = "  [off-path, no delta]" if room not in phi else ""
        print(f"  -> room {room:2d}  Phi={phi_new:7.1f}  delta={delta:+7.3f}{tag}")
        prev = phi_new

simulate_path("Along reversed sword path (correct direction → guard room 2→1 end)",
              [12, 20, 7, 8, 6, 2, 1], phi, prev_phi)
simulate_path("Wrong direction (deeper into sword path)",
              [20, 12, 7, 8], phi, prev_phi)

# ── Now simulate: agent found guard room 3 while carrying sword ───────────────
# path_sword_to_guard = [15, 12, 20, 7, 8, 6, 2, 3]
path_sword_to_guard = [15, 12, 20, 7, 8, 6, 2, 3]
mem["paths"]["to_guard"] = path_sword_to_guard

print("\n" + "=" * 60)
print("PHASE 2 — path_to_guard now known (agent reached guard with sword)")
print("=" * 60)

hint2 = merge_pbrs_hints([mem])
phi2  = build_pbrs_map(hint2)
print(f"\nPBRS path used: {hint2['path']}")
print(f"\nPotential map:")
for r in sorted(phi2):
    hops = int(-phi2[r] / _PBRS_LAMBDA)
    print(f"  room {r:2d}  Phi={phi2[r]:6.1f}  ({hops} steps from guard room {path_sword_to_guard[-1]})")

prev_phi2 = phi2.get(SWORD_ROOM, 0.0)
print(f"\nSword pickup room {SWORD_ROOM}: prev_phi = {prev_phi2:.1f}  ({int(-prev_phi2/_PBRS_LAMBDA)} steps from guard)")
simulate_path("Correct path toward guard room 3",
              [12, 20, 7, 8, 6, 2, 3], phi2, prev_phi2)
simulate_path("Off-path / backtrack",
              [10, 19], phi2, prev_phi2)

print("\n" + "=" * 60)
print("EXPECTED: toward guard => large positive delta at each new hop level")
print("          off-path     => 0 delta (stays at prev_phi)")
print("=" * 60)

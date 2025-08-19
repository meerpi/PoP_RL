#!/usr/bin/env python3
"""
corrected_test_harness.py — Corrected Level 1 Empirical Verification Harness

Fixes applied vs previous harness (see FIX-1 through FIX-4 comments):

FIX-1: Full state reset + readback confirmation before every individual test.
FIX-2: Explicit final_room == declared_dst_room assertion; 'unexpected_destination'
        is its own status, never collapsed into the passable/fatal booleans.
        room_id == 0 is the death sentinel: logged as died=True, never as a room number.
FIX-3: verification_status on each edge computed from actual per-edge results
        after rerun, not copied forward from the static graph.
FIX-4: Report JSON excerpts are produced by json.dumps() on actual result records —
        never retyped or reconstructed from memory.

Non-negotiable engine facts (never re-derived):
- room_id == 0 is the engine death sentinel.
- R8->R11 (down) is confirmed fatal. Included in sweep; result is expected.

Output files:
  level1_empirical_results_v2.json
  level1_traversability_graph_v2.json
  level1_verification_report_v2.md
"""

import sys, json, copy, datetime
sys.path.insert(0, './PrincipiaDev')
import env1

GRAPH_PATH    = './PrincipiaDev/level1_traversability_graph.json'
RESULTS_PATH  = './PrincipiaDev/level1_empirical_results_v2.json'
GRAPH_V2_PATH = './PrincipiaDev/level1_traversability_graph_v2.json'
REPORT_PATH   = './PrincipiaDev/level1_verification_report_v2.md'

# Triage order: problematic edges run first so problems surface early
TRIAGE_EDGES = [
    (1,  2,  'down'),   # "always fatal" cluster - known non-fatal, was flagged v1
    (7,  14, 'down'),
    (12, 19, 'down'),
    (20, 4,  'down'),
    (15, 10, 'down'),   # "never departs" cluster
    (22, 15, 'down'),
    (23, 20, 'down'),
]


# ── FIX-1: reset helper called before EVERY individual test ──────────
def reset_and_confirm(env, src_room, pos, max_retries=3):
    """
    Full level restart + teleport + readback confirmation.
    Returns (ok, confirmed_hp). ok=False -> flag as RESET_FAILED, do not run.
    """
    for _ in range(max_retries):
        env.start_room = src_room
        env.start_pos = pos
        env.reset()  # full restart via rl_request_restart_level — restores hp/alive
        env._refresh()
        ok = (
            int(env.data.kid.room) == src_room
            and int(env.data.hitp_curr) > 0
            and int(env.data.kid.alive) != 0
        )
        if ok:
            return True, int(env.data.hitp_curr)
        for _ in range(5):
            env._wait_frames(1)
        env._refresh()
    return False, 0


def kid_is_dead(env):
    """FIX-2: room==0 is the death sentinel; also check hp and alive."""
    if int(env.data.kid.room) == 0:
        return True
    if env.data.hitp_curr == 0:
        return True
    # alive > 0 in the engine means death animation state
    if int(env.data.kid.alive) > 0:
        return True
    return False


def make_reset_failed(src, dst, d, pos, row_or_col, label, predicted_passable):
    return {
        "edge": f"R{src}->R{dst} ({d})",
        "boundary_index": label,
        "test_conditions": f"spawn R{src} pos {pos}, dir={d}",
        "reset_confirmed": False,
        "status": "RESET_FAILED",
        "died": False,
        "predicted_passable": predicted_passable,
        "observed_passable": None,
        "observed_fatal": None,
        "final_room": None,
        "declared_dst_room": dst,
        "matches_prediction": False,
    }


def run_horizontal(env, src, dst, direction, row, pred_passable):
    """One left/right boundary test. FIX-1,2 applied."""
    act_id = 3 if direction == 'right' else 4
    col    = 9 if direction == 'right' else 0
    pos    = row * 10 + col

    # FIX-1
    ok, start_hp = reset_and_confirm(env, src, pos)
    if not ok:
        return make_reset_failed(src, dst, direction, pos, f"row {row}", f"row {row}", pred_passable)

    observed_passable = False
    died = False
    unexpected_dest = False
    final_room = src

    for _ in range(4):
        env.step([act_id, 4])
        env._refresh()
        curr = int(env.data.kid.room)
        # FIX-2: room==0 is death
        if curr == 0 or kid_is_dead(env):
            died = True
            final_room = 0
            break
        if curr == dst:
            observed_passable = True
            final_room = curr
            break
        if curr != src and curr != 0:
            unexpected_dest = True
            final_room = curr
            break

    if not died and not observed_passable and not unexpected_dest:
        env._refresh()
        cr = int(env.data.kid.room)
        final_room = cr if cr != 0 else src

    # FIX-2: unexpected_destination is its own status
    if died:
        status = "FATAL"
    elif unexpected_dest:
        status = "UNEXPECTED_DESTINATION"
    elif observed_passable:
        status = "PASSED"
    else:
        status = "BLOCKED"

    matches = (
        (not died and not unexpected_dest and observed_passable == pred_passable)
        or (died and not pred_passable)  # unexpected death still a mismatch
    )
    # Correct the above: only count as match if prediction was fatal too
    matches = (
        (not died and not unexpected_dest and observed_passable == pred_passable)
    )

    return {
        "edge": f"R{src}->R{dst} ({direction})",
        "boundary_index": f"row {row}",
        "test_conditions": f"spawn R{src} pos {pos}, dir={direction}",
        "reset_confirmed": True,
        "status": status,
        "died": died,
        "predicted_passable": pred_passable,
        "observed_passable": observed_passable,
        "observed_fatal": died,
        "final_room": final_room,
        "declared_dst_room": dst,
        "matches_prediction": matches,
    }


def run_down(env, src, dst, col, passable_idxs, pred_fatal, risky_hp_cost):
    """One down boundary test. FIX-1,2 applied."""
    pos = 20 + col

    # FIX-1
    ok, start_hp = reset_and_confirm(env, src, pos)
    if not ok:
        return make_reset_failed(src, dst, 'down', pos, f"col {col}", f"col {col}",
                                 col in passable_idxs)

    max_fall_y = 0
    observed_passable = False
    died = False
    unexpected_dest = False
    final_room = src
    hp_after = start_hp

    for _ in range(15):
        env.step([0, 0])
        env._refresh()
        curr = int(env.data.kid.room)
        max_fall_y = max(max_fall_y, int(env.data.kid.fall_y))

        # FIX-2: room==0 is the death sentinel
        if curr == 0 or kid_is_dead(env):
            died = True
            hp_after = int(env.data.hitp_curr)
            final_room = 0
            break

        if curr == dst:
            observed_passable = True
            final_room = curr
            hp_after = int(env.data.hitp_curr)
            # Don't break — continue to detect deferred death (spikes etc.)
        elif curr != src and curr != 0:
            unexpected_dest = True
            final_room = curr
            hp_after = int(env.data.hitp_curr)
            break

    if not died:
        env._refresh()
        hp_after = int(env.data.hitp_curr)
        if kid_is_dead(env):
            died = True
            final_room = 0

    expected_passable = col in passable_idxs

    # FIX-2
    if died:
        status = "FATAL"
    elif unexpected_dest:
        status = "UNEXPECTED_DESTINATION"
    elif observed_passable:
        status = "PASSED"
    else:
        status = "BLOCKED"

    hp_loss = start_hp - hp_after if not died else start_hp

    if pred_fatal and died:
        matches = True
    elif not pred_fatal and not died and not unexpected_dest:
        matches = (observed_passable == expected_passable)
    else:
        matches = False  # unexpected_destination or fatal when not predicted

    return {
        "edge": f"R{src}->R{dst} (down)",
        "boundary_index": f"col {col}",
        "test_conditions": f"spawn R{src} pos {pos}, gravity fall, max_fall_y={max_fall_y}",
        "reset_confirmed": True,
        "status": status,
        "died": died,
        "predicted_passable": expected_passable,
        "predicted_fatal": pred_fatal,
        "observed_passable": observed_passable,
        "observed_fatal": died,
        "final_room": final_room,
        "declared_dst_room": dst,
        "hp_loss": hp_loss,
        "matches_prediction": matches,
    }


def run_up(env, src, dst, col, passable_idxs):
    """One up boundary test. FIX-1,2 applied."""
    pos = col

    # FIX-1
    ok, start_hp = reset_and_confirm(env, src, pos)
    if not ok:
        return make_reset_failed(src, dst, 'up', pos, f"col {col}", f"col {col}",
                                 col in passable_idxs)

    observed_passable = False
    died = False
    unexpected_dest = False
    final_room = src

    for _ in range(10):
        env.step([1, 0])  # action 1 = UP
        env._refresh()
        curr = int(env.data.kid.room)
        # FIX-2
        if curr == 0 or kid_is_dead(env):
            died = True
            final_room = 0
            break
        if curr == dst:
            observed_passable = True
            final_room = curr
            break
        if curr != src and curr != 0:
            unexpected_dest = True
            final_room = curr
            break

    expected_passable = col in passable_idxs

    if died:
        status = "FATAL"
    elif unexpected_dest:
        status = "UNEXPECTED_DESTINATION"
    elif observed_passable:
        status = "PASSED"
    else:
        status = "BLOCKED"

    matches = (not died and not unexpected_dest
               and observed_passable == expected_passable)

    return {
        "edge": f"R{src}->R{dst} (up)",
        "boundary_index": f"col {col}",
        "test_conditions": f"spawn R{src} pos {pos}, action=up",
        "reset_confirmed": True,
        "status": status,
        "died": died,
        "predicted_passable": expected_passable,
        "observed_passable": observed_passable,
        "observed_fatal": died,
        "final_room": final_room,
        "declared_dst_room": dst,
        "matches_prediction": matches,
    }


def triage_r13_r18(env):
    """R13->R18 roomlink cross-check (static, no engine action)."""
    lv = env.data.level
    r13 = lv.roomlinks[12]
    r18 = lv.roomlinks[17]
    r19 = lv.roomlinks[18]
    return {
        "investigation": "R13->R18 cross-check",
        "R13": {"left": int(r13.left), "right": int(r13.right),
                "up": int(r13.up), "down": int(r13.down)},
        "R18": {"left": int(r18.left), "right": int(r18.right),
                "up": int(r18.up), "down": int(r18.down)},
        "R19": {"left": int(r19.left), "right": int(r19.right),
                "up": int(r19.up), "down": int(r19.down)},
        "R13_down": int(r13.down),
        "R18_up": int(r18.up),
        "R13_R18_is_symmetric": int(r13.down) == 18 and int(r18.up) == 13,
        "R18_down": int(r18.down),
        "R19_up": int(r19.up),
        "R18_R19_is_phantom": int(r19.up) != 18,
        "note": (
            "R13->R18(down) symmetric iff R18.up==13. "
            "R18->R19(down) already confirmed phantom: R19.up=12, not 18. "
            "R13 itself is unreachable from normal play (phantom overlay cluster)."
        )
    }


# FIX-3: compute, never copy
def compute_verification_status(edge_key, all_results):
    tests = [t for t in all_results if t["edge"] == edge_key]
    if not tests:
        return "untested", []
    if any(t.get("status") == "RESET_FAILED" for t in tests):
        return "reset_failed", [t for t in tests if t.get("status") == "RESET_FAILED"]
    failures = [t for t in tests if not t.get("matches_prediction", False)]
    if not failures:
        return "empirically_confirmed", []
    if len(failures) == len(tests):
        return "all_tests_failed", failures
    return "conflicting", failures


def generate_report(results_v2, graph_v2):
    """
    FIX-4: all JSON excerpts produced by json.dumps() on actual result records.
    No excerpt is ever retyped.
    """
    from collections import Counter
    total   = results_v2["total_tests"]
    matched = results_v2["matched_tests"]
    rate    = results_v2["match_rate"]
    by_dir  = results_v2["by_direction"]
    tests   = results_v2["tests"]
    ts      = results_v2["generated_at"]

    lines = [
        "# Level 1 Traversability Graph — Corrected Empirical Verification Report v2",
        f"\nGenerated: {ts}",
        "\n## Global Summary Statistics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total tests run | {total} |",
        f"| Tests matched | {matched} |",
        f"| Overall match rate | {100*rate:.1f}% |",
    ]
    for d, s in by_dir.items():
        pct = f"{100*s['match_rate']:.0f}%" if s['total'] else "n/a"
        lines.append(f"| {d} match rate | {pct} ({s['matched']}/{s['total']}) |")

    status_counts = Counter(t.get("status", "UNKNOWN") for t in tests)
    lines += ["\n### Status Breakdown\n", "| Status | Count |", "|--------|-------|"]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines += ["\n## Triage: R13->R18 Cross-Check\n", "```json",
              json.dumps(results_v2["triage_investigation"]["R13_R18_cross_check"], indent=2),
              "```"]

    lines.append("\n## Non-Confirmed Edges\n")
    for edge in graph_v2["edges"]:
        vs = edge["verification_status"]
        if vs == "empirically_confirmed":
            continue
        ek = f"R{edge['src_room']}->R{edge['dst_room']} ({edge['direction']})"
        lines.append(f"### {ek}  `{vs}`\n")
        lines.append(f"- fatal={edge['fatal']}, risky_hp_cost={edge['risky_hp_cost']}, "
                     f"passable_indexes={edge['passable_indexes']}")
        edge_tests = [t for t in tests if t["edge"] == ek]
        failing = [t for t in edge_tests if not t.get("matches_prediction", True)]
        if failing:
            lines.append("\n**Failing records (direct copy from results file):**\n")
            for t in failing:
                # FIX-4: json.dumps on the actual object, never retyped
                lines += ["```json", json.dumps(t, indent=2), "```"]

    confirmed = [e for e in graph_v2["edges"]
                 if e["verification_status"] == "empirically_confirmed"]
    lines += [
        f"\n## Empirically Confirmed Edges\n",
        f"{len(confirmed)} of {len(graph_v2['edges'])} edges passed all tests.\n",
        "| Edge | Direction | Fatal | Risky HP | Passable Indexes |",
        "|------|-----------|-------|----------|-----------------|",
    ]
    for e in confirmed:
        lines.append(f"| R{e['src_room']}->R{e['dst_room']} | {e['direction']} | "
                     f"{e['fatal']} | {e['risky_hp_cost']} | {e['passable_indexes']} |")

    lines += [
        "\n## PENDING Items\n",
        "- **Shift-grab timing across 2-row drops**: requires frame-perfect multi-step "
        "macro; not automatable with no-op sweeps. PENDING for manual follow-up.",
        "- **Rooms 13, 18, 24**: unreachable overlay rooms; no inbound symmetric "
        "roomlinks from normal play. Engine tests not meaningful.",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def run():
    with open(GRAPH_PATH) as f:
        graph_data = json.load(f)
    edges = graph_data["edges"]

    triage_keys = {(s, d, di) for s, d, di in TRIAGE_EDGES}
    ordered_edges = (
        [e for e in edges if (e["src_room"], e["dst_room"], e["direction"]) in triage_keys]
        + [e for e in edges if (e["src_room"], e["dst_room"], e["direction"]) not in triage_keys]
    )

    env = env1.PoPEnv(headless=True)
    env.reset()
    r13r18 = triage_r13_r18(env)

    all_results = []
    total = matched = 0
    by_dir = {"left": [0, 0], "right": [0, 0], "up": [0, 0], "down": [0, 0]}

    print(f"=== Corrected Level 1 Empirical Verification (v2) ===")
    print(f"{len(ordered_edges)} edges, full column/row sweep, triage-first.\n")

    for edge in ordered_edges:
        src = edge["src_room"]
        dst = edge["dst_room"]
        d   = edge["direction"]
        pidxs = edge["passable_indexes"]
        pfatal = edge["fatal"]
        rhp   = edge.get("risky_hp_cost", 0)
        ek    = f"R{src}->R{dst} ({d})"
        er    = []

        if d in ("left", "right"):
            for row in range(3):
                pred = (row in pidxs) and not pfatal
                er.append(run_horizontal(env, src, dst, d, row, pred))
        elif d == "down":
            for col in range(10):
                er.append(run_down(env, src, dst, col, pidxs, pfatal, rhp))
        elif d == "up":
            for col in range(10):
                er.append(run_up(env, src, dst, col, pidxs))

        for r in er:
            total += 1
            by_dir[d][0] += 1
            if r.get("matches_prediction"):
                matched += 1
                by_dir[d][1] += 1
        all_results.extend(er)

        ok_count = sum(1 for r in er if r.get("matches_prediction"))
        statuses = {r.get("status") for r in er}
        print(f"  {ek}: {ok_count}/{len(er)} — {statuses}")

    env.close()
    rate = matched / total if total else 0.0
    print(f"\n=== DONE: {matched}/{total} ({100*rate:.1f}%) ===")
    for dk, (cnt, ok) in by_dir.items():
        r = f"{100*ok//cnt}%" if cnt else "n/a"
        print(f"  {dk}: {ok}/{cnt} ({r})")

    results_v2 = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_tests": total,
        "matched_tests": matched,
        "match_rate": rate,
        "by_direction": {
            k: {"total": v[0], "matched": v[1],
                "match_rate": v[1] / v[0] if v[0] else 0.0}
            for k, v in by_dir.items()
        },
        "triage_investigation": {"R13_R18_cross_check": r13r18},
        "tests": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_v2, f, indent=2)
    print(f"Results -> {RESULTS_PATH}")

    # FIX-3: computed verification_status
    graph_v2 = copy.deepcopy(graph_data)
    for edge in graph_v2["edges"]:
        ek = f"R{edge['src_room']}->R{edge['dst_room']} ({edge['direction']})"
        status, failing = compute_verification_status(ek, all_results)
        edge["verification_status"] = status
        edge["verification_failing_records"] = [
            {"boundary_index": r["boundary_index"],
             "status": r.get("status"),
             "final_room": r.get("final_room"),
             "died": r.get("died")}
            for r in failing
        ]

    with open(GRAPH_V2_PATH, "w") as f:
        json.dump(graph_v2, f, indent=2)
    print(f"Graph v2 -> {GRAPH_V2_PATH}")

    generate_report(results_v2, graph_v2)
    print(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    run()

from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

SW_DIR = RESULTS_DIR / "sw_fixed"
HW_NMS_DIR = RESULTS_DIR / "hw_local_nms"
HW_ANMS_DIR = RESULTS_DIR / "hw_anms"

K_TOP = 2048


def read_kps(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y, s = line.split()[:3]
            out.append((int(x), int(y), int(s)))
    return out


def by_xy(kps):
    d = defaultdict(list)
    for x, y, s in kps:
        d[(x, y)].append(s)
    return d


def final_by_xy(kps):
    d = {}
    for x, y, s in kps:
        d[(x, y)] = s
    return d


def kth_score(kps, K):
    scores = sorted((s for _, _, s in kps), reverse=True)
    if len(scores) < K:
        return scores[-1] if scores else 0
    return scores[K - 1]


stats = defaultdict(int)

print()
print("ANMS DOWNSTREAM SELECTION AUDIT")
print("=" * 96)

for seq in range(1, 9):
    for img in range(1, 7):

        sw_nms = read_kps(
            SW_DIR /
            f"seq{seq}_img{img}_thr100_L8_sw_fixed_local_nms_kps.txt"
        )

        hw_nms = read_kps(
            HW_NMS_DIR /
            f"seq{seq}_img{img}_thr100_L8_K2048_N1000_c0.90_merged_nms_kps.txt"
        )

        sw_final = read_kps(
            SW_DIR /
            f"seq{seq}_img{img}_thr100_L8_K2048_N1000_c0.90_sw_fixed_anms_kps.txt"
        )

        hw_final = read_kps(
            HW_ANMS_DIR /
            f"seq{seq}_img{img}_thr100_L8_K2048_N1000_c0.90_anms_kps.txt"
        )

        sw_nms_xy = by_xy(sw_nms)
        hw_nms_xy = by_xy(hw_nms)

        sw_f = final_by_xy(sw_final)
        hw_f = final_by_xy(hw_final)

        sw_only = set(sw_f) - set(hw_f)
        hw_only = set(hw_f) - set(sw_f)

        sw_cut = kth_score(sw_nms, K_TOP)
        hw_cut = kth_score(hw_nms, K_TOP)

        # ------------------------------------------------------------
        # SW final coordinate missing from HW final
        # ------------------------------------------------------------
        for xy in sw_only:
            s = sw_f[xy]
            opp_scores = hw_nms_xy.get(xy, [])

            stats["sw_only_total"] += 1

            if not opp_scores:
                stats["sw_upstream_absent"] += 1
                continue

            stats["sw_present_hw_nms"] += 1

            if s in opp_scores:
                stats["sw_exact_tuple_hw_nms"] += 1

                if s > hw_cut:
                    stats["sw_exact_above_hw_topk_cut"] += 1
                elif s == hw_cut:
                    stats["sw_exact_at_hw_topk_cut"] += 1
                else:
                    stats["sw_exact_below_hw_topk_cut"] += 1
            else:
                stats["sw_xy_only_score_diff"] += 1

        # ------------------------------------------------------------
        # HW final coordinate missing from SW final
        # ------------------------------------------------------------
        for xy in hw_only:
            s = hw_f[xy]
            opp_scores = sw_nms_xy.get(xy, [])

            stats["hw_only_total"] += 1

            if not opp_scores:
                stats["hw_upstream_absent"] += 1
                continue

            stats["hw_present_sw_nms"] += 1

            if s in opp_scores:
                stats["hw_exact_tuple_sw_nms"] += 1

                if s > sw_cut:
                    stats["hw_exact_above_sw_topk_cut"] += 1
                elif s == sw_cut:
                    stats["hw_exact_at_sw_topk_cut"] += 1
                else:
                    stats["hw_exact_below_sw_topk_cut"] += 1
            else:
                stats["hw_xy_only_score_diff"] += 1


print("SW-final coordinates missing from HW final")
print("-------------------------------------------")
print("Total                         :", stats["sw_only_total"])
print("Absent already from HW NMS    :", stats["sw_upstream_absent"])
print("Present in HW NMS             :", stats["sw_present_hw_nms"])
print("  Same (x,y,score) tuple      :", stats["sw_exact_tuple_hw_nms"])
print("  Same xy, different score    :", stats["sw_xy_only_score_diff"])
print("    exact tuple > HW TopK cut :", stats["sw_exact_above_hw_topk_cut"])
print("    exact tuple = HW TopK cut :", stats["sw_exact_at_hw_topk_cut"])
print("    exact tuple < HW TopK cut :", stats["sw_exact_below_hw_topk_cut"])

print()
print("HW-final coordinates missing from SW final")
print("-------------------------------------------")
print("Total                         :", stats["hw_only_total"])
print("Absent already from SW NMS    :", stats["hw_upstream_absent"])
print("Present in SW NMS             :", stats["hw_present_sw_nms"])
print("  Same (x,y,score) tuple      :", stats["hw_exact_tuple_sw_nms"])
print("  Same xy, different score    :", stats["hw_xy_only_score_diff"])
print("    exact tuple > SW TopK cut :", stats["hw_exact_above_sw_topk_cut"])
print("    exact tuple = SW TopK cut :", stats["hw_exact_at_sw_topk_cut"])
print("    exact tuple < SW TopK cut :", stats["hw_exact_below_sw_topk_cut"])

print()
print("=" * 96)

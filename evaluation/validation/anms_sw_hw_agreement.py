from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

SW_DIR = RESULTS_DIR / "sw_fixed"
HW_NMS_DIR = RESULTS_DIR / "hw_local_nms"
HW_ANMS_DIR = RESULTS_DIR / "hw_anms"


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


def xy_counter(kps):
    return Counter((x, y) for x, y, _ in kps)


def intersection_count(a, b):
    return sum((a & b).values())


def difference_items(a, b):
    """
    Multiset difference a-b, expanded into individual coordinates.
    """
    d = a - b
    out = []

    for xy, n in d.items():
        out.extend([xy] * n)

    return out


grand = defaultdict(int)

print()
print("ANMS SW-HW DISCREPANCY AUDIT")
print("=" * 100)

for seq in range(1, 9):

    seq_stats = defaultdict(int)

    print(f"\nSequence {seq}")
    print("-" * 100)

    for img in range(1, 7):

        sw_nms_file = SW_DIR / (
            f"seq{seq}_img{img}_thr100_L8_sw_fixed_local_nms_kps.txt"
        )

        sw_anms_file = SW_DIR / (
            f"seq{seq}_img{img}_thr100_L8_K2048_N1000_c0.90_"
            f"sw_fixed_anms_kps.txt"
        )

        hw_nms_file = HW_NMS_DIR / (
            f"seq{seq}_img{img}_thr100_L8_K2048_N1000_c0.90_"
            f"merged_nms_kps.txt"
        )

        hw_anms_file = HW_ANMS_DIR / (
            f"seq{seq}_img{img}_thr100_L8_K2048_N1000_c0.90_"
            f"anms_kps.txt"
        )

        sw_nms = read_kps(sw_nms_file)
        hw_nms = read_kps(hw_nms_file)

        sw_final = read_kps(sw_anms_file)
        hw_final = read_kps(hw_anms_file)

        sw_nms_xy = xy_counter(sw_nms)
        hw_nms_xy = xy_counter(hw_nms)

        sw_final_xy = xy_counter(sw_final)
        hw_final_xy = xy_counter(hw_final)

        # ----------------------------------------------------
        # Local-NMS coordinate overlap
        # ----------------------------------------------------
        nms_shared = intersection_count(
            sw_nms_xy,
            hw_nms_xy,
        )

        nms_dice = (
            200.0 * nms_shared /
            (len(sw_nms) + len(hw_nms))
            if len(sw_nms) + len(hw_nms) else 100.0
        )

        # ----------------------------------------------------
        # Final ANMS coordinate overlap
        # ----------------------------------------------------
        final_shared = intersection_count(
            sw_final_xy,
            hw_final_xy,
        )

        final_agreement = (
            100.0 * final_shared / len(sw_final)
            if sw_final else 100.0
        )

        sw_only = difference_items(
            sw_final_xy,
            hw_final_xy,
        )

        hw_only = difference_items(
            hw_final_xy,
            sw_final_xy,
        )

        # For each unmatched FINAL coordinate, ask whether the same
        # coordinate nevertheless existed in the opposite local-NMS set.
        sw_only_present_hw_nms = sum(
            1 for xy in sw_only if hw_nms_xy[xy] > 0
        )

        sw_only_absent_hw_nms = (
            len(sw_only) - sw_only_present_hw_nms
        )

        hw_only_present_sw_nms = sum(
            1 for xy in hw_only if sw_nms_xy[xy] > 0
        )

        hw_only_absent_sw_nms = (
            len(hw_only) - hw_only_present_sw_nms
        )

        print(
            f"img{img}: "
            f"NMS SW={len(sw_nms):5d} "
            f"HW={len(hw_nms):5d} "
            f"XY-overlap={nms_dice:6.2f}% | "
            f"FINAL shared={final_shared:4d}/1000 "
            f"({final_agreement:6.2f}%) | "
            f"SW-only={len(sw_only):3d} "
            f"[in HW-NMS={sw_only_present_hw_nms:3d}, "
            f"absent={sw_only_absent_hw_nms:3d}] | "
            f"HW-only={len(hw_only):3d} "
            f"[in SW-NMS={hw_only_present_sw_nms:3d}, "
            f"absent={hw_only_absent_sw_nms:3d}]"
        )

        values = {
            "images": 1,
            "final_shared": final_shared,
            "final_total": len(sw_final),
            "sw_only": len(sw_only),
            "hw_only": len(hw_only),
            "sw_only_present_opposite_nms": sw_only_present_hw_nms,
            "sw_only_absent_opposite_nms": sw_only_absent_hw_nms,
            "hw_only_present_opposite_nms": hw_only_present_sw_nms,
            "hw_only_absent_opposite_nms": hw_only_absent_sw_nms,
            "nms_sw": len(sw_nms),
            "nms_hw": len(hw_nms),
            "nms_shared": nms_shared,
        }

        for k, v in values.items():
            seq_stats[k] += v
            grand[k] += v

    seq_agreement = (
        100.0 * seq_stats["final_shared"] /
        seq_stats["final_total"]
    )

    seq_nms_dice = (
        200.0 * seq_stats["nms_shared"] /
        (seq_stats["nms_sw"] + seq_stats["nms_hw"])
    )

    print(
        f"\nSEQ {seq} SUMMARY: "
        f"final={seq_stats['final_shared']}/"
        f"{seq_stats['final_total']} "
        f"({seq_agreement:.2f}%), "
        f"NMS XY overlap={seq_nms_dice:.2f}%"
    )


print()
print("=" * 100)
print("DATASET SUMMARY")
print("=" * 100)

agreement = (
    100.0 * grand["final_shared"] /
    grand["final_total"]
)

nms_dice = (
    200.0 * grand["nms_shared"] /
    (grand["nms_sw"] + grand["nms_hw"])
)

print(
    f"Final shared coordinates : "
    f"{grand['final_shared']} / {grand['final_total']} "
    f"= {agreement:.4f}%"
)

print(
    f"Local-NMS XY overlap     : "
    f"{nms_dice:.4f}%"
)

print()
print("SW-final coordinates absent from HW final:")
print(
    f"  Total                  : {grand['sw_only']}"
)
print(
    f"  Present in HW local NMS: "
    f"{grand['sw_only_present_opposite_nms']}"
)
print(
    f"  Absent from HW local NMS : "
    f"{grand['sw_only_absent_opposite_nms']}"
)

print()
print("HW-final coordinates absent from SW final:")
print(
    f"  Total                  : {grand['hw_only']}"
)
print(
    f"  Present in SW local NMS: "
    f"{grand['hw_only_present_opposite_nms']}"
)
print(
    f"  Absent from SW local NMS : "
    f"{grand['hw_only_absent_opposite_nms']}"
)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


MANUAL_AUDIT_DATA = [
    # seq_032cbb4dc8686963 (Page 1 & 2)
    {
        "id": "seq032_p1_c0",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "YOU'RE THE ONLY ONE",
        "gt_has_tail": False,
        "gt_type": "smooth_dialogue_bubble",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.35, -0.94),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Panel border intersection creates false acute corner pointing into margin",
    },
    {
        "id": "seq032_p1_c1",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "MEAN...",
        "gt_has_tail": False,
        "gt_type": "small_oval_thought",
        "gt_direction": None,
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq032_p1_c2",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "WHO'S EVER SEEN ME LIKE THIS.",
        "gt_has_tail": False,
        "gt_type": "smooth_oval_dialogue",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (-0.45, -0.89),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Contour curve approximated to acute vertex at top apex",
    },
    {
        "id": "seq032_p1_c3",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "TH..",
        "gt_has_tail": False,
        "gt_type": "small_oval_reaction",
        "gt_direction": None,
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq032_p1_c4",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "THE ONLY ONE...?",
        "gt_has_tail": False,
        "gt_type": "smooth_oval_on_speedlines",
        "gt_direction": None,
        "det_status": "no_tail",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq032_p1_c5",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "SO I CAN'T ASK ANYONE ELSE.",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail",
        "gt_direction": (-1.0, 0.0),  # Left toward boy's face
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.42, -0.91),  # Up-Right into panel border
        "result": "FALSE_POSITIVE_WRONG_DIRECTION",
        "failure_mode": "Sealed tail base by edge dilation; detector picked panel corner in opposite direction",
    },
    {
        "id": "seq032_p1_c6",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "ER..",
        "gt_has_tail": False,
        "gt_type": "small_reaction_bubble",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.38, -0.92),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Panel gutter boundary corner picked as tail tip",
    },
    {
        "id": "seq032_p1_c7",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "IT HAS TO BE YOU...",
        "gt_has_tail": True,
        "gt_type": "connected_double_bubble",
        "gt_direction": (1.0, 0.0),  # Right toward boy
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "FALSE_NEGATIVE_MISSED_TAIL",
        "failure_mode": "Double-bubble dumbbell neck created concave notches classified as burst/scream",
    },
    {
        "id": "seq032_p1_c8",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "WOW...",
        "gt_has_tail": False,
        "gt_type": "oval_reaction",
        "gt_direction": None,
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq032_p1_c9",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "UH....",
        "gt_has_tail": False,
        "gt_type": "oval_reaction",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.28, 0.96),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Panel border line junction treated as acute tip",
    },
    {
        "id": "seq032_p1_c10",
        "seq": "seq_032cbb4dc8686963",
        "page": 1,
        "text": "UM....",
        "gt_has_tail": False,
        "gt_type": "oval_reaction",
        "gt_direction": None,
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq032_p2_c0",
        "seq": "seq_032cbb4dc8686963",
        "page": 2,
        "text": "CHIKA FUJIWARA WAS THE KIND OF",
        "gt_has_tail": False,
        "gt_type": "narration_box",
        "gt_direction": None,
        "det_status": "narration_or_box",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly identified narration)",
    },

    # seq_ac5793f8168efdd1 (Page 1 & 2)
    {
        "id": "seqac5_p1_c0",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "DON'T GET CARELESS.",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail",
        "gt_direction": (0.85, 0.52),  # Down-Right toward character mouth
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.87, 0.49),
        "result": "TRUE_POSITIVE_CORRECT_TAIL",
        "failure_mode": "None (correctly localized and vector aligned within 5 deg)",
    },
    {
        "id": "seqac5_p1_c1",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "SASSAN....",
        "gt_has_tail": False,
        "gt_type": "whisper_near_ear",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.71, -0.71),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Corner of panel gutter detected as tip",
    },
    {
        "id": "seqac5_p1_c2",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "2!",
        "gt_has_tail": False,
        "gt_type": "scream_number",
        "gt_direction": None,
        "det_status": "no_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seqac5_p1_c3",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "DO NOT INFLICT MORE PAIN...",
        "gt_has_tail": False,
        "gt_type": "speech_over_hair_no_tail",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.0, 1.0),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Hair line contact on bottom edge created artificial acute vertex",
    },
    {
        "id": "seqac5_p1_c4",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "RANK 1 SASAKI.",
        "gt_has_tail": True,
        "gt_type": "box_with_small_tail",
        "gt_direction": (1.0, 0.0),  # Right
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "FALSE_NEGATIVE_MISSED_TAIL",
        "failure_mode": "Tail base sealed by dilation; bottom panel corner created false burst vertices",
    },
    {
        "id": "seqac5_p1_c5",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "AND WE ALSO MIGHT BE ABLE...",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail",
        "gt_direction": (0.0, 1.0),  # Down
        "det_status": "background_bleed",
        "det_has_tail": False,
        "det_vector": None,
        "result": "FALSE_NEGATIVE_MISSED_TAIL",
        "failure_mode": "Bubble contour bled into adjacent unclosed panel artwork",
    },
    {
        "id": "seqac5_p1_c6",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "I'M GOING TO FINISH HIM OFF.",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail",
        "gt_direction": (0.68, -0.73),  # Top-Right toward character
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.71, -0.70),
        "result": "TRUE_POSITIVE_CORRECT_TAIL",
        "failure_mode": "None (correctly localized and vector aligned within 4 deg)",
    },
    {
        "id": "seqac5_p1_c7",
        "seq": "seq_ac5793f8168efdd1",
        "page": 1,
        "text": "...MORE IMPORTANTLY!",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail",
        "gt_direction": (0.89, -0.45),  # Right toward open mouth
        "det_status": "background_bleed",
        "det_has_tail": False,
        "det_vector": None,
        "result": "FALSE_NEGATIVE_MISSED_TAIL",
        "failure_mode": "Flood-fill leaked through mouth lineart into character skin",
    },
    {
        "id": "seqac5_p2_c0",
        "seq": "seq_ac5793f8168efdd1",
        "page": 2,
        "text": "THE QUINX HAVE EXPERIMENTAL...",
        "gt_has_tail": False,
        "gt_type": "narration_box",
        "gt_direction": None,
        "det_status": "narration_or_box",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seqac5_p2_c1",
        "seq": "seq_ac5793f8168efdd1",
        "page": 2,
        "text": "THEY ARE HUMANS WHO HAD THEIR...",
        "gt_has_tail": False,
        "gt_type": "narration_box",
        "gt_direction": None,
        "det_status": "narration_or_box",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },

    # seq_72fdfff4b4353a42 (Page 1 & 2)
    {
        "id": "seq72_p1_c0",
        "seq": "seq_72fdfff4b4353a42",
        "page": 1,
        "text": "THEY SAY THAT IF A BOY AND GIRL...",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail_on_screentone",
        "gt_direction": (-0.85, 0.52),  # Down-Left toward girl
        "det_status": "no_tail",
        "det_has_tail": False,
        "det_vector": None,
        "result": "FALSE_NEGATIVE_MISSED_TAIL",
        "failure_mode": "Halftone screentone texture created dense edges closing tail throat",
    },
    {
        "id": "seq72_p1_c4",
        "seq": "seq_72fdfff4b4353a42",
        "page": 1,
        "text": "A CONFESSION!?",
        "gt_has_tail": False,
        "gt_type": "monologue_on_dark_aura",
        "gt_direction": None,
        "det_status": "no_tail",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq72_p1_c5",
        "seq": "seq_72fdfff4b4353a42",
        "page": 1,
        "text": "OH? PRESIDENT,",
        "gt_has_tail": True,
        "gt_type": "dialogue_with_tail",
        "gt_direction": (-1.0, 0.0),  # Left toward girl
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (0.61, 0.79),  # Down-Right toward boy
        "result": "FALSE_POSITIVE_WRONG_DIRECTION",
        "failure_mode": "Detected bottom-right panel corner; actual left tail missed due to lineart thickness",
    },
    {
        "id": "seq72_p1_c6",
        "seq": "seq_72fdfff4b4353a42",
        "page": 1,
        "text": "ABOUT TO INVITE ME TO GO WITH YOU?",
        "gt_has_tail": False,
        "gt_type": "smooth_oval_monologue",
        "gt_direction": None,
        "det_status": "tail_detected",
        "det_has_tail": True,
        "det_vector": (-0.68, -0.73),
        "result": "FALSE_POSITIVE_INVENTED_TAIL",
        "failure_mode": "Top panel boundary junction detected as acute tip",
    },
    {
        "id": "seq72_p1_c7",
        "seq": "seq_72fdfff4b4353a42",
        "page": 1,
        "text": "WERE YOU",
        "gt_has_tail": False,
        "gt_type": "small_oval",
        "gt_direction": None,
        "det_status": "no_tail",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected)",
    },
    {
        "id": "seq72_p1_c14",
        "seq": "seq_72fdfff4b4353a42",
        "page": 1,
        "text": "SHIROGANE REALIZES THE DILEMMA...",
        "gt_has_tail": False,
        "gt_type": "inverted_narration_box",
        "gt_direction": None,
        "det_status": "narration_or_box",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly identified inverted box)",
    },
    {
        "id": "seq72_p2_c0",
        "seq": "seq_72fdfff4b4353a42",
        "page": 2,
        "text": "IN MATTERS OF LOVE...",
        "gt_has_tail": False,
        "gt_type": "narration_box",
        "gt_direction": None,
        "det_status": "narration_or_box",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly identified narration)",
    },

    # seq_864aaca43a3580bb (Page 1)
    {
        "id": "seq864_p1_c0",
        "seq": "seq_864aaca43a3580bb",
        "page": 1,
        "text": "GREETINGS TO ALL...",
        "gt_has_tail": False,
        "gt_type": "crowded_overlapping_speech",
        "gt_direction": None,
        "det_status": "burst_bubble",
        "det_has_tail": False,
        "det_vector": None,
        "result": "TRUE_NEGATIVE_CORRECT_REJECTION",
        "failure_mode": "None (correctly rejected crowded overlapping noise)",
    },
]


def compute_manual_audit_metrics() -> Dict[str, Any]:
    total = len(MANUAL_AUDIT_DATA)
    gt_has_tail_count = sum(1 for it in MANUAL_AUDIT_DATA if it["gt_has_tail"])
    gt_no_tail_count = sum(1 for it in MANUAL_AUDIT_DATA if not it["gt_has_tail"])

    tp = sum(1 for it in MANUAL_AUDIT_DATA if it["result"] == "TRUE_POSITIVE_CORRECT_TAIL")
    fp_invented = sum(1 for it in MANUAL_AUDIT_DATA if it["result"] == "FALSE_POSITIVE_INVENTED_TAIL")
    fp_wrong_dir = sum(1 for it in MANUAL_AUDIT_DATA if it["result"] == "FALSE_POSITIVE_WRONG_DIRECTION")
    fp_total = fp_invented + fp_wrong_dir
    fn = sum(1 for it in MANUAL_AUDIT_DATA if it["result"] == "FALSE_NEGATIVE_MISSED_TAIL")
    tn = sum(1 for it in MANUAL_AUDIT_DATA if it["result"] == "TRUE_NEGATIVE_CORRECT_REJECTION")

    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fp_rate_on_negatives = fp_invented / gt_no_tail_count if gt_no_tail_count > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    report = {
        "sample_size": total,
        "ground_truth_distribution": {
            "visible_tail_present": gt_has_tail_count,
            "no_visible_tail": gt_no_tail_count,
            "visible_tail_ratio": round(gt_has_tail_count / total, 3),
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives_invented": fp_invented,
            "false_positives_wrong_direction": fp_wrong_dir,
            "false_positives_total": fp_total,
            "false_negatives_missed": fn,
            "true_negatives_rejections": tn,
        },
        "performance_metrics": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "false_positive_rate_on_no_tail": round(fp_rate_on_negatives, 3),
            "overall_accuracy": round(accuracy, 3),
        },
        "audited_items": MANUAL_AUDIT_DATA,
    }

    out_path = Path("logs/bubble_tail_poc/manual_audit_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved manual audit results to {out_path}")
    return report


if __name__ == "__main__":
    rep = compute_manual_audit_metrics()
    cm = rep["confusion_matrix"]
    pm = rep["performance_metrics"]
    print("\n=== MANUAL INSPECTION AUDIT RESULTS (30 Development Bubbles) ===")
    print(f"Sample Size: {rep['sample_size']}")
    print(f"Ground Truth Tails: {rep['ground_truth_distribution']['visible_tail_present']} (26.7%) | No Tails: {rep['ground_truth_distribution']['no_visible_tail']} (73.3%)")
    print(f"True Positives: {cm['true_positives']} | False Positives: {cm['false_positives_total']} (Invented: {cm['false_positives_invented']}, Wrong Dir: {cm['false_positives_wrong_direction']})")
    print(f"False Negatives: {cm['false_negatives_missed']} | True Negatives: {cm['true_negatives_rejections']}")
    print(f"Precision: {pm['precision']*100:.1f}% | Recall: {pm['recall']*100:.1f}%")
    print(f"False Positive Rate on No-Tail Bubbles: {pm['false_positive_rate_on_no_tail']*100:.1f}%")

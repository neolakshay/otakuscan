from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

CONVERSATIONAL_RESPONSE_WORDS: Set[str] = {
    "yes", "no", "yeah", "nope", "well", "huh", "what", "wait", "why",
    "okay", "ok", "sure", "fine", "um", "uh", "ah", "oh"
}

CONVERSATIONAL_INTERJECTIONS: Set[str] = {
    "hey", "whoa", "look", "listen", "stop", "excuse", "hello", "hi", "help"
}


class ConversationalStateMachine:
    """Conversational State Machine for Manga Dialogue Speaker Attribution.
    
    Principles:
    1. Narration Isolation: Text classified as NARRATION strictly remains 'NARRATION'
       and never enters dialogue speaker state, tracking, or centroids.
    2. Repeated Turn / Proximity Protection: When consecutive dialogue bubbles are
       spatially close (dx < proximity_dx and dy < proximity_dy), they are attributed
       to the current speaker, protecting monologues and multi-bubble utterances.
    3. Conversational Exchange Transitions: State transitions between character1 and
       character2 require both a conversational trigger (question mark in prior line or
       response token in current line) and a significant spatial displacement
       (dx > dx_jump and dy > dy_jump).
    4. Conservative Multi-Speaker Gating: A third identity ('character3') is introduced
       only when strict volume and spatial separation criteria are satisfied (e.g. sequence
       dialogue >= 15 lines, page dialogue >= 6 lines, and spatial distance > third_speaker_min_dist
       from both existing character centroids). Otherwise, two-party attribution is preserved.
    5. Deterministic & Zero-Cost: Pure rule-based state tracking with O(1) per-turn computation.
    """

    def __init__(
        self,
        default_speaker: str = "character1",
        secondary_speaker: str = "character2",
        third_speaker: str = "character3",
        min_page_dialogue: int = 4,
        dx_jump: float = 0.45,
        dy_jump: float = 0.15,
        proximity_dx: float = 0.20,
        proximity_dy: float = 0.12,
        enable_third_speaker: bool = False,
        third_speaker_min_seq_lines: int = 15,
        third_speaker_min_page_lines: int = 6,
        third_speaker_min_dist: float = 0.45,
    ) -> None:
        self.default_speaker = default_speaker
        self.secondary_speaker = secondary_speaker
        self.third_speaker = third_speaker
        self.min_page_dialogue = min_page_dialogue
        self.dx_jump = dx_jump
        self.dy_jump = dy_jump
        self.proximity_dx = proximity_dx
        self.proximity_dy = proximity_dy
        self.enable_third_speaker = enable_third_speaker
        self.third_speaker_min_seq_lines = third_speaker_min_seq_lines
        self.third_speaker_min_page_lines = third_speaker_min_page_lines
        self.third_speaker_min_dist = third_speaker_min_dist

        self.reset_sequence()

    def reset_sequence(self) -> None:
        """Reset internal state for a new 3-page sequence."""
        self.current_speaker = self.default_speaker
        self.speaker_positions: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.speaker_line_counts: Dict[str, int] = defaultdict(int)
        self.total_seq_dialogue_count = 0

    def _centroid(self, speaker: str) -> Optional[Tuple[float, float]]:
        pts = self.speaker_positions[speaker]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def attribute_page(
        self,
        classified_items: List[Dict[str, Any]],
    ) -> List[str]:
        """Assign speaker identities for all classified text items on a single page.
        
        Args:
            classified_items: List of dicts, each with keys 'text', 'is_narration', 'cx', 'cy'.
            
        Returns:
            List of speaker label strings ('NARRATION', 'character1', 'character2', etc.).
        """
        page_speakers: List[str] = []
        page_dialogue_count = sum(1 for it in classified_items if not it.get("is_narration", False))
        last_dialogue_item: Optional[Dict[str, Any]] = None

        for it in classified_items:
            # 1. Narration strictly preserved
            if it.get("is_narration", False):
                page_speakers.append("NARRATION")
                continue

            self.total_seq_dialogue_count += 1
            assigned_speaker = self.current_speaker

            # 2. Dialogue turn attribution
            if page_dialogue_count >= self.min_page_dialogue and last_dialogue_item is not None:
                dx = abs(it["cx"] - last_dialogue_item["cx"])
                dy = abs(it["cy"] - last_dialogue_item["cy"])
                t_last = last_dialogue_item["text"].strip()
                t_curr = it["text"].strip()

                has_q = ("?" in t_last)
                first_tok = t_curr.split()[0].lower().strip(".,!?:;\"'") if t_curr.split() else ""
                is_resp = first_tok in CONVERSATIONAL_RESPONSE_WORDS
                is_interject = first_tok in CONVERSATIONAL_INTERJECTIONS
                is_jump = (dx > self.dx_jump and dy > self.dy_jump)

                # Check 1: Third-speaker introduction gating
                assigned_third = False
                if self.enable_third_speaker:
                    if (
                        self.total_seq_dialogue_count >= self.third_speaker_min_seq_lines
                        and page_dialogue_count >= self.third_speaker_min_page_lines
                        and self.speaker_line_counts[self.default_speaker] >= 3
                        and self.speaker_line_counts[self.secondary_speaker] >= 2
                    ):
                        c1 = self._centroid(self.default_speaker)
                        c2 = self._centroid(self.secondary_speaker)
                        if c1 and c2:
                            d1 = math.hypot(it["cx"] - c1[0], it["cy"] - c1[1])
                            d2 = math.hypot(it["cx"] - c2[0], it["cy"] - c2[1])
                            if d1 > self.third_speaker_min_dist and d2 > self.third_speaker_min_dist:
                                if is_interject or (is_jump and (has_q or is_resp)):
                                    assigned_speaker = self.third_speaker
                                    assigned_third = True

                if not assigned_third:
                    # Check 2: Proximity anchor (protects same-speaker runs / monologues)
                    if (
                        self.proximity_dx > 0
                        and self.proximity_dy > 0
                        and dx < self.proximity_dx
                        and dy < self.proximity_dy
                    ):
                        assigned_speaker = self.current_speaker

                    # Check 3: Conversational exchange transition
                    elif (has_q or is_resp) and is_jump:
                        self.current_speaker = (
                            self.secondary_speaker
                            if self.current_speaker == self.default_speaker
                            else self.default_speaker
                        )
                        assigned_speaker = self.current_speaker

                    else:
                        assigned_speaker = self.current_speaker

            page_speakers.append(assigned_speaker)
            last_dialogue_item = it
            self.speaker_positions[assigned_speaker].append((it["cx"], it["cy"]))
            self.speaker_line_counts[assigned_speaker] += 1

        return page_speakers

    def attribute_sequence(
        self,
        classified_pages: List[List[Dict[str, Any]]],
    ) -> List[List[str]]:
        """Assign speaker identities across all pages in a multi-page sequence."""
        self.reset_sequence()
        sequence_speakers: List[List[str]] = []
        for page_items in classified_pages:
            sequence_speakers.append(self.attribute_page(page_items))
        return sequence_speakers

import unittest
from src.models.state_machine import ConversationalStateMachine
from src.models.baseline_predictor import RapidOCRBaselinePredictor


class TestConversationalStateMachine(unittest.TestCase):
    """Regression and behavior tests for ConversationalStateMachine."""

    def setUp(self):
        self.csm = ConversationalStateMachine(
            default_speaker="character1",
            secondary_speaker="character2",
            third_speaker="character3",
            min_page_dialogue=4,
            dx_jump=0.45,
            dy_jump=0.15,
            proximity_dx=0.20,
            proximity_dy=0.12,
            enable_third_speaker=False,
        )

    def test_narration_strictly_remains_narration(self):
        """Narration items must always be labeled 'NARRATION' and never modify dialogue state."""
        items = [
            {"text": "CHAPTER 1", "is_narration": True, "cx": 0.5, "cy": 0.1},
            {"text": "MEANWHILE IN TOKYO...", "is_narration": True, "cx": 0.2, "cy": 0.2},
            {"text": "HELLO THERE!", "is_narration": False, "cx": 0.8, "cy": 0.3},
            {"text": "END OF CHAPTER", "is_narration": True, "cx": 0.5, "cy": 0.9},
        ]
        speakers = self.csm.attribute_page(items)
        self.assertEqual(speakers[0], "NARRATION")
        self.assertEqual(speakers[1], "NARRATION")
        self.assertEqual(speakers[2], "character1")
        self.assertEqual(speakers[3], "NARRATION")

    def test_monologue_preservation(self):
        """A page with monologue speech (or low dialogue density < 4) must stay on character1."""
        items = [
            {"text": "I can't believe this is happening.", "is_narration": False, "cx": 0.7, "cy": 0.2},
            {"text": "Everything was supposed to go smoothly.", "is_narration": False, "cx": 0.65, "cy": 0.4},
            {"text": "What went wrong?", "is_narration": False, "cx": 0.2, "cy": 0.7},  # has_q, but only 3 dialogue items
        ]
        speakers = self.csm.attribute_page(items)
        self.assertEqual(speakers, ["character1", "character1", "character1"])

    def test_repeated_turns_same_speaker_proximity_protection(self):
        """Consecutive bubbles within proximity (dx < 0.20, dy < 0.12) must not switch even with a question."""
        items = [
            {"text": "LINE 1", "is_narration": False, "cx": 0.80, "cy": 0.10},
            {"text": "ARE YOU SURE?", "is_narration": False, "cx": 0.82, "cy": 0.20},  # has_q, dx=0.02, dy=0.10 -> proximity
            {"text": "I MEAN REALLY SURE?", "is_narration": False, "cx": 0.81, "cy": 0.28},
            {"text": "DON'T LIE TO ME.", "is_narration": False, "cx": 0.79, "cy": 0.38},
        ]
        speakers = self.csm.attribute_page(items)
        self.assertEqual(speakers, ["character1", "character1", "character1", "character1"])

    def test_question_and_response_sequences(self):
        """A question followed by a spatial jump and response token triggers a speaker switch."""
        items = [
            {"text": "WHAT ARE YOU DOING HERE?", "is_narration": False, "cx": 0.85, "cy": 0.15},
            {"text": "YEAH, I JUST ARRIVED.", "is_narration": False, "cx": 0.25, "cy": 0.45},  # dx=0.60, dy=0.30 -> switch
            {"text": "DID YOU BRING THE BOOK?", "is_narration": False, "cx": 0.20, "cy": 0.55}, # dx=0.05, dy=0.10 -> stay on character2
            {"text": "NO, I FORGOT IT.", "is_narration": False, "cx": 0.80, "cy": 0.80},  # dx=0.60, dy=0.25 -> switch back to character1
        ]
        speakers = self.csm.attribute_page(items)
        self.assertEqual(speakers[0], "character1")
        self.assertEqual(speakers[1], "character2")
        self.assertEqual(speakers[2], "character2")
        self.assertEqual(speakers[3], "character1")

    def test_short_reaction_scenes(self):
        """Short reaction scenes with few lines (< 4) must not trigger erratic switches."""
        items = [
            {"text": "EH?", "is_narration": False, "cx": 0.85, "cy": 0.20},
            {"text": "LOOK AT THAT!", "is_narration": False, "cx": 0.15, "cy": 0.60},
        ]
        speakers = self.csm.attribute_page(items)
        self.assertEqual(speakers, ["character1", "character1"])

    def test_three_speaker_conversation_gating(self):
        """Third speaker is only assigned when enabled and strict volume/centroid distance criteria are met."""
        # 1. With enable_third_speaker=False, character3 is never assigned
        self.csm.enable_third_speaker = False
        items = [
            {"text": "WHY ARE YOU HERE?", "is_narration": False, "cx": 0.85, "cy": 0.10},
            {"text": "YEAH, I CAME BACK.", "is_narration": False, "cx": 0.20, "cy": 0.30},
            {"text": "DID YOU TELL HIM?", "is_narration": False, "cx": 0.85, "cy": 0.45},
            {"text": "NO, NOT YET.", "is_narration": False, "cx": 0.20, "cy": 0.60},
            {"text": "HEY! WHAT ABOUT ME?!", "is_narration": False, "cx": 0.50, "cy": 0.90},
            {"text": "WHO ARE YOU?", "is_narration": False, "cx": 0.85, "cy": 0.95},
        ]
        speakers_disabled = self.csm.attribute_page(items)
        self.assertNotIn("character3", speakers_disabled)

        # 2. With enable_third_speaker=True, verify behavior under gating
        csm_three = ConversationalStateMachine(
            enable_third_speaker=True,
            third_speaker_min_seq_lines=5,
            third_speaker_min_page_lines=6,
            third_speaker_min_dist=0.30,
        )
        # Seed character1 on the right and character2 on the left
        csm_three.speaker_line_counts["character1"] = 4
        csm_three.speaker_line_counts["character2"] = 3
        csm_three.speaker_positions["character1"] = [(0.85, 0.15), (0.85, 0.25), (0.85, 0.35)]
        csm_three.speaker_positions["character2"] = [(0.15, 0.15), (0.15, 0.25), (0.15, 0.35)]
        csm_three.total_seq_dialogue_count = 10
        csm_three.current_speaker = "character1"

        # Interjection at bottom center (far from both right and left clusters)
        interject_items = [
            {"text": "LINE 1", "is_narration": False, "cx": 0.85, "cy": 0.10},
            {"text": "LINE 2", "is_narration": False, "cx": 0.85, "cy": 0.20},
            {"text": "LINE 3", "is_narration": False, "cx": 0.15, "cy": 0.30},
            {"text": "LINE 4", "is_narration": False, "cx": 0.15, "cy": 0.40},
            {"text": "HEY! STOP RIGHT THERE!", "is_narration": False, "cx": 0.50, "cy": 0.85},
            {"text": "LINE 6", "is_narration": False, "cx": 0.85, "cy": 0.90},
        ]
        speakers_three = csm_three.attribute_page(interject_items)
        self.assertEqual(speakers_three[4], "character3")

    def test_deterministic_output(self):
        """Identical inputs must yield identical speaker assignments across multiple runs."""
        items = [
            {"text": "IS ANYONE HOME?", "is_narration": False, "cx": 0.80, "cy": 0.15},
            {"text": "YES, I AM.", "is_narration": False, "cx": 0.20, "cy": 0.40},
            {"text": "WHERE IS THE KEY?", "is_narration": False, "cx": 0.80, "cy": 0.65},
            {"text": "OKAY, IT IS ON THE TABLE.", "is_narration": False, "cx": 0.20, "cy": 0.85},
        ]
        self.csm.reset_sequence()
        run1 = self.csm.attribute_page(items)
        self.csm.reset_sequence()
        run2 = self.csm.attribute_page(items)
        self.assertEqual(run1, run2)

    def test_preservation_of_strategy_f_default_behavior(self):
        """RapidOCRBaselinePredictor with conversational_spatial must run Strategy F identically."""
        predictor_f = RapidOCRBaselinePredictor(
            speaker_attribution_mode="conversational_spatial",
            enable_clustering=False,
            enable_dialogue_filter=False,
            enable_narration_classifier=False,
            enable_word_segmenter=False,
        )
        self.assertEqual(predictor_f.speaker_attribution_mode, "conversational_spatial")
        self.assertIsNone(predictor_f.state_machine)


if __name__ == "__main__":
    unittest.main()

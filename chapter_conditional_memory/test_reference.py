"""Behavioral tests for the address boundary and exact lab, using unittest."""
from itertools import product
from pathlib import Path
import re
import unittest

import sympy as sp

import reference as cm
import homework_starter as starter


class ConditionalMemoryTests(unittest.TestCase):
    def test_completed_homework(self):
        result = cm.verify()
        self.assertEqual(result["stream_configurations"], 1600)
        self.assertEqual(result["collision_experiment"]["cases"], 625)
        self.assertEqual(result["exposed_us"], sp.Rational(262, 125))

    def test_complete_padded_address_domain(self):
        # All 25 C^2 keys are in the hash domain, even keys not realized by a prefix.
        for key in product(range(5), repeat=2):
            i, j = cm.address(key)
            self.assertTrue(0 <= i < 5 and 0 <= j < 7)
        for key in ((0,), (0, 1, 2), (-1, 0), (0, 5), (True, 0), (0.0, 1)):
            with self.assertRaises(ValueError):
                cm.address(key)

    def test_invalid_sequences_and_orders(self):
        for tokens in ((), (4,), (-1,), (True,), (1.0,), (sp.Integer(1),), (3, 0)):
            with self.assertRaises(ValueError):
                cm.suffixes(tokens)
        for order in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                cm.suffixes(cm.CORE, order)

    def test_stream_reset_and_invalid_input_are_atomic(self):
        stream = cm.SuffixStream(3)
        self.assertEqual(stream.push(0), (4, 4, 0))
        before = stream.history, stream.ended
        for invalid in (4, True, -1):
            with self.assertRaises(ValueError):
                stream.push(invalid)
            self.assertEqual((stream.history, stream.ended), before)
        with self.assertRaises(ValueError):
            stream.push(1, 4)
        self.assertEqual((stream.history, stream.ended), before)
        stream.push(3)
        with self.assertRaises(ValueError):
            stream.push(0)
        stream.reset()
        self.assertEqual(stream.push(1), (4, 4, 1))

    def test_zero_length_history_at_order_one(self):
        stream = cm.SuffixStream(1)
        for token in cm.REPLAY:
            self.assertEqual(stream.push(token), (token,))
            self.assertEqual(stream.history, ())

    def test_selector_validation_and_empty_gather(self):
        self.assertEqual(cm.selection([], 5).shape, (0, 5))
        for addresses in ((-1,), (5,), (True,), (1.0,)):
            with self.assertRaises(ValueError):
                cm.selection(addresses, 5)

    def test_fusion_shapes_and_overlap_domain(self):
        with self.assertRaises(ValueError):
            cm.fusion(sp.zeros(2, 1), sp.zeros(3, 1), sp.eye(2), sp.zeros(2, 1), sp.zeros(3, 1))
        for values in ((1, 0, 0, 0), (-1, 1, 0, 0), (1, 1, -1, 0), (1, 1, 0, -1)):
            with self.assertRaises(ValueError):
                cm.ideal_overlap(*values)
        self.assertEqual(cm.ideal_overlap(0, 1, 0, 0), (0, 0))

    def test_payload_table_identity(self):
        result = cm.payload(cm.REPLAY)
        self.assertEqual(result["distinct_rows"], 9)
        self.assertEqual(len({row for _, row in result["requests"]}), 5)
        self.assertIn((0, 1), result["distinct_requests"])
        self.assertIn((1, 1), result["distinct_requests"])
        self.assertEqual(cm.table_storage((5, 7), (1, 1), (1, 2)), 19)
        with self.assertRaises(ValueError):
            cm.table_storage((5,), (1, 1), (2,))

    def check_student_interface(self, namespace):
        for tokens in (cm.CORE, cm.REPLAY):
            self.assertEqual(namespace["suffix_bigrams"](tokens), cm.suffixes(tokens))
            for key in cm.suffixes(tokens):
                self.assertEqual(namespace["address"](key), cm.address(key))
                self.assertEqual(namespace["read"](key), cm.read(key))

    def test_companion_starter_matches_reference(self):
        self.check_student_interface(vars(starter))

    def test_printed_specification_keeps_inputs_and_domain(self):
        repo = Path(__file__).resolve().parents[2]
        source = repo / "book/chapter_conditional_memory/sections/10_homework.tex"
        if not source.is_file():
            self.skipTest("Manuscript parity check requires the source repository")
        manuscript = source.read_text()
        for label in ("eq:conditional-fixture", "eq:conditional-replay"):
            self.assertIn("\\ref{" + label + "}", manuscript)
        for rule in ("BOS code 4", "EOS code 3", "nonempty integer-code", "reject EOS", "reset between"):
            self.assertIn(rule, manuscript)
        self.assertNotIn("\\begin{lstlisting}", manuscript)
        self.check_student_interface(vars(starter))


if __name__ == "__main__":
    unittest.main()

import json
import os
import unittest
from pathlib import Path

from app.data_layer.bitext_data_layer import BitextDataLayer, BitextStatField
from app.tools.bitext_tools import (
    _format_stats,
    _resolve_stat_field,
    create_bitext_tools,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATHS = [
    PROJECT_ROOT / "bitext_data.csv",
    PROJECT_ROOT.parent / "bitext_data.csv",
]


def _resolve_data_path() -> str:
    for path in DATA_PATHS:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "bitext_data.csv not found. Expected at project root or parent directory."
    )


def _tools_by_name(data_layer) -> dict:
    return {tool.name: tool for tool in create_bitext_tools(data_layer)}


class TestBitextToolHelpers(unittest.TestCase):
    def test_resolve_stat_field_by_value(self):
        self.assertEqual(_resolve_stat_field("category"), BitextStatField.CATEGORY)

    def test_resolve_stat_field_by_name(self):
        self.assertEqual(_resolve_stat_field("FLAGS"), BitextStatField.FLAGS)

    def test_resolve_stat_field_invalid(self):
        self.assertIsNone(_resolve_stat_field("unknown"))

    def test_format_stats_from_list(self):
        stats = [{"value": "ORDER", "count": 1, "percentage": 100.0}]
        self.assertEqual(json.loads(_format_stats(stats)), stats)

    def test_format_stats_from_string(self):
        self.assertEqual(_format_stats("already formatted"), "already formatted")


class TestBitextToolsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_layer = BitextDataLayer(_resolve_data_path())
        cls.tools = _tools_by_name(cls.data_layer)
        cls.first_row = cls.data_layer.df.iloc[0]

    def test_lookup_by_id_returns_record(self):
        result = self.tools["lookup_by_id"].invoke({"record_id": "0"})

        parsed = json.loads(result)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["category"], self.first_row["category"])
        self.assertEqual(parsed[0]["intent"], self.first_row["intent"])

    def test_lookup_by_id_not_found(self):
        result = self.tools["lookup_by_id"].invoke({"record_id": "9999999"})

        self.assertIn("No record found", result)

    def test_lookup_by_category_returns_records(self):
        category = self.first_row["category"]
        result = self.tools["lookup_by_category"].invoke({"category": category})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertLessEqual(len(parsed), 5)
        self.assertTrue(all(row["category"] == category for row in parsed))

    def test_lookup_by_category_not_found(self):
        result = self.tools["lookup_by_category"].invoke(
            {"category": "UNKNOWN_CATEGORY_XYZ"}
        )

        self.assertIn("No records found", result)

    def test_lookup_by_question_returns_records(self):
        instruction = self.first_row["instruction"]
        result = self.tools["lookup_by_question"].invoke({"question": instruction})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertEqual(parsed[0]["instruction"], instruction)

    def test_lookup_by_question_not_found(self):
        result = self.tools["lookup_by_question"].invoke(
            {"question": "xyzzyx-not-in-dataset-12345"}
        )

        self.assertIn("No record found", result)

    def test_get_flags_statistics(self):
        result = self.tools["get_flags_statistics"].invoke({})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertEqual(set(parsed[0].keys()), {"value", "count", "percentage"})

    def test_get_category_statistics(self):
        result = self.tools["get_category_statistics"].invoke({})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertIn("ORDER", {row["value"] for row in parsed})

    def test_get_intent_statistics(self):
        result = self.tools["get_intent_statistics"].invoke({})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertIn("cancel_order", {row["value"] for row in parsed})

    def test_get_field_statistics_valid_field(self):
        result = self.tools["get_field_statistics"].invoke({"field": "category"})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertIn("ORDER", {row["value"] for row in parsed})

    def test_get_field_statistics_invalid_field(self):
        result = self.tools["get_field_statistics"].invoke({"field": "invalid"})

        self.assertIn("Invalid field", result)
        self.assertIn("flags", result)

    def test_get_categories(self):
        result = self.tools["get_categories"].invoke({})

        self.assertIn("category", result)
        self.assertIn("flags", result)
        self.assertIn("intent", result)

    def test_search_instruction_returns_matches(self):
        result = self.tools["search_instruction"].invoke({"query": "cancelling order"})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertLessEqual(len(parsed), 5)
        self.assertTrue(
            all("cancelling order" in row["instruction"] for row in parsed)
        )

    def test_search_instruction_not_found(self):
        result = self.tools["search_instruction"].invoke(
            {"query": "xyzzyx-not-in-dataset-12345"}
        )

        self.assertIn("No records found", result)

    def test_search_response_returns_matches(self):
        result = self.tools["search_response"].invoke({"query": "canceling order"})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertTrue(all("canceling order" in row["response"] for row in parsed))

    def test_search_response_not_found(self):
        result = self.tools["search_response"].invoke(
            {"query": "xyzzyx-not-in-dataset-12345"}
        )

        self.assertIn("No records found", result)

    def test_search_instruction_and_response_returns_matches(self):
        result = self.tools["search_instruction_and_response"].invoke({"query": "ORDER"})

        parsed = json.loads(result)
        self.assertGreater(len(parsed), 0)
        self.assertTrue(
            any(
                "ORDER" in row["instruction"] or "ORDER" in row["response"]
                for row in parsed
            )
        )

    def test_search_instruction_and_response_not_found(self):
        result = self.tools["search_instruction_and_response"].invoke(
            {"query": "xyzzyx-not-in-dataset-12345"}
        )

        self.assertIn("No records found", result)

    def test_create_bitext_tools_returns_all_tools(self):
        tool_names = {tool.name for tool in create_bitext_tools(self.data_layer)}

        self.assertEqual(
            tool_names,
            {
                "lookup_by_id",
                "lookup_by_category",
                "lookup_by_question",
                "get_flags_statistics",
                "get_category_statistics",
                "get_intent_statistics",
                "get_field_statistics",
                "get_categories",
                "search_instruction",
                "search_response",
                "search_instruction_and_response",
            },
        )


if __name__ == "__main__":
    unittest.main()

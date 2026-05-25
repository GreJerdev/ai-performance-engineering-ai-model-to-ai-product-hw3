from collections import Counter
import json

from langchain_core.tools import tool

from app.data_layer.bitext_data_layer import BitextDataLayer, BitextStatField
from app.models.data_item import DataItem
from app.models.search_items import SearchItems

_FIELD_LOOKUP = {
    **{f.value: f for f in BitextStatField},
    **{f.name.lower(): f for f in BitextStatField},
}


def _resolve_stat_field(field: str) -> BitextStatField | None:
    return _FIELD_LOOKUP.get(field.strip().lower())


def _format_stats(stats) -> str:
    if isinstance(stats, str):
        return stats
    return json.dumps([val["value"] for val in stats])


def _data_items_to_json(items: list[DataItem]) -> str:
    return json.dumps([item.model_dump() for item in items], ensure_ascii=False)


def _rows_to_data_items(df, row_limit: int = 5) -> list[DataItem]:
    return [
        DataItem(
            flags=str(row["flags"]),
            instruction=str(row["instruction"]),
            category=str(row["category"]),
            intent=str(row["intent"]),
            response=str(row["response"]),
        )
        for _, row in df.head(row_limit).iterrows()
    ]


def create_bitext_tools(data_layer: BitextDataLayer) -> list:
    @tool
    def lookup_by_id(record_id: int) -> str:
        """Fetch one Bitext record by row index via BitextDataLayer.get_data_by_id.

        Args:
            record_id: Integer row index in the dataset (e.g. 0, 2230).

        Returns:
            JSON string with flags, instruction, category, intent, and response.
        """
        result = data_layer.get_data_by_id(record_id)
        if result.empty:
            return f"No record found for id: {record_id}"

        row = result.iloc[0]
        return DataItem(
            flags=str(row["flags"]),
            instruction=str(row["instruction"]),
            category=str(row["category"]),
            intent=str(row["intent"]),
            response=str(row["response"]),
        ).model_dump_json()

    @tool
    def get_flags_list() -> str:
        """Retrieve a list of all unique flags available in the dataset."""
        return _format_stats(
            data_layer.get_unique_values_stats(BitextStatField.FLAGS)
        )

    @tool
    def get_category_list() -> str:
        """Retrieve a list of all unique categories available in the dataset."""
        return _format_stats(
            data_layer.get_unique_values_stats(BitextStatField.CATEGORY)
        )

    @tool
    def get_intent_list() -> str:
        """Retrieve a list of all unique intents available in the dataset."""
        return _format_stats(
            data_layer.get_unique_values_stats(BitextStatField.INTENT)
        )

    @tool(args_schema=SearchItems)
    def search_instruction_and_response_with_schema(
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None,
        row_limit: int = 5,
        skip_from_start: int = 0,
    ) -> str:
        """Multi-field filtered search via BitextDataLayer.search_instruction_and_response_with_schema.

        Combines filters with AND logic. Omit a parameter (or pass null) to skip that filter.

        Args:
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).
            row_limit: Max rows to return (default 5).
            skip_from_start: Number of rows to skip from the start of the dataset (default 0).

        Returns:
            JSON string of matching records, or an error message if none match.
        """
        search_items = SearchItems(
            flags=flags,
            instruction=instruction,
            category=category,
            intent=intent,
            response=response,
            row_limit=row_limit,
            skip_from_start=skip_from_start,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        return _data_items_to_json(_rows_to_data_items(result, search_items.row_limit))

    @tool
    def summarize_counts(items_json: str) -> str:
        """Count frequency of flags, category, and intent in a JSON list of records.

        Args:
            items_json: JSON array of objects with flags, category, and intent fields.

        Returns:
            JSON string mapping flags, category, and intent to frequency counts.
        """
        items = json.loads(items_json)
        flags_counter = Counter()
        category_counter = Counter()
        intent_counter = Counter()

        for item in items:
            flags = item.get("flags")
            if isinstance(flags, list):
                flags_counter.update(flags)
            elif flags is not None:
                flags_counter[flags] += 1

            category = item.get("category")
            if category is not None:
                category_counter[category] += 1

            intent = item.get("intent")
            if intent is not None:
                intent_counter[intent] += 1

        return json.dumps(
            {
                "flags": dict(flags_counter),
                "category": dict(category_counter),
                "intent": dict(intent_counter),
            },
            ensure_ascii=False,
            indent=2,
        )

    return [
        lookup_by_id,
        get_flags_list,
        get_category_list,
        get_intent_list,
        search_instruction_and_response_with_schema,
        summarize_counts,
    ]

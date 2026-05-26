from collections import Counter
import json

from langchain_core.tools import tool

from app.data_layer.bitext_data_layer import BitextDataLayer, BitextStatField
from app.models.data_item import DataItem
from app.models.search_items import (
    CrossTabSearchItems,
    RandomSampleSearchItems,
    SearchForCountItems,
    SearchItems,
    TextLengthStatsItems,
)

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

    @tool(args_schema=SearchForCountItems)
    def summarizes_search_instruction_and_response_with_schema(
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None,
       
    ) -> str:
        """Print a structural summary of the filtered result set via DataFrame.info().

        Applies the same multi-field AND filters as search_instruction_and_response_with_schema,
        then prints the DataFrame shape, column names, non-null counts, dtypes, and memory
        usage to stdout. Use this to inspect the shape and completeness of a result set
        rather than retrieving actual records.

        Args:
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).
        Returns:
            A message indicating how many rows matched, or an error message if none match.
        """
        search_items = SearchItems(
            flags=flags,
            instruction=instruction,
            category=category,
            intent=intent,
            response=response,
            row_limit=0,
            skip_from_start=0,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        result.info()
        return f"{len(result)} matching rows found."
    
    @tool(args_schema=SearchForCountItems)
    def count_search_instruction_and_response(
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None
    ) -> str:
        """Return the total number of rows matching the given filters.

        Applies the same multi-field AND filters as search_instruction_and_response_with_schema
        but returns only the match count rather than the records themselves. Use this to
        check result set size before fetching data, or to answer "how many" questions.

        Args:
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).
        Returns:
            The number of matching rows counts as a string, or an error message if none match.
        """
        search_items = SearchItems(
            flags=flags,
            instruction=instruction,
            category=category,
            intent=intent,
            response=response,
            row_limit=0,
            skip_from_start=0,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        return f"{len(result)}"

    @tool(args_schema=SearchForCountItems)
    def summarize_search_as_json(
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None,
    ) -> str:
        """Return a JSON summary of the rows matching the given filters.
 
        Applies the same multi-field AND filters as search_instruction_and_response_with_schema
        and returns aggregate statistics about the result set: total row count and value
        frequency distributions for flags, category, and intent columns. Use this to
        understand the composition of a result set without retrieving the raw records.

        Args:
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).

        Returns:
            JSON string with total_rows and value_counts for flags, category, and intent,
            or an error message if no rows match.
        """
        search_items = SearchItems(
            flags=flags,
            instruction=instruction,
            category=category,
            intent=intent,
            response=response,
            row_limit=0,
            skip_from_start=0,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        summary = {
            "total_rows": len(result),
            "value_counts": {
                "flags": result["flags"].value_counts().to_dict(),
                "category": result["category"].value_counts().to_dict(),
                "intent": result["intent"].value_counts().to_dict(),
            },
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

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

    @tool
    def get_intents_by_category(category: str) -> str:
        """Return all distinct intents that belong to a given category.

        Args:
            category: Exact category name (case-insensitive, e.g. 'ORDER', 'REFUND').

        Returns:
            JSON with the category name, a sorted list of unique intents, and their count,
            or an error message if the category is not found.
        """
        matches = data_layer.df[
            data_layer.df["category"].str.lower() == category.strip().lower()
        ]
        if matches.empty:
            return f"No records found for category: {category}"
        intents = sorted(matches["intent"].dropna().unique().tolist())
        return json.dumps(
            {"category": category, "total_intents": len(intents), "intents": intents},
            ensure_ascii=False,
            indent=2,
        )

    @tool(args_schema=CrossTabSearchItems)
    def cross_tab_search(
        row_field: str,
        col_field: str,
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None,
    ) -> str:
        """Return a pivot table of counts for two categorical fields in the filtered result set.

        Applies the standard multi-field AND filters then produces a cross-tabulation
        (pivot table) of row_field vs col_field. Use this to answer questions like
        "how are intents distributed within each category?" or "which flags appear
        under each intent?".

        Args:
            row_field: Column for rows. Must be one of: flags, category, intent.
            col_field: Column for columns. Must be one of: flags, category, intent.
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).

        Returns:
            JSON object where keys are row_field values and values are dicts mapping
            col_field values to counts, or an error message if inputs are invalid.
        """
        valid_fields = {"flags", "category", "intent"}
        if row_field not in valid_fields or col_field not in valid_fields:
            return f"Invalid field. row_field and col_field must be one of: {sorted(valid_fields)}"
        if row_field == col_field:
            return "row_field and col_field must be different."
        search_items = SearchItems(
            flags=flags, instruction=instruction, category=category,
            intent=intent, response=response, row_limit=0, skip_from_start=0,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        pivot = (
            result.groupby([row_field, col_field])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        return json.dumps(pivot, ensure_ascii=False, indent=2)

    @tool(args_schema=RandomSampleSearchItems)
    def random_sample_search(
        sample_size: int = 5,
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None,
    ) -> str:
        """Return a random sample of records from the filtered result set.

        Applies the standard multi-field AND filters then returns a random selection
        of matching rows. Prefer this over search_instruction_and_response_with_schema
        when you want representative examples rather than the first N rows.

        Args:
            sample_size: Number of random records to return (default 5, max 50).
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).

        Returns:
            JSON string of randomly sampled records, or an error message if none match.
        """
        sample_size = max(1, min(sample_size, 50))
        search_items = SearchItems(
            flags=flags, instruction=instruction, category=category,
            intent=intent, response=response, row_limit=0, skip_from_start=0,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        sample = result.sample(n=min(sample_size, len(result)))
        return _data_items_to_json(_rows_to_data_items(sample, sample_size))

    @tool(args_schema=TextLengthStatsItems)
    def text_length_stats(
        field: str,
        flags: list[str] | None = None,
        instruction: list[str] | None = None,
        category: list[str] | None = None,
        intent: list[str] | None = None,
        response: list[str] | None = None,
    ) -> str:
        """Return character and word-count statistics for a text field in the filtered result set.

        Applies the standard multi-field AND filters, then computes length statistics
        for either the instruction or response column. Use this to answer questions like
        "how long are typical responses for refund intents?" or "are cancellation
        instructions longer than account queries?".

        Args:
            field: Text column to analyse. Must be one of: instruction, response.
            flags: Exact flag codes; row must match one of these values.
            instruction: Keywords; row matches if instruction contains any (case-insensitive).
            category: Exact categories; row must match one of these values.
            intent: Exact intents; row must match one of these values.
            response: Keywords; row matches if response contains any (case-insensitive).

        Returns:
            JSON with min, max, mean, p50, and p95 for both character count and word count,
            or an error message if the field is invalid or no rows match.
        """
        if field not in {"instruction", "response"}:
            return "Invalid field. Must be one of: instruction, response."
        search_items = SearchItems(
            flags=flags, instruction=instruction, category=category,
            intent=intent, response=response, row_limit=0, skip_from_start=0,
        )
        result = data_layer.search_instruction_and_response_with_schema(search_items)
        if result.empty:
            return "No records found matching the given filters."
        texts = result[field].dropna().astype(str)
        char_lengths = texts.str.len()
        word_lengths = texts.str.split().str.len()
        stats = {
            "field": field,
            "total_rows": len(texts),
            "char_count": {
                "min": int(char_lengths.min()),
                "max": int(char_lengths.max()),
                "mean": round(float(char_lengths.mean()), 1),
                "p50": int(char_lengths.quantile(0.50)),
                "p95": int(char_lengths.quantile(0.95)),
            },
            "word_count": {
                "min": int(word_lengths.min()),
                "max": int(word_lengths.max()),
                "mean": round(float(word_lengths.mean()), 1),
                "p50": int(word_lengths.quantile(0.50)),
                "p95": int(word_lengths.quantile(0.95)),
            },
        }
        return json.dumps(stats, ensure_ascii=False, indent=2)

    @tool
    def get_response_variants(intent: str, sample_size: int = 5) -> str:
        """Return a sample of distinct response phrasings for a given intent.

        Each intent has multiple response variations in the dataset. Use this to explore
        how a specific intent is handled in different ways, or to answer questions like
        "what are the different ways cancel_order requests are handled?".

        Args:
            intent: Exact intent name (case-insensitive, e.g. 'cancel_order').
            sample_size: Number of distinct response samples to return (default 5, max 20).

        Returns:
            JSON with the intent, total distinct response count, and a list of sampled
            response texts, or an error message if the intent is not found.
        """
        sample_size = max(1, min(sample_size, 20))
        matches = data_layer.df[
            data_layer.df["intent"].str.lower() == intent.strip().lower()
        ]
        if matches.empty:
            return f"No records found for intent: {intent}"
        distinct = matches["response"].dropna().drop_duplicates()
        sample = distinct.sample(n=min(sample_size, len(distinct)))
        return json.dumps(
            {
                "intent": intent,
                "total_distinct_responses": len(distinct),
                "sample": sample.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )

    return [
        lookup_by_id,
        get_flags_list,
        get_category_list,
        get_intent_list,
        count_search_instruction_and_response,
        search_instruction_and_response_with_schema,
        summarize_counts,
        summarize_search_as_json,
        summarizes_search_instruction_and_response_with_schema,
        get_intents_by_category,
        cross_tab_search,
        random_sample_search,
        text_length_stats,
        get_response_variants,
    ]

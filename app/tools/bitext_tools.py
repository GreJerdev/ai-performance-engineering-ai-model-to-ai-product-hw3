from langchain_core.tools import tool

from app.data_layer.bitext_data_layer import BitextDataLayer, BitextStatField

_FIELD_LOOKUP = {
    **{f.value: f for f in BitextStatField},
    **{f.name.lower(): f for f in BitextStatField},
}


def _resolve_stat_field(field: str) -> BitextStatField | None:
    return _FIELD_LOOKUP.get(field.strip().lower())


def create_bitext_tools(data_layer: BitextDataLayer) -> list:
    @tool
    def lookup_by_id(record_id: str) -> str:
        """Look up a customer support record by its id."""
        result = data_layer.get_data_by_id(record_id)
        if result.empty:
            return f"No record found for id: {record_id}"
        return result.to_json(orient="records")

    @tool
    def lookup_by_category(category: str) -> str:
        """Look up customer support records by category."""
        result = data_layer.get_data_by_category(category)
        if result.empty:
            return f"No records found for category: {category}"
        return result.head(5).to_json(orient="records")

    @tool
    def lookup_by_question(question: str) -> str:
        """Look up a customer support record by exact question text."""
        result = data_layer.get_data_by_question(question)
        if result.empty:
            return f"No record found for question: {question}"
        return result.to_json(orient="records")

    @tool
    def get_flags_statistics() -> str:
        """Get unique flag values with count and percentage statistics."""
        return data_layer.get_unique_values_stats(BitextStatField.FLAGS).to_json(
            orient="records"
        )

    @tool
    def get_category_statistics() -> str:
        """Get unique category values with count and percentage statistics."""
        return data_layer.get_unique_values_stats(BitextStatField.CATEGORY).to_json(
            orient="records"
        )

    @tool
    def get_intent_statistics() -> str:
        """Get unique intent values with count and percentage statistics."""
        return data_layer.get_unique_values_stats(BitextStatField.INTENT).to_json(
            orient="records"
        )

    @tool
    def get_field_statistics(field: str) -> str:
        """Get unique values with count and percentage for flags, category, or intent."""
        stat_field = _resolve_stat_field(field)
        if stat_field is None:
            valid = ", ".join(f.value for f in BitextStatField)
            return f"Invalid field '{field}'. Use one of: {valid}"
        return data_layer.get_unique_values_stats(stat_field).to_json(orient="records")

    return [
        lookup_by_id,
        lookup_by_category,
        lookup_by_question,
        get_flags_statistics,
        get_category_statistics,
        get_intent_statistics,
        get_field_statistics,
    ]

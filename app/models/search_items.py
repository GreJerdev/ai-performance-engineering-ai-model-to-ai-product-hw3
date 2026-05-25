from pydantic import BaseModel, Field, field_validator

_LIST_FIELDS = ("flags", "instruction", "category", "intent", "response")


def _coerce_to_str_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    raise TypeError(f"Expected str or list[str], got {type(value).__name__}")


class SearchItems(BaseModel):
    flags: list[str] | None = Field(
        default=None,
        description="Exact flag codes to match (e.g. ['B', 'BL']). Omit to skip this filter.",
    )
    instruction: list[str] | None = Field(
        default=None,
        description="Keywords to match in instruction (any keyword matches, case-insensitive). Omit to skip.",
    )
    category: list[str] | None = Field(
        default=None,
        description="Exact category values to match (e.g. ['ORDER']). Omit to skip this filter.",
    )
    intent: list[str] | None = Field(
        default=None,
        description="Exact intent values to match (e.g. ['cancel_order']). Omit to skip this filter.",
    )
    response: list[str] | None = Field(
        default=None,
        description="Keywords to match in response (any keyword matches, case-insensitive). Omit to skip.",
    )
    row_limit: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Maximum number of matching records to return (default 5).",
    )
    skip_from_start: int = Field(
        default=0,
        ge=0,
        description="Number of rows to skip from the start of the dataset (default 0).",
    )

    @field_validator(*_LIST_FIELDS, mode="before")
    @classmethod
    def coerce_list_fields(cls, value: object) -> list[str] | None:
        return _coerce_to_str_list(value)
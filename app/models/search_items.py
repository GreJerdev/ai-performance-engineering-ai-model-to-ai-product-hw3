from pydantic import BaseModel, Field


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
        le=100,
        description="Maximum number of matching records to return (default 5).",
    )
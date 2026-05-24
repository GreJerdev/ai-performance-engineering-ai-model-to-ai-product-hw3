from pydantic import BaseModel, Field
from typing import Literal

class DataItem(BaseModel):
    flags:str = Field(..., description="The flags of the data item")
    instruction: str = Field(..., description="The instruction of the data item")
    category: str = Field(..., description="The category of the data item")
    intent: str = Field(..., description="The intent of the data item")
    response: str = Field(..., description="The response of the data item")
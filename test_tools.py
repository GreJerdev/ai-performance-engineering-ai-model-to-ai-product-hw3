# %%
from app.data_layer.bitext_data_layer import BitextDataLayer
from app.tools.bitext_tools import create_bitext_tools

bitext_data_layer = BitextDataLayer("bitext_data.csv")
tools = {tool.name: tool for tool in create_bitext_tools(bitext_data_layer)}

# %%
tools["lookup_by_id"].invoke({"record_id": 0})

# %%
tools["get_flags_list"].invoke({})

# %%
tools["get_category_list"].invoke({})

# %%
tools["get_intent_list"].invoke({})

# %%
tools["get_field_list"].invoke({"field": "category"})

# %%
tools["get_field_list"].invoke({"field": "invalid"})

# %%
tools["search_instruction_and_response_with_schema"].invoke(
    {
        "category": ["ORDER"],
        "intent": ["cancel_order"],
        "instruction": ["order", "cancelling"],
        "response": ["cancel"],
        "row_limit": 5,
    }
)

# %%
tools["search_instruction_and_response_with_schema"].invoke(
    {
   #     "flags": ["B"],
        "category": ["order"],
        "instruction": ["order"],
        "intent": ["cancel_order"],
        "response": ["help"],
        "row_limit": 3,
    }
)

# %%

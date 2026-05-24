# %%
from app.data_layer.bitext_data_layer import BitextDataLayer, BitextStatField
from app.models.search_items import SearchItems

bitext_data_layer = BitextDataLayer("bitext_data.csv")

import pickle

# %%
with open("bitext_data_layer.pkl", "wb") as f:
    pickle.dump(bitext_data_layer, f)

print("bitext_data_layer saved")


# %%
bitext_data_layer.get_data()

# %%
bitext_data_layer.get_data_by_id(1)

# %%
bitext_data_layer.get_data_by_category("ORDER")
# %%
# %%
bitext_data_layer.get_unique_values_stats(BitextStatField.FLAGS)
# %%
bitext_data_layer.get_unique_values_stats(BitextStatField.CATEGORY)
# %%
bitext_data_layer.get_unique_values_stats(BitextStatField.INTENT)
# %%
from app.data_layer.bitext_data_layer import BitextDataLayer, BitextStatField
from app.models.search_items import SearchItems

bitext_data_layer = BitextDataLayer("bitext_data.csv")
search_items = SearchItems(
    category=["ORDER"],
    intent=["cancel_order"],
    instruction=["order", "trackng", "help"],
    response=["looking","decoded","facing"],
    
)



bitext_data_layer.search_instruction_and_response_with_schema(search_items)
# %%

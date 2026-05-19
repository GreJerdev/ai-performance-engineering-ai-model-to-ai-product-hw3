from app.data_layer.bitext_data_layer import BitextDataLayer
from app.tools.bitext_tools import create_bitext_tools


def get_tools(data_layer: BitextDataLayer) -> list:
    return create_bitext_tools(data_layer)

import pandas as pd

from app.data_layer.embeddeing import Embedding
from langchain_core.documents import Document

class InstructionRAGDataLayer:

    data_column = "instruction"
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

        docs = [ Document(
                 page_content=value[self.data_column],
                 metadata={"id": index}
             ) for  index, value in enumerate(dataframe.to_dict(orient="records"))]
        self.embedding = Embedding(docs)
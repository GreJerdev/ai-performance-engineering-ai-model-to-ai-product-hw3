import os
from enum import Enum

import pandas as pd


class BitextStatField(Enum):
    FLAGS = "flags"
    CATEGORY = "category"
    INTENT = "intent"


class BitextDataLayer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        # If file does not exist locally, download from Hugging Face (`hf://`)
        if not os.path.exists(self.data_path):
            df = pd.read_csv(
                "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
            )
            # Save to the given local path
            df.to_csv(self.data_path, index=False)
        # Load the dataframe from the local file
        self.df = pd.read_csv(self.data_path)

    def get_data(self) -> pd.DataFrame:
        return self.df

    def get_data_by_id(self, id: str) -> pd.DataFrame:
        return self.df[self.df["id"] == id]

    def get_data_by_category(self, category: str) -> pd.DataFrame:
        return self.df[self.df["category"] == category]

    def get_data_by_question(self, question: str) -> pd.DataFrame:
        return self.df[self.df["question"] == question]

    def get_unique_values_stats(self, field: BitextStatField) -> pd.DataFrame:
        column = field.value
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")

        counts = self.df[column].value_counts(dropna=False)
        total = len(self.df)

        return pd.DataFrame(
            {
                "value": counts.index.astype(str),
                "count": counts.values,
                "percentage": (counts.values / total * 100).round(2),
            }
        ).reset_index(drop=True)

    
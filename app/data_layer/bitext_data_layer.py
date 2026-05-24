import os
from enum import Enum
import pandas as pd

from app.models.search_items import SearchItems



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
        print("db initialized")

    def get_data(self) -> pd.DataFrame:
        return self.df

    def get_data_by_id(self, id: int | str) -> pd.DataFrame:
        try:
            row = self.df.loc[int(id)]
        except (KeyError, ValueError, TypeError):
            return pd.DataFrame()
        return row.to_frame().T

    def get_data_by_category(self, category: str) -> pd.DataFrame:
        return self.df[self.df["category"] == category]

    def get_data_by_question(self, question: str) -> pd.DataFrame:
        return self.df[self.df["instruction"] == question]

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
        ).reset_index(drop=True).to_dict(orient="records")


    def search_instruction(self, query: str) -> pd.DataFrame:
        return self.df[self.df.instruction.str.contains(query)]

    def search_response(self, query: str) -> pd.DataFrame:
        return self.df[self.df.response.str.contains(query)]

    def search_instruction_and_response(self, query: str) -> pd.DataFrame:
        return self.df[self.df.instruction.str.contains(query) | self.df.response.str.contains(query)]

    def get_categories(self) -> dict[str, str]:
        return {
            BitextStatField.CATEGORY.value: self.get_unique_values_stats(BitextStatField.CATEGORY),
            BitextStatField.FLAGS.value: self.get_unique_values_stats(BitextStatField.FLAGS),
            BitextStatField.INTENT.value: self.get_unique_values_stats(BitextStatField.INTENT),
        }
    
    def search_instruction_and_response_with_schema(self, search_items: SearchItems) -> pd.DataFrame:
        
        instruction_filter = (
            self.df["instruction"].str.contains("|".join(search_items.instruction), case=False, na=False)
            if search_items.instruction else True
        )
        response_filter = (
            self.df["response"].str.contains("|".join(search_items.response), case=False, na=False)
            if search_items.response else True
        )
        # Case-insensitive category comparison
        category_filter = (
            self.df["category"].str.lower().isin([cat.lower() for cat in search_items.category])
            if search_items.category else True
        )
        # Case-insensitive flags comparison
        flags_filter = (
            self.df["flags"].str.lower().isin([flag.lower() for flag in search_items.flags])
            if search_items.flags else True
        )
        # Case-insensitive intent comparison
        intent_filter = (
            self.df["intent"].str.lower().isin([intent.lower() for intent in search_items.intent])
            if search_items.intent else True
        )
        return self.df[instruction_filter & response_filter & category_filter & flags_filter & intent_filter ]
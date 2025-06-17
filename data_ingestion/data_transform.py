import numpy as np
import pandas as pd
from langchain_core.documents import Document
from dotenv import load_dotenv
import os


load_dotenv()


class data_converter:
    def __init__(self):
        print("Data Converter Initialized")
        self.product_data = pd.read_csv("data/data.csv")
        

    def data_transformation(self):
        required_columns = self.product_data.columns
        required_columns = required_columns[1:]

        product_list = []
        for index,row in self.product_data.iterrows():
            object = {
                "product_name": row["product_title"],
                "product_rating": row["rating"],
                "product_summary": row["summary"],
                "product_review": row["review"],
            }

            product_list.append(object)
            
        docs = []
        # Create Document objects for each product entry
        for entry in product_list:
            metadata = {
                "product_name": entry["product_name"],
                "product_rating": entry["product_rating"],
                "product_summary": entry["product_summary"],
            }

            docs.append(
                Document(
                    page_content=entry["product_review"],
                    metadata=metadata
                )
            )

        return docs

        



if __name__ == "__main__":
    data_con = data_converter()
    data_con.data_transformation()
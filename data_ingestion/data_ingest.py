from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from data_ingestion.data_transform import data_converter


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["ASTRA_DB_KEYSPACE"] = ASTRA_DB_KEYSPACE
os.environ["ASTRA_DB_ENDPOINT"] = ASTRA_DB_ENDPOINT
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = ASTRA_DB_APPLICATION_TOKEN


class ingest_data:
    def __init__(self):
        print("Data Ingestion Initialized")
        self.embeddings = GoogleGenerativeAIEmbeddings(model = 'models/text-embedding-004')
        self.data_converter = data_converter()
        

    def data_ingestion(self,status):
        vstore = AstraDBVectorStore(
            collection_name="customer_support_chatbot_rag",
            keyspace=ASTRA_DB_KEYSPACE,
            api_endpoint=ASTRA_DB_ENDPOINT,
            application_token=ASTRA_DB_APPLICATION_TOKEN,
            embedding=self.embeddings
        )
        

        storage = status

        if storage == "None":
            docs = self.data_converter.data_transformation()
            inserted_ids = vstore.add_documents(docs)
            print(f"Inserted {len(inserted_ids)} documents into the vector store.")
        else:
            return vstore
        


        return vstore,inserted_ids





if __name__ == "__main__":
    data_ingestion = ingest_data()
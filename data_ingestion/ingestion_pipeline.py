import os
import time
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec,Pinecone
from uuid import uuid4

BATCH_SIZE = 20        # documents per batch (well under the 100/min free-tier limit)
BATCH_DELAY = 65       # seconds to wait between batches for quota reset
MAX_RETRIES = 5        # max retries per batch on rate-limit errors
class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self):
        """
        Initialize environment variables, embedding model, and set CSV file path.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader=ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config=load_config()

    def _load_env_variables(self):
        """
        Load and validate required environment variables.
        """
        load_dotenv()
        
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE","PINECONE_API_KEY"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        self.pine_cone_api_key = os.getenv("PINECONE_API_KEY")

       

    def _get_csv_path(self):
        """
        Get path to the CSV file located inside 'data' folder.
        """
        current_dir = os.getcwd()
        #csv_path = os.path.join(current_dir, 'data', 'data.csv')
        csv_path = "data/data.csv"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    def _load_csv(self):
        """
        Load product data from CSV.
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_title', 'rating', 'summary', 'review'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def transform_data(self):
        """
        Transform product data into list of LangChain Document objects.
        """
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                "product_name": row['product_title'],
                "product_rating": row['rating'],
                "product_summary": row['summary'],
                "product_review": row['review']
            }
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                "product_name": entry["product_name"],
                "product_rating": entry["product_rating"],
                "product_summary": entry["product_summary"]
            }
            doc = Document(page_content=entry["product_review"], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into AstraDB vector store in batches to avoid rate limits.
        """
        try:
            collection_name=self.config["astra_db"]["collection_name"]
            vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )
        except Exception as e:
            # Fallback to Pinecone if AstraDB connection fails
            print(f"AstraDB connection failed with error: {e}. Falling back to Pinecone.")
            pinecone = Pinecone(api_key=self.pine_cone_api_key)
            index_name=self.config["pinecone"]["index_name"]
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=768,  
                    serverless_spec=ServerlessSpec(min_nodes=1, max_nodes=3)
                )
            vstore = PineconeVectorStore(
                index_name=index_name,
                embedding=self.model_loader.load_embeddings(),
            )

        # --- Batch insertion with retry to respect free-tier rate limits ---
        all_inserted_ids = []
        total = len(documents)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, total, BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"Inserting batch {batch_num}/{total_batches}  ({len(batch)} docs)  [attempt {attempt}]...")
                    inserted_ids = vstore.add_documents(batch)
                    all_inserted_ids.extend(inserted_ids)
                    break  # success — exit retry loop
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                        wait = BATCH_DELAY * attempt  # exponential-ish backoff
                        print(f"  ⚠ Rate limited on batch {batch_num}. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                        time.sleep(wait)
                    else:
                        raise  # re-raise non-rate-limit errors
            else:
                # All retries exhausted for this batch
                print(f"  ✗ Failed to insert batch {batch_num} after {MAX_RETRIES} retries. Skipping.")

            # Wait between batches (skip wait after the last batch)
            if i + BATCH_SIZE < total:
                print(f"Waiting {BATCH_DELAY}s to respect rate limits...")
                time.sleep(BATCH_DELAY)

        print(f"Successfully inserted {len(all_inserted_ids)} documents into vector store.")
        return vstore, all_inserted_ids

    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        documents = self.transform_data()
        vstore, inserted_ids = self.store_in_vector_db(documents)

        # Optionally do a quick search
        query = "Can you tell me the low budget headphone?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()
import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from utils.model_loader import ModelLoader

load_dotenv()

def debug_vector_store():
    """Debug script to check vector store contents"""
    
    model_loader = ModelLoader()
    embeddings = model_loader.load_embeddings()
    
    print("=== Environment Variables ===")
    print(f"ASTRA_DB_API_ENDPOINT: {os.getenv('ASTRA_DB_API_ENDPOINT')}")
    print(f"ASTRA_DB_KEYSPACE: {os.getenv('ASTRA_DB_KEYSPACE')}")
    print(f"Collection Name: Check your config")
    print("==============================\n")
    
    try:
        vstore = AstraDBVectorStore(
            collection_name="customer_support_chatbot_rag",  # Make sure this matches
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            keyspace=os.getenv("ASTRA_DB_KEYSPACE"),
            embedding=embeddings,
        )
        
        # Test 1: Check if we can connect
        print("Connected to AstraDB successfully\n")
        
        # Test 2: Try a simple similarity search
        test_query = "laptop"
        print(f"Testing search with query: '{test_query}'")
        
        results = vstore.similarity_search(test_query, k=3)
        
        if results:
            print(f"✅ Found {len(results)} documents:\n")
            for idx, doc in enumerate(results, 1):
                print(f"--- Document {idx} ---")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}\n")
        else:
            print("❌ No documents found! The collection might be empty.")
            print("   Run the ingestion pipeline first: python data_ingestion/ingestion_pipeline.py")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_vector_store()

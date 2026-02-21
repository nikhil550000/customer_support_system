import os
import logging
from langchain_astradb import AstraDBVectorStore
from typing import List, Tuple
from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.3


class Retriever:

    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_namespace = os.getenv("ASTRA_DB_KEYSPACE")

    def _ensure_vstore(self):
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_namespace,
            )

    def load_retriever(self):
        if self.retriever:
            return self.retriever

        self._ensure_vstore()
        top_k = self.config.get("retriever", {}).get("top_k", 3)
        self.retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
        logger.info("Retriever loaded successfully.")
        return self.retriever

    def call_retriever(self, query: str) -> List[Document]:
        retriever = self.load_retriever()
        return retriever.invoke(query)

    def call_retriever_with_scores(self, query: str) -> Tuple[List[Document], float]:
        """Retrieve documents with similarity scores and filter by relevance threshold."""
        self._ensure_vstore()
        top_k = self.config.get("retriever", {}).get("top_k", 3)
        results = self.vstore.similarity_search_with_score(query, k=top_k)

        if not results:
            return [], 0.0

        scores = [score for doc, score in results]
        avg_score = sum(scores) / len(scores)

        logger.info(f"Retrieved {len(results)} docs, avg similarity: {avg_score:.3f}")

        relevant_docs = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]
        if len(relevant_docs) < len(results):
            logger.info(f"Filtered to {len(relevant_docs)} relevant docs (threshold={RELEVANCE_THRESHOLD})")

        return relevant_docs, avg_score


if __name__ == "__main__":
    retriever_obj = Retriever()
    query = "Can you suggest good budget laptops?"
    docs, avg_score = retriever_obj.call_retriever_with_scores(query)
    print(f"\nQuery: {query}")
    print(f"Avg similarity: {avg_score:.3f}")
    print(f"Relevant docs: {len(docs)}")
    for idx, doc in enumerate(docs, 1):
        print(f"\nResult {idx}: {doc.page_content[:200]}...")
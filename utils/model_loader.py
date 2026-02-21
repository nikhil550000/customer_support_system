import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


class ModelLoader:

    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self._validate_env()
        self._embeddings = None
        self._llm = None

    def _validate_env(self):
        required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def load_embeddings(self):
        if not self._embeddings:
            model_name = self.config["embedding_model"]["model_name"]
            self._embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            logger.info(f"Embedding model loaded: {model_name}")
        return self._embeddings

    def load_llm(self):
        if not self._llm:
            model_name = self.config["llm"]["model_name"]
            self._llm = ChatGroq(model=model_name, api_key=self.groq_api_key)
            logger.info(f"LLM loaded: {model_name}")
        return self._llm
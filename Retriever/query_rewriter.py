import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are a search query optimizer for a product review database.

Given a customer's question and their conversation history, generate a concise search query 
that would best match relevant product reviews in a vector database.

Rules:
- Extract the core product and attributes the user cares about
- Resolve pronouns using conversation history (e.g., "that one" → actual product name)
- Remove filler words, keep only search-relevant terms
- If the user asks a follow-up, merge context from history into the query
- Output ONLY the rewritten query, nothing else

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

REWRITTEN SEARCH QUERY:"""


class QueryRewriter:
    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_template(REWRITE_PROMPT)
            | llm
            | StrOutputParser()
        )

    def rewrite(self, question: str, history: str = "No previous conversation.") -> str:
        rewritten = self.chain.invoke({"question": question, "history": history})
        rewritten = rewritten.strip()
        logger.info(f"Query rewrite: '{question}' → '{rewritten}'")
        return rewritten
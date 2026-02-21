"""
Offline evaluation of RAG pipeline quality.
Measures retrieval relevance and answer faithfulness.

Usage:
    python evaluation/evaluate.py
"""

import sys
import os
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from Retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Test cases: questions with expected traits ---
TEST_CASES = [
    {
        "question": "Can you suggest good budget laptops?",
        "expected_keywords": ["laptop", "budget", "price", "affordable"],
        "should_have_product_name": True,
    },
    {
        "question": "What do customers say about battery life of phones?",
        "expected_keywords": ["battery", "phone", "hours", "charge"],
        "should_have_product_name": True,
    },
    {
        "question": "Tell me about the best rated headphones",
        "expected_keywords": ["headphone", "sound", "quality", "rating"],
        "should_have_product_name": True,
    },
    {
        "question": "What is the capital of France?",
        "expected_keywords": [],
        "should_have_product_name": False,
        "expect_redirect": True,  # Bot should say this is outside scope
    },
    {
        "question": "Are there any complaints about delivery?",
        "expected_keywords": ["delivery", "shipping", "late", "delay"],
        "should_have_product_name": False,
    },
]


def evaluate_retrieval(retriever_obj: Retriever, question: str, expected_keywords: list) -> dict:
    """Check if retrieved documents are relevant to the question."""
    docs = retriever_obj.call_retriever(question)
    combined_text = " ".join([doc.page_content.lower() for doc in docs])

    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

    return {
        "num_docs_retrieved": len(docs),
        "keyword_overlap_score": round(keyword_score, 2),
        "keywords_found": [kw for kw in expected_keywords if kw.lower() in combined_text],
        "keywords_missed": [kw for kw in expected_keywords if kw.lower() not in combined_text],
    }


def evaluate_answer(answer: str, test_case: dict) -> dict:
    """Check basic quality of the generated answer."""
    answer_lower = answer.lower()

    # Check if answer is not empty
    is_non_empty = len(answer.strip()) > 10

    # Check keyword presence in answer
    expected = test_case["expected_keywords"]
    keyword_hits = sum(1 for kw in expected if kw.lower() in answer_lower)
    keyword_score = keyword_hits / len(expected) if expected else 1.0

    # Check if it mentions a product name (any capitalized multi-word or brand-like term)
    has_specific_info = any(char.isupper() for char in answer[10:]) if len(answer) > 10 else False

    # Check if out-of-scope questions are redirected
    if test_case.get("expect_redirect"):
        redirect_phrases = ["i can't help", "outside", "not related", "product", "can't assist",
                            "don't have", "unable", "beyond", "sorry", "redirect"]
        is_redirected = any(phrase in answer_lower for phrase in redirect_phrases)
    else:
        is_redirected = None

    return {
        "is_non_empty": is_non_empty,
        "answer_keyword_score": round(keyword_score, 2),
        "has_specific_info": has_specific_info,
        "is_redirected": is_redirected,
        "answer_length": len(answer),
    }


def run_evaluation():
    logger.info("Starting evaluation...")
    retriever_obj = Retriever()
    model_loader = ModelLoader()
    llm = model_loader.load_llm()
    retriever = retriever_obj.load_retriever()

    from prompt_library.prompt import PROMPT_TEMPLATES
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])

    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "history": lambda x: "No previous conversation.",
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    results = []
    total_retrieval_score = 0
    total_answer_score = 0

    for i, test_case in enumerate(TEST_CASES, 1):
        question = test_case["question"]
        logger.info(f"\n[Test {i}/{len(TEST_CASES)}] {question}")

        # Evaluate retrieval
        retrieval_result = evaluate_retrieval(retriever_obj, question, test_case["expected_keywords"])
        logger.info(f"  Retrieval score: {retrieval_result['keyword_overlap_score']}")

        # Evaluate answer
        answer = chain.invoke({"question": question})
        answer_result = evaluate_answer(answer, test_case)
        logger.info(f"  Answer score: {answer_result['answer_keyword_score']}")
        logger.info(f"  Answer preview: {answer[:100]}...")

        total_retrieval_score += retrieval_result["keyword_overlap_score"]
        total_answer_score += answer_result["answer_keyword_score"]

        results.append({
            "question": question,
            "retrieval": retrieval_result,
            "answer": answer_result,
            "raw_answer": answer,
        })

    # Summary
    n = len(TEST_CASES)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_tests": n,
        "avg_retrieval_score": round(total_retrieval_score / n, 2),
        "avg_answer_score": round(total_answer_score / n, 2),
        "results": results,
    }

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    output_path = "evaluation/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Tests run:            {n}")
    logger.info(f"Avg retrieval score:  {summary['avg_retrieval_score']}")
    logger.info(f"Avg answer score:     {summary['avg_answer_score']}")
    logger.info(f"Results saved to:     {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
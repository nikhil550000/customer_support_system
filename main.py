import os
import logging
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from Retriever.retrieval import Retriever
from Retriever.query_rewriter import QueryRewriter
from utils.model_loader import ModelLoader
from prompt_library.prompt import PROMPT_TEMPLATES
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Product Support Chatbot", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals initialized at startup ---
retriever_obj = Retriever()
model_loader = ModelLoader()
query_rewriter = None
llm = None
prompt = None

MAX_HISTORY_TURNS = 5
conversation_store: dict[str, list[dict]] = defaultdict(list)


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"Customer: {turn['user']}")
        lines.append(f"Assistant: {turn['bot']}")
    return "\n".join(lines)


@app.on_event("startup")
def startup():
    global query_rewriter, llm, prompt
    logger.info("Loading components...")
    retriever_obj.load_retriever()
    llm = model_loader.load_llm()
    query_rewriter = QueryRewriter(llm)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])
    logger.info("All components ready.")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="https://static.vecteezy.com/system/resources/previews/016/017/018/non_2x/ecommerce-icon-free-png.png")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get", response_class=HTMLResponse)
async def chat(request: Request, msg: str = Form(...)):
    if not msg.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        session_id = request.client.host
        history_str = format_history(conversation_store[session_id])

        # Step 1: Rewrite query using LLM
        rewritten_query = query_rewriter.rewrite(msg, history_str)

        # Step 2: Retrieve with confidence scoring
        docs, avg_score = retriever_obj.call_retriever_with_scores(rewritten_query)

        # Step 3: Build context
        if not docs:
            context_str = "No relevant product reviews found for this query."
        else:
            context_str = "\n\n".join([doc.page_content for doc in docs])

        # Step 4: Generate response
        chain_input = {
            "context": context_str,
            "question": msg,
            "history": history_str,
        }
        result = (prompt | llm | StrOutputParser()).invoke(chain_input)

        # Store conversation turn
        conversation_store[session_id].append({"user": msg, "bot": result})
        if len(conversation_store[session_id]) > MAX_HISTORY_TURNS:
            conversation_store[session_id] = conversation_store[session_id][-MAX_HISTORY_TURNS:]

        logger.info(f"[{session_id}] User: {msg}")
        logger.info(f"[{session_id}] Rewritten: {rewritten_query}")
        logger.info(f"[{session_id}] Docs: {len(docs)}, Avg score: {avg_score:.3f}")
        return result
    except Exception as e:
        logger.error(f"Chain invocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
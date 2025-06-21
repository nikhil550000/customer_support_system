import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from Retriever.retrieval import Retriever

from utils.model_loader import ModelLoader

from prompt_library.prompt import PROMPT_TEMPLATES
import os



app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory = "templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

retriever_obj = Retriever()

model_loader = ModelLoader()

def invoke_chain(query: str):
    """
    Invoke the chain with the provided query.
    """
    # Load the retriever
    retriever = retriever_obj.load_retriever()
    
    # Prepare the prompt template
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])
    llm = model_loader.load_llm()


    chain = (
        {"context": retriever,"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    output = chain.invoke(query)

    return output



@app.get("/", response_class=HTMLResponse)

async def index(request:Request):
    """
    Render the main index page.
    """
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get",response_class=HTMLResponse)
async def chat(msg:str = Form(...)):
    result = invoke_chain(msg)
    print(f"Response: {result}")
    return result
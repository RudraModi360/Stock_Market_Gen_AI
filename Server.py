from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from datetime import datetime, date
from langchain_cohere import CohereEmbeddings
import json
from web_scrap import search_urls

groq_api_key = "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"
llm = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview"
)
embeddings_obj = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key="fb1d788zuAEdb83rWpe5MESR6Gx16sI7wu0rHQVP",
)

app = FastAPI()


class Query(BaseModel):
    description: str | None = None


@app.get("/ping")
async def ping():
    return "hi"


@app.post("/predict")
async def predict(query: Query):
    links = []
    loader = WebBaseLoader(
        "https://raw.githubusercontent.com/Ansh-Chamriya/forinstall/main/NSE_EQ.json",
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    json_data = loader.load()
    json_splitted = text_splitter.split_documents(json_data)
    loaded_db = FAISS.load_local(
        "faiss_index", embeddings_obj, allow_dangerous_deserialization=True
    )
    docs = loaded_db.similarity_search("instrument key of RBI")
    json_prompt = ChatPromptTemplate.from_template(
        """
Given today's date {date} (which is {day}), answer user questions accurately and set the start and end dates as per the question. If no period is specified, use 'day' from today to the past 30 days. Choose the time frame from [1minute, 30minute, day, week, month].

Context:
<context>
{context}
</context>

Generate URLs based on user questions as follows:

1. For specific days within the past 6 months:
    - url = 'https://api.upstox.com/v2/historical-candle/:instrument_key/1minute/:end_date'
2. For specific months:
    - url = 'https://api.upstox.com/v2/historical-candle/:instrument_key/30minute/:end_date'
3. For specific years:
    - url = 'https://api.upstox.com/v2/historical-candle/:instrument_key/day/:end_date/:start_date'
4. For periods of days, weeks, or months:
    - url = 'https://api.upstox.com/v2/historical-candle/:instrument_key/day/:end_date/:start_date'

For today's data for Axis Bank:
- url = 'https://api.upstox.com/v2/historical-candle/intraday/:instrument_key/30minute'

Example: 
If the user asks for data of Axis, HDFC, and SBI banks for today, generate:
[
    'https://api.upstox.com/v2/historical-candle/NSE_EQ|INE238A01034/30minute/{date}',
    'https://api.upstox.com/v2/historical-candle/NSE_EQ|INF179KC1DH4/30minute/{date}',
    'https://api.upstox.com/v2/historical-candle/NSE_EQ|INE123W01016/30minute/{date}'
]

Do not use 1minute or 30minute for data older than 6 months.

Questions:
{input}

Generate URLs in JSON format for each company mentioned in the input. If no company is specified, return None.

Example: 
If the prompt is "give the top 5 companies to invest in currently," return None.

Output JSON without extra characters or formatting.

    """
    )
    document_chain = create_stuff_documents_chain(llm, json_prompt)
    retriever = loaded_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke(
        input={
            "input": query.description,
            "context": json_splitted,
            "date": date.today(),
            "day": datetime.today().strftime("%A"),
        }
    )
    data = response["answer"]
    if data==None:
        web_splitted=None
    else:
        if isinstance(data, str):
            data = json.loads(response["answer"])
        print(data)
        print(type(data))
        try:
            if isinstance(data, list):
                print("Entered in region of list forms data block ..........")
                url_dict = {"urls": [d["url"] for d in data]}
                print(url_dict)
                links = url_dict["urls"]
            else:
                print("Entered in region of json forms data block ..........")
                json_object = json.dumps(data, indent=None)
                json_string = json.loads(json_object)
                for key in json_string:
                    if key == "url":
                        links = data["url"]
                    elif key == "urls":
                        links = data["urls"]
            print(links)
        except Exception as e:
            links = search_urls(query.description)
    loader = WebBaseLoader(links)
    web_data = loader.load()
    loader = WebBaseLoader(
        [
            "https://newsapi.org/v2/everything?q=(NSE%20AND%20BSE)&from=2024-07-26&to=2024-07-26&sortBy=popularity&apiKey=d1858fd8650743deb697aea90617d602",
            "https://newsapi.org/v2/everything?domains=moneycontrol.com,cnbc.com,bloomberg.com,thehindubusinessline.com/topic/nse,nseindia.com/resources/exchange-communication-media-center&apiKey=d1858fd8650743deb697aea90617d602",
        ]
    )
    news_web_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    web_splitted = text_splitter.split_documents(web_data)
    news_splitted = text_splitter.split_documents(news_web_data)
    db = FAISS.from_documents(web_splitted, embeddings_obj)
    print(db.similarity_search(query.description))
    print("--" * 25)
    edited_prompt = ChatPromptTemplate.from_template(
        """
    By Using the Context as reference given below give the appropriate answer of the user question:
    <context>
    {context}
    </context>
    question of user should be answer properly with the the actual values from the context if the yesterday date's data is there also use it for generation of response if exists along with articles data.
    If values have to predict then use the context analyse stock data variation trend properly and based on it predict the values with proper reasons also mention the reasons along with some numbering data for predicting the values, use the news for stock-market {news} as a reference. Include the title of the articles and the urls of the news as well which you use .
    else 
    Questions:
    {input}

    give the answer without commenting anything about the context and who is providing the data with response size almost around 1500-10,000 tokens & behave like you are stocks advisor and if prompt is genral talks like [eg. "hi","what can you do for me" ..] and respond according to your role.
    """
    )
    document_chain = create_stuff_documents_chain(llm, edited_prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke(
        input={
            "input": query.description,
            "context": web_splitted,
            "news": news_splitted,
            "date": date.today(),
            "day": datetime.today().strftime("%A"),
        }
    )
    print(response["answer"])
    return response["answer"]


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8000")

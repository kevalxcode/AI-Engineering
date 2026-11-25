from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

# load_dotenv()

# # model = ChatOpenAI(model="gpt-4o", temperature=0)
# model = ChatHuggingFace(model="penai/gpt-oss-120b", tempreture=0)

# res = model.invoke("Can you list top popular online games that user search")

# print(res.content)


import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is missing or empty in .env")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)

res = (model.invoke("where is india"))


print(res.content)
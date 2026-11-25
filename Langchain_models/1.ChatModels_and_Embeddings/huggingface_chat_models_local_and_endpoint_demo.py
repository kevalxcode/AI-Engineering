# Downloading model to our machine
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os


os.environ["HF_HOME"] = "D:/hugginhface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 400},
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("When India get Independent and from who? ")
print("res:", res.content)



# Directly using the model through inference but it requires the credit so here is just demo code to referece
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=400
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)

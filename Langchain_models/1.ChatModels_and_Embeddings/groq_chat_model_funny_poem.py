from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)


result = model.invoke("Write 5 lines of funny poem")
print(result.content)
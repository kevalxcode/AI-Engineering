
from langchain_groq import Groq
from dotenv import load_dotenv

load_dotenv()

model = Groq(
    model="openai/gpt-oss-120b"
)

result = model.invoke(
    "Who is dhoni?"
)

print(result)
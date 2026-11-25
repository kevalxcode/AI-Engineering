from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=1500,
    api_key=api_key
)

chat_history = []


while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    chat_history.append(user_input)
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)



print(chat_history)
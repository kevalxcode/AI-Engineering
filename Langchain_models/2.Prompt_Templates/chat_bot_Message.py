from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")

chat_history = [
    SystemMessage(content='You are helful assistant'),
]

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=1000,
    api_key=api_key
)

while True:

    user_input = input("User:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    print("AI: " , result.content)
    chat_history.append(AIMessage(content=result.content))

print("chat history; ", chat_history)



prompt = ChatPromptTemplate.from_messages([
    ('system', "you are helpfull assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
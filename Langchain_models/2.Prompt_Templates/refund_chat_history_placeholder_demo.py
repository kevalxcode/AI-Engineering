# MessagesPlaceholder dynamically inserts chat history at runtime.
# It helps the chatbot understand earlier user messages and AI responses.

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=1000,
    api_key=api_key,
)

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name="refund_chat_his"),
    ("human", "{query}"),
])

from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="I want to request a refund for my order #12345."),

    AIMessage(content="Your refund request has been initiated. It will be processed in 3-5 business days."),

    HumanMessage(content="Can you tell me when I will receive my refund?"),

    AIMessage(content="Your refund should be credited within 3-5 business days.")
]

# # Load chat history
# with open("refund_chat_his.txt") as f:
#     chat_history.append(f.readlines())


print ("chat is ", chat_history)

while True:
    user_input = input("User: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    prompt = chat_template.invoke({
        "refund_chat_his": chat_history,
        "query": user_input,
    })

    print("Prompt is ", prompt)

    result = model.invoke(prompt)
    print("AI:", result.content)

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))

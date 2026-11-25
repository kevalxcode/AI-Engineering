from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# chat_history = []

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

# if not api_key:
#     raise RuntimeError("GROQ_API_KEY is missing")


# model = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0.7,
#     max_tokens=1000,
#     api_key=api_key
# )

# question = input("Ask you query: ")


# prompt = ChatPromptTemplate.from_messages([
#     ("system", "you are helpful assistant"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{question}")
# ])


# final_prompt = prompt.invoke({
#     "chat_history": chat_history,
#     "question":question
# })

# print("Final prompt ", final_prompt)


# res = model.invoke(final_prompt)

# print("AI:", res.content)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LLM? Explain in 100 words.")
]

result = model.invoke(messages)

print("AI:", result.content)
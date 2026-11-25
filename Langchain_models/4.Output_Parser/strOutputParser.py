# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",)

# # 1st prompt -> Detailed Report
# template1 = ChatPromptTemplate.from_template(
#     template="Write a detailed report on {topic}") 
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
#2nd prompt -> summary

# template2 = ChatPromptTemplate.from_template(
#     template="Write 5 line summary on the following text. /n {text}")
template2 = PromptTemplate(
    template="Write 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({
    "topic": "Blackhole "
})

res1 = model.invoke(prompt1)

print(res1.content)


prompt2 = template2.invoke({
    "text": res1.content
})

res2 = model.invoke(prompt2)

print("-------------------------------------")

print(res2.content)

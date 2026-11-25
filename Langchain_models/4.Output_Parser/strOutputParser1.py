# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

 
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b")

# 1st prompt -> Detailed Report
template1 = ChatPromptTemplate.from_template(
    template="Write a detailed report on {topic} in 300 words") 
#2nd prompt -> summary

template2 = ChatPromptTemplate.from_template(
    template="Write 5 line summary on the following text. /n {text}")


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "Black hole"})

print(result)

# If we do manual without the chain then 
# res1 = model.invoke(prompt1)
# text1 = parser.invoke(res1)  OR
# text1 = parser.parse(res1.content)


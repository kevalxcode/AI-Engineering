from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(  
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Write detailed report on {topic}",
    input_variables=["topic"],
    validate_template= True
)



template2 = PromptTemplate(
    template="Write summary from this {text} report in one paragraph",
    input_variables=["text"]
)

# chain = template1 | model | parser | template2 | model | parser

# res = chain.invoke({"topic" : "Coding 2010 vs 2024"})

# chain.get_graph().print_ascii()

# print("res is ", res)


# Manual 

prompt1 = template1.invoke({"topic" : "Coding 2010 vs 2024"})

res1 = model.invoke(prompt1)

# text1 = parser.parse(res1.content)
# print("text1" , text1)

prompt2 = template2.invoke({"text" : res1.content})

res2 = model.invoke(prompt2)

text2 = parser.invoke(res2)

print("text2 ", text2)



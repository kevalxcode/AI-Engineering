from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=1000
)


prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic} from {time_between}",
    input_variables=["topic", "time_between"]
)

parser = StrOutputParser()


chain = prompt | model | parser

res = chain.invoke({"topic":"USA", "time_between" : "2000 to 2010"})

chain.get_graph().print_ascii()

print("res is ", res)
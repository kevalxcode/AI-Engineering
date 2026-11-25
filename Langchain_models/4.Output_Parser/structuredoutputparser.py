# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.output_parsers import (
    ResponseSchema,
    StructuredOutputParser
)
 
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b")

schema = [
    ResponseSchema(name="fact_1", description='Fact 1 about the topic'),
    ResponseSchema(name="fact_2", description='Fact 2 about the topic'),
    ResponseSchema(name="fact_3", description='Fact 3 about the topic'),
]  

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)

chain = template | model | parser

final_res = chain.invoke({"topic":"black hole"})

# prompt = template.invoke({"topic":"black hole"})

# res = model.invoke(prompt)

# final_res = parser.parse(res.content)

print(final_res)           
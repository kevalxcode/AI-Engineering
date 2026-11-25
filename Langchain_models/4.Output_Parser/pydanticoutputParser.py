# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_classic.output_parsers import (
    ResponseSchema,
    StructuredOutputParser
)
 
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b")

class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age:int = Field(ge=18, description="age of the person")
    city:str = Field(description="Name of the city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of the fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)

prompt = template.invoke({"place" : "UAE"})

res = model.invoke(prompt)

parser = parser.invoke(res)

print("parser ", parser)

# chain = template | model | parser

# res = chain.invoke({"place" : "Indian"})

# print("res" , res)


# First import the req modules
# Second intialize the model
#initialize the schema
# initialize the parser
# write the template, input vari, partial var
# inovke template inside the prompt 
#invoke the prompt inside the model
# print final_result.content
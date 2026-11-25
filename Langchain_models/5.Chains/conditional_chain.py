from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from langchain_classic.output_parsers import (ResponseSchema, StructuredOutputParser, PydanticOutputParser)
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

parser = PydanticOutputParser(pydantic_object=Feedback) 

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postivie or negative \n {feedback} \n {format_instruction}',
    input_variables=["feedback"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser


prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback'],
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this Negative feedback \n {feedback}",
    input_variables=['feedback'],
    
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

chain.invoke({"feedback": "this is terrible phone"})

# If I Use Strucutredoutputparser
# schema = [
#     ResponseSchema(name="Sentiment", description="Write only positive, negative and nuetral based on User Feedback.")
# ]

# parser = StructuredOutputParser.from_response_schemas(schema)

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into postivie or negative \n {feedback} \n {format_instruction}',
#     input_variables=["feedback"],
#     partial_variables={"format_instruction" : parser.get_format_instructions()}
# )



# res = classifier_chain.invoke({"feedback" : "This is not good smartphone"})

# print("res", res) 
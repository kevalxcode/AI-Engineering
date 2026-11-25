# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

 
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b")


parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me 4 facts abut {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={'format_instruction': parser.get_format_instructions()},
)

# prompt = template.format()
# res = model.invoke(prompt)
# print("res ", res)
# final_res = parser.parse(res.content)
# print("final  ", final_res)


chain = template | model | parser

final_res = chain.invoke({"topic":"Black hole"})

print(final_res)

# manual coding# Step 1: Create prompt
# prompt = template.invoke({"topic": "Black hole"})

# # Step 2: Send prompt to LLM
# res = model.invoke(prompt)

# # Step 3: Parse LLM response
# final_res = parser.parse(res.content)

# # Step 4: Print result
# print(final_res)

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, load_prompt
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=1000,
    api_key=api_key
)

st.header("Research tool")


paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)



template = load_prompt('research_paper_prompt_template.json')


prompt = template.invoke({
    'paper_input': paper_input,
    "style_input": style_input,
    "length_input":length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)


    st.write(result.content)

# # This is the simple way one way we can communicate with the ai model


# user_input = input("could you enter your prompt: ")

# # now there is another method to talk with the ai model which is Prompt

# prompt = f"""

# User will ask you the question. you have to act as the senior doctor that give advice to the user, 
# if there is bad points then also include other wise there is no need 
# ut always give real information to the user

# Now classify this message:
# user_query : {user_input}

# """

# res = model.invoke(prompt)

# print(res.content)


# template1 = PromptTemplate(
#    template= "My name is {nameIs}"
#     "Can u pass me {thing}"    
# )

# prompt = template1.invoke({
#     "nameIs":"Keval",
#     "thing": "salt"
# })


# model.invoke(prompt)
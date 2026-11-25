#OpenAI Embedding
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv


# load_dotenv()


# embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32) 

# documents = [
#     "Delhi is the capital of India",
#     "Kolkta is the capital of wsb",
#     "India is not best coutnry"
# ]

# # single query geenrate
# result = embedding.embed_query("I love working with Chatgpt")

# # document embedding

# res = embedding.embed_documents(documents)

# print(str(res))


# Hugging face Embedding

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text1 = "I'm testing this feature as Developer"
text2 = "I am a programmer testing a new feature"
text3 = "The weather is very cold today"

result1 = embedding.embed_query(text1)
# result2 = embedding.embed_query(text2)
# result3 = embedding.embed_query(text3)
num1 = str(result1[:5])
print(num1[0])
print("Vector dimensions:", len(result1))
print("First sentence:", num1)
print('Type is ', type(num1))
# print("Second sentence:", result2[:5])
# print("Third sentence:", result3[:5])
# Let practice How embedding works with the semantic search inside the application


from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


cricketers = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting.",
    "He is one of the best modern-day batsmen in international cricket.",
    "Kohli is famous for his consistency in chasing targets.",
    "He has scored centuries across Test, ODI, and T20 formats.",
    "He served as captain of the Indian cricket team.",
    "His batting style combines excellent timing, fitness, and aggression.",

    "Sachin Tendulkar is a legendary former Indian cricketer.",
    "He is widely known as the Master Blaster.",
    "Sachin was one of the greatest batsmen in cricket history.",
    "He was the first player to score 200 runs in an ODI innings.",
    "He scored 100 international centuries during his career.",
    "Sachin won the Cricket World Cup with India in 2011.",

    "MS Dhoni is a former Indian cricketer and wicketkeeper.",
    "He is famous for his calm personality and finishing ability.",
    "Dhoni led India to victory in the 2007 T20 World Cup.",
    "He captained India to the 2011 ODI World Cup victory.",
    "He is known for his helicopter shot and quick stumpings.",
    "Dhoni is considered one of the greatest captains in cricket.",

    "Virender Sehwag is a former Indian opening batsman.",
    "He was famous for his fearless and aggressive batting.",
    "Sehwag scored two triple centuries in Test cricket.",
    "He was known for attacking bowlers from the beginning.",
    "He scored a double century in ODI cricket.",
    "Sehwag played an important role in India's batting lineup.",

    "Jasprit Bumrah is an Indian fast bowler.",
    "He is famous for his unique bowling action.",
    "Bumrah is known for his accurate yorkers and pace.",
    "He performs exceptionally well in death-over bowling.",
    "He represents India in Test, ODI, and T20 cricket.",
    "Bumrah is considered one of the best fast bowlers in modern cricket.",

    "Bhuvneshwar Kumar is an Indian right-arm medium-fast bowler.",
    "He is famous for his ability to swing the cricket ball.",
    "He can move the ball in both directions.",
    "Bhuvneshwar is particularly effective with the new ball.",
    "He has represented India in all three international formats.",
    "He is known for his accuracy and controlled bowling."
]

doc_embedding = embedding.embed_documents(cricketers)

query = "Tell me about Bhuvneshwar"


query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)


index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]
print(query)
print(cricketers[index])
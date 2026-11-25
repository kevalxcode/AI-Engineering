from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
 
load_dotenv()

model1 = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


model2 = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9) 


prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge and provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

print("before parallel_chain")
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})
print("after parallel_chain")

merge_chain = prompt3 | model1 | parser

print("before chain")
chain = parallel_chain | merge_chain
print("after chain")


text = """
Generative Artificial Intelligence (Gen AI)

Generative Artificial Intelligence, commonly known as Generative AI or Gen AI, is one of the most important technological developments in the modern digital era. It is a branch of artificial intelligence that focuses on creating new and original content based on patterns learned from existing data. Unlike traditional artificial intelligence systems that mainly classify information, identify patterns, or make predictions, generative AI can produce different types of content, including text, images, audio, videos, music, and computer programs. This technology has transformed the way individuals, businesses, educational institutions, and industries interact with digital information. Generative AI uses advanced machine learning algorithms, deep learning techniques, neural networks, and large datasets to understand information and generate meaningful outputs. It has become an important part of modern technology because it can perform various creative and analytical tasks with speed and efficiency.

The development of generative AI has changed the relationship between humans and computers. In the past, computers were mainly used to perform specific instructions provided by users. However, modern AI systems can understand natural language, process complex requests, and generate responses that resemble human communication. Users can interact with generative AI by providing instructions known as prompts. Based on these prompts, the system generates suitable responses. For example, a user can ask an AI system to write an article, explain a scientific concept, create a business proposal, generate a computer program, or summarize a long document. This makes generative AI useful for people with different levels of technical knowledge.

Generative AI works by learning patterns and relationships from large amounts of training data. During the training process, AI models analyze information and develop mathematical representations of the relationships between different data elements. These models use the learned patterns to generate new content when they receive a user request. One of the most important technologies behind generative AI is deep learning, which uses artificial neural networks inspired by the structure of the human brain. These networks contain multiple layers that process information and identify complex patterns. Large language models are a major category of generative AI systems that are designed to understand and generate human language. They can perform tasks such as answering questions, translating languages, writing essays, summarizing documents, and generating conversations.

Generative AI has several important applications in the field of education. Students can use AI tools to understand difficult subjects, prepare notes, practice questions, and improve their writing skills. Teachers can use these systems to prepare lesson plans, create educational materials, and develop assessment questions. AI-powered educational platforms can provide personalized learning experiences according to the knowledge and learning speed of individual students. Generative AI can also help students learn programming by explaining code, identifying errors, and suggesting improvements. However, students should use AI as a learning assistant rather than depending entirely on automatically generated answers. Independent thinking, creativity, research, and problem-solving skills remain essential for academic success.

In the healthcare industry, generative AI has the potential to support medical research, healthcare administration, and clinical documentation. It can help summarize medical literature, organize patient information, generate administrative reports, and assist researchers in analyzing scientific information. AI systems can also support the development of educational resources for healthcare professionals. However, medical decisions require professional knowledge and careful verification. Generative AI should not independently replace qualified doctors or other healthcare professionals. Accuracy, patient confidentiality, data security, and ethical responsibility are essential when using AI in healthcare environments.

Generative AI is also transforming the business world. Organizations use AI-powered applications to improve productivity, automate repetitive activities, and develop better customer experiences. Businesses can generate marketing content, prepare reports, analyze customer feedback, create product descriptions, and develop communication strategies using generative AI tools. Customer service departments can use AI assistants to respond to common questions and provide information to customers. Marketing professionals can use AI to generate creative campaign ideas and personalized content. Software companies can use AI coding assistants to generate code, explain programming concepts, and support debugging activities. These applications can reduce the time required for certain tasks and allow employees to focus on more complex responsibilities.

Another important application of generative AI is content creation. Writers, designers, musicians, filmmakers, and digital creators can use AI systems to develop new ideas and produce creative materials. Text generation tools can create stories, articles, blogs, advertisements, and professional documents. Image generation systems can transform written descriptions into visual artwork. Audio generation systems can produce speech, sound effects, and music. Video generation tools can create visual sequences based on text or image instructions. These capabilities have expanded the possibilities of digital creativity and introduced new methods of producing multimedia content.

"""

res = chain.invoke({"text": text})

chain.get_graph().print_ascii()


print("res ", res)
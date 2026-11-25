
# What is a Prompt?

A prompt is the input given to an AI model to guide or request its output.

### Types of Prompts

1. **Text-based prompts:** Instructions, questions, context and examples provided through text.
2. **Multimodal prompts:** Inputs containing combinations of text, images, audio, video, etc., depending on model capabilities.

### Why are Prompts Important?

The quality, clarity and relevance of a prompt can significantly affect the model's output.

---

# Providing Prompts to Chat Models through LangChain

There are two common ways to provide input to a Chat Model.

## 1. Single-message Input (Single-turn Interaction)

A single input is sent to the model.

### A. Static Message

Directly provide a fixed message.

```python
result = model.invoke("What is LangChain?")
```

### B. Dynamic Message using PromptTemplate

Create a reusable text template with variables.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Explain {topic} in simple words.",
    input_variables=["topic"]
)

formatted_prompt = prompt.invoke({"topic": "LangChain"})
result = model.invoke(formatted_prompt)
```

Here, `{topic}` is a dynamic variable whose value can change.

## 2. List of Messages (Multi-message Interaction)

Instead of passing one plain text input, we provide a sequence of messages with different roles.

### A. Static Messages

Use the following message classes:

- **SystemMessage:** Provides instructions about the model's behaviour.
- **HumanMessage:** Represents the user's message.
- **AIMessage:** Represents the AI's response.

Example:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a Python tutor."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="What is a variable?")
]

result = model.invoke(messages)
```

### B. Dynamic Messages using ChatPromptTemplate

Create a reusable template for a sequence of chat messages and dynamically insert variables.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "Explain {topic} in simple words.")
])

formatted_messages = chat_prompt.invoke({
    "role": "Python tutor",
    "topic": "OOP"
})

result = model.invoke(formatted_messages)
```

Here, both `{role}` and `{topic}` are dynamic variables.

---

# Difference Between PromptTemplate and ChatPromptTemplate

| Concept | Purpose |
|---|---|
| PromptTemplate | Creates a reusable text prompt. |
| ChatPromptTemplate | Creates a reusable sequence of chat messages. |
| Chat History | Stores a sequence of actual conversation messages. |

**Important:** ChatPromptTemplate does not automatically remember previous conversations. It creates messages from a template. Chat history is a separate concept.

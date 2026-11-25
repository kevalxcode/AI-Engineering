
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1000,
    api_key=api_key
)

# Define the expected output structure
from pydantic import BaseModel, Field
from typing import Optional, Literal


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="List all key themes discussed in the review."
    )

    summary: str = Field(
        description="A brief summary of the review."
    )

    sentiment: Literal["pos", "neg", "neu"] = Field(
        description="Classify the overall sentiment of the review."
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="List all the pros mentioned in the review."
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="List all the cons mentioned in the review."
    )

    name: Optional[str] = Field(
        default=None,
        description="Name of the reviewer, if mentioned."
    )

# JSON Example 
review_schema = {
    "title": "Review",
    "description": "Structured analysis of a mobile phone review.",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List all key themes discussed in the review."
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review."
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neu"],
            "description": "Overall sentiment: pos for positive, neg for negative, neu for neutral."
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List all the pros mentioned in the review."
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "List all the cons mentioned in the review."
        },
        "name": {
            "type": ["string", "null"],
            "description": "Name of the reviewer, if mentioned."
        }
    },
    "required": [
        "key_themes",
        "summary",
        "sentiment",
        "pros",
        "cons",
        "name"
    ],
    "additionalProperties": False
}

# Enable structured output
structured_model = model.with_structured_output(review_schema)

# Invoke the model
res = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
Reviewd by Keval

""")


print(res)


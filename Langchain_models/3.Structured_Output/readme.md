# Structured Output in LangChain

## 1. What is Structured Output?

In LangChain, **Structured Output** refers to instructing a Language Model (LLM) to return its response in a predefined structure or format, such as JSON, instead of unstructured, free-form text.

This makes the model's output more predictable and easier to:
- Parse
- Validate
- Store in databases
- Process programmatically
- Pass to other tools and APIs

### Example

Instead of returning a paragraph:

```text
The product arrived damaged. The customer wants a refund
for order 12345, worth ₹2,000.
```

The LLM can return:

```json
{
    "issue": "damaged product",
    "order_id": "12345",
    "refund_amount": 2000
}
```

Now our application can directly access individual fields and perform further operations.

---

## 2. What Problem Does Structured Output Solve?

Normally, when we ask an LLM a question, it returns a natural-language response.

For example:

```python
response = model.invoke(
    "Analyze this Samsung phone review and tell me its pros and cons."
)

print(response.content)
```

The model might return:

```text
The Samsung S24 Ultra has an excellent camera
and powerful processor.

Pros:
- Excellent camera
- Fast processor
- Good battery life

Cons:
- Expensive
- Heavy
```

This response is readable for humans, but creates a problem for developers.

**How will our application reliably extract the pros, cons, and sentiment from this response?**

We could manually parse the text, but that becomes unreliable when the LLM changes its response format.

Structured Output solves this problem by giving the LLM a predefined output structure.

### Example of structured response

```json
{
    "key_themes": [
        "Camera quality",
        "Battery life",
        "Performance"
    ],
    "summary": "The Samsung S24 Ultra offers excellent performance and camera quality but is expensive and bulky.",
    "sentiment": "pos",
    "pros": [
        "Excellent camera",
        "Fast processor",
        "Good battery life"
    ],
    "cons": [
        "Expensive",
        "Heavy"
    ],
    "name": "Keval"
}
```

Now our application knows exactly where to find each piece of information.

---

## 3. Real-World Use Cases

### 3.1 Data Extraction (Resume Parsing)

Suppose we want to extract specific information from resumes.

We can instruct an LLM to extract:

- Candidate name
- Previous companies
- Work experience
- Technical skills
- Projects

The LLM returns structured data that can be stored in a database and used for further processing.

### 3.2 API Building (Product Review Analysis)

Imagine Amazon has thousands of product reviews.

We can use an LLM to extract information such as:

- Product category
- Keywords (battery, processor, performance)
- Pros and cons
- Customer sentiment (positive, negative, neutral)
- Main topics discussed

We can store this extracted information in a database and build APIs around it.

Other applications can consume these APIs to retrieve meaningful information without processing thousands of raw reviews themselves.

### 3.3 AI Agents and Tool Calling

AI agents often need to interact with external tools to perform tasks.

For example, suppose a user asks:

"Calculate the total price of 5 laptops, each costing ₹50,000."

The agent needs to identify:

```json
{
    "quantity": 5,
    "price": 50000
}
```

It can then provide these values as structured arguments to a calculator tool.

The calculator performs the calculation and returns the result.

**Important:** Tools generally require inputs that follow their expected schema. An agent needs to extract and provide the correct arguments rather than simply passing an unstructured paragraph.

### 3.4 Customer Support Automation

Suppose a customer says:

"I want a refund because my order #12345 arrived damaged."

The LLM can extract:

```json
{
    "issue": "Damaged product",
    "order_id": "12345",
    "request_type": "Refund"
}
```

The backend can use this information to verify the order and initiate the appropriate workflow.

### 3.5 Database Operations

Suppose a user asks:

"Find all employees earning more than ₹50,000."

The LLM can extract:

```json
{
    "table": "Employees",
    "condition": {
        "column": "salary",
        "operator": ">",
        "value": 50000
    }
}
```

Your application can use these values to construct and execute a database query after validating the request and applying appropriate security restrictions.

---

## 4. Different Ways to Define Structured Output

We explored three approaches to defining the expected output structure:

1. TypedDict
2. Pydantic
3. JSON Schema

All three describe what the expected output should look like, but they differ in how we define and validate that structure.

### 4.1 TypedDict

`TypedDict` is a way to define the expected structure of a Python dictionary using type hints.

It specifies:
- Which keys should exist.
- What types of values those keys should contain.

Example:

```python
from typing import TypedDict

class Student(TypedDict):
    name: str
    age: int
    email: str
```

Example dictionary:

```python
student: Student = {
    "name": "Keval",
    "age": 21,
    "email": "keval@gmail.com"
}
```

**Important points:**

- Defines the expected dictionary structure.
- Provides type hints for better code readability and static type checking.
- Does not perform runtime validation by itself.
- Python's static type checkers can identify type-related mistakes.

### 4.2 Pydantic

Pydantic allows us to define data structures using Python classes and provides runtime data validation.

Example:

```python
from pydantic import BaseModel, Field

class Student(BaseModel):
    name: str = Field(description="Student's name")
    age: int = Field(description="Student's age")
    email: str = Field(description="Student's email")
```

Creating an object:

```python
student = Student(
    name="Keval",
    age=21,
    email="keval@gmail.com"
)

print(student)
print(student.name)
```

**Important points:**

- Defines schemas using Python classes.
- Supports runtime validation.
- Can perform type coercion depending on configuration.
- Supports constraints, defaults, and field descriptions.
- Allows accessing fields using dot notation.

### 4.3 JSON Schema

JSON Schema defines the expected structure using a JSON-compatible dictionary.

Example:

```python
student_schema = {
    "title": "Student",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Student's name"
        },
        "age": {
            "type": "integer",
            "description": "Student's age"
        },
        "email": {
            "type": "string",
            "description": "Student's email"
        }
    },
    "required": ["name", "age", "email"],
    "additionalProperties": False
}
```

**Important points:**

- Defines the expected output using a dictionary.
- Uses properties to define fields.
- Uses types to define expected value types.
- Uses required to specify mandatory fields.
- Uses enum to restrict values to predefined options.
- Does not require defining a Python class.

### Comparison

| Feature | TypedDict | Pydantic | JSON Schema |
|---|---|---|---|
| Definition | Python class | Python class | Dictionary |
| Type hints | Yes | Yes | Schema types |
| Runtime validation | No, by itself | Yes | Requires a validator |
| Field descriptions | Not directly built in | `Field()` | `description` |
| Field restrictions | Type hints | Validators and constraints | Schema constraints |

---

## 5. What is `with_structured_output()`?

`with_structured_output()` is a LangChain method that connects our schema with the LLM.

Example:

```python
structured_model = model.with_structured_output(Review)
```

Let's understand each part:

1. `model` is our LLM, such as ChatGroq.
2. `with_structured_output()` configures the model to return structured data.
3. `Review` is the schema defining the expected response.
4. LangChain uses the schema to guide the model's output and process the result.

### Invoking the model

```python
res = structured_model.invoke(
    "Samsung S24 Ultra has an excellent camera but poor battery life."
)

print(res)
```

We receive structured data instead of ordinary free-form text.

### Accessing the response

The way we access the returned fields depends on the schema:

**Pydantic:**

```python
print(res.sentiment)
```

**TypedDict or JSON Schema:**

```python
print(res["sentiment"])
```

**Note:** The exact return format can depend on the LangChain integration and configuration.

---

## 6. Practical Implementation: Mobile Phone Review Analysis

Imagine we are building an AI-powered customer review analysis application.

### Workflow

```text
User submits a mobile phone review
              |
              v
       Natural Language Input
              |
              v
    LLM + Structured Output
              |
              v
     Extract information using
           our schema
              |
              v
       Structured Response
              |
        +-----+-----+
        |           |
        v           v
       Pros        Cons
        |           |
        v           v
  Excellent      Expensive
    camera         Heavy
        |           |
        +-----+-----+
              |
              v
     Frontend displays review
           analysis
              |
              v
       React UI, charts,
       filters, etc.
```

### Other applications

- Resume parsing: Extract name, skills, experience, and education.
- Invoice processing: Extract invoice number, amount, and date.
- Customer support: Categorize customer complaints.
- Sentiment analysis: Classify customer feedback.
- Data extraction: Convert unstructured documents into structured records.

---

## 7. Important Points to Remember

### 7.1 Structured Output does not guarantee factual accuracy

Structured Output does NOT mean the LLM always provides factually correct information.

For example, if a review doesn't mention the reviewer's name, the model might still hallucinate one if the instructions are weak.

Structured Output primarily helps with:

- Output format
- Schema compliance
- Predictable data structure

It does not guarantee factual correctness.

You should still validate the extracted data and handle missing or incorrect information.

### 7.2 Structured Output vs Tool Calling

These concepts are related but different.

| Structured Output | Tool Calling |
|---|---|
| Returns data following a predefined schema. | Allows the model to request execution of an external tool. |
| Focuses on the response structure. | Focuses on interacting with external functionality. |
| Example: Extracting review sentiment. | Example: Calling a calculator to calculate a total. |

---

## 8. Key Takeaway

**Structured Output bridges the gap between natural language and application logic.**

It allows LLMs to convert unstructured user input into predictable, machine-readable data that applications can validate, store, retrieve, and process.

As developers, this makes it easier to integrate LLMs into real-world applications, APIs, databases, and AI agents.
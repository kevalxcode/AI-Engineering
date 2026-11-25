# Output Parser

- In langchain help convert raw LLM response into strucutrd formats like JSON,CSV, pydantic model, and more.
- They ensure consistency, validation and ease of use in application


StrOutputParser
- It is simpliest output parser in Langchain. IT used to parse the output of LLM & return it as plain string. 

JsonoutputParser

-Structured output parser
- it is used to extarct the struct json data from the LLM resoinse based on the predefined field schemas.
- it works by defining list of fields that the mnodel should return. ensuring the output follows strcutured format
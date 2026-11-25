from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int


new_person : Person = {
    'name' : "Testing",
    "age" : 22
}

print(new_person)


 
# # TypeDict
# - TypeDict is a way to define a dictinory in python where u specify what keys and values should exit. It helps ensure what your dictionary fllows a specific structure

# - IT tells python what keys are required and what types values they should have
# - it donesnt validate data at runtime, it just give type hints for better coding
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name:str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="It will show the decimal value of student score")


new_student = {
    "name" : "Keval",
    "age" : "21",
    "email" : "timepas@gmail.com",
    "cgpa": 1
    }

student = Student(**new_student)

student_dict = dict(student)
# print(dict(student))
print(student_dict)

student_json = student.model_dump_json()
print("JSON ", student_json)

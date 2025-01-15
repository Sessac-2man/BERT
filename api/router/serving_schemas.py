from pydantic import BaseModel

class TextInput(BaseModel):
    texts: list[str]

class ClassificationResult(BaseModel):
    text: str
    label: str
    score: float
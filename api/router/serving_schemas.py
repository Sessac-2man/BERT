from pydantic import BaseModel

class ModelRequest(BaseModel):
    request : str 
    

class ModelResponse(BaseModel):
    response : str 
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from prompt_template import create_chat_messages

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Initialize OpenAI client in function to avoid startup errors
def get_openai_client():
    return openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.get("/")
async def read_root():
    return FileResponse("modern_chat.html")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"Received message: {request.message}")
        client = get_openai_client()
        print("OpenAI client created")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=create_chat_messages(request.message),
            temperature=0.7
        )
        
        print("OpenAI response received")
        return ChatResponse(response=response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {str(e)}")
        return ChatResponse(response=f"Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
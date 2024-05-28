from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Define the model ID
model_id = 'CampAIgn/Phi-3-mini_16bit'

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

# Create the HuggingFacePipeline instance
local_llm = HuggingFacePipeline(pipeline=pipe)

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve static files from the React build directory
app.mount("/static", StaticFiles(directory="build/assets"), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react_app(full_path: str):
    # Serve the index.html for any path that doesn't match an API route
    return StaticFiles(directory="build", html=True).get_response("index.html")


# Define a request body model
class TextGenerationRequest(BaseModel):
    prompt: str


# Define a response model (optional, but recommended for clarity)
class TextGenerationResponse(BaseModel):
    generated_text: str


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        # Use the local_llm pipeline to generate text
        result = local_llm.pipeline(request.prompt)
        # Extract the generated text from the result
        generated_text = result[0]['generated_text']
        return TextGenerationResponse(generated_text=generated_text)
    except Exception as e:
        # Handle exceptions and return a 500 error with the exception message
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")


# Run the app using `uvicorn` (uncomment to run directly from the script)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

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
app.mount("/", StaticFiles(directory="build", html=True), name="static")


# Define a request body model
class TextGenerationRequest(BaseModel):
    prompt: str


# Define a response model
class TextGenerationResponse(BaseModel):
    generated_text: str


@app.post("/#/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        # Use the local_llm pipeline to generate text
        result = local_llm.pipeline(request.prompt)
        # Extract the generated text from the result
        generated_text = result[0]['generated_text']
        return TextGenerationResponse(generated_text=generated_text)
    except Exception as e:
        # Handle exceptions and return a 500 error with the exception message
        raise HTTPException(status_code=500, detail=str(e))


# Run the app using `uvicorn` (uncomment to run directly from the script)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

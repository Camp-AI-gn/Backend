from fastapi import FastAPI, HTTPException , File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

# Define the model ID
model_id = 'CampAIgn/Phi-3-mini_16bit'


# Initialize FastText embeddings for text encoding
embedding = FastEmbedEmbeddings()


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


# Define a request body model
class TextGenerationRequest(BaseModel):
    prompt: str


# Define a response model
class TextGenerationResponse(BaseModel):
    generated_text: str




#Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=80,  
    length_function=len,  
    is_separator_regex=False  
)


raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] Vous êtes une IA générateur d'histoires de jeux vidéo chargée de créer des récits captivants pour de nouveaux jeux. Si vous ne parvenez pas à trouver une réponse dans les informations fournies, veuillez l'indiquer. [/INST] </s>
    [INST] {input}
           Contexte : {contexte}
           Réponse :
    [/INST]
"""
)



class PDFResponse(BaseModel):
    """
    Represents a response containing an answer and a list of sources.
    """
    answer: str  
    sources: list 
    
    
    

@app.post("/search_pdf")
async def search_pdf(query: Query):
    print("Post /search_pdf called")
    print(f"query: {query.query}")
    print("Loading vector store")
    vector_store = Chroma(persist_directory="Docs", embedding_function=embedding)
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    document_chain = create_stuff_documents_chain(retriever, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query.query})
    print(result)
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )
    return {"answer": result["answer"], "sources": sources}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_name = file.filename
    save_file = "Docs/" + file_name
    with open(save_file, "wb") as buffer:
        buffer.write(await file.read())
    print(f"filename: {file_name}")
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory="Docs"
    )
    vector_store.persist()
    return {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }












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
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files from the React build directory
app.mount("/", StaticFiles(directory="build", html=True), name="static")


# Run the app using `uvicorn` (uncomment to run directly from the script)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

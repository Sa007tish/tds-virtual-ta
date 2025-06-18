from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import os
from openai import OpenAI
from PIL import Image
import pytesseract
import base64
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware  # <-- Added import

# Load embeddings and chunks
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "../all_embeddings.npy")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../all_chunks.json")
embeddings = np.load(EMBEDDINGS_PATH)
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # Optional
client_openai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

app = FastAPI()

# Add CORS middleware immediately after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    question: str
    image: str = None  # base64, optional

def get_embedding(text):
    response = client_openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)

def ocr_from_base64(b64str):
    try:
        img_bytes = base64.b64decode(b64str)
        img = Image.open(BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""

@app.post("/api/")
async def answer_query(req: QueryRequest):
    # Step 1: OCR if image present
    ocr_text = ""
    if req.image:
        ocr_text = ocr_from_base64(req.image)

    # Step 2: Combine question and OCR text
    full_query = req.question
    if ocr_text.strip():
        full_query += "\n\nImage text:\n" + ocr_text.strip()

    # Step 3: Embed query and retrieve top chunks
    query_emb = get_embedding(full_query)
    sims = cosine_similarity(query_emb, embeddings)
    top_k = 5
    top_idx = np.argsort(sims)[-top_k:][::-1]
    context_chunks = [all_chunks[i] for i in top_idx]

    # Step 4: Build LLM prompt
    context_text = "\n\n".join(f"Source: {c.get('source')}\n{c.get('text')}" for c in context_chunks)
    prompt = (
        "You are a helpful virtual TA for the Tools in Data Science course. "
        "Answer the following question using only the provided context. "
        "Cite links where relevant.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {req.question}\nAnswer:"
    )

    # Step 5: Call LLM
    completion = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content.strip()

    # Step 6: Build links list (from the top chunks)
    links = []
    for c in context_chunks:
        url = c.get("source")
        text = (c.get("text") or "")[:120].replace("\n", " ")
        links.append({"url": url, "text": text})

    # Step 7: Return response
    return {"answer": answer, "links": links}
    
@app.get("/")
def root():
    return {"message": "API is running"}



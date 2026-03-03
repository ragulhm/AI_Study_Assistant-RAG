import fitz
import os
import numpy as np
from django.shortcuts import render
from sentence_transformers import SentenceTransformer
from google import genai

from .models import Document
from .chunker import chunk_text
from .retriever import retrieve


# ✅ Load local embedding model (runs once)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Gemini client (only for generation)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# 🔹 Generate local embedding
def generate_embedding(text):
    return embedding_model.encode(text).tolist()


# 🔹 Home + Ask Question
def ask_page(request):
    if request.method == "POST":
        question = request.POST.get("question")

        if not question:
            return render(request, "index.html", {
                "answer": "Please enter a question."
            })

        # 1️⃣ Embed question
        query_embedding = generate_embedding(question)

        # 2️⃣ Retrieve relevant chunks
        context_chunks = retrieve(query_embedding)

        context_text = "\n\n".join(context_chunks)

        # 3️⃣ Strict prompt
        prompt = f"""
You are an academic AI tutor.

Answer the question strictly using the provided context.

Guidelines:
- Base your answer only on the context.
- If insufficient information exists, respond exactly: NOT FOUND
- Keep the explanation clear and structured.
- Use bullet points if helpful.
- Do not reference the context explicitly.

Context:
{context_text}

Question:
{question}

Answer:
"""

        # 4️⃣ Gemini generation
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return render(request, "index.html", {
            "answer": response.text
        })

    return render(request, "index.html")


# 🔹 Upload PDF
def upload_pdf(request):
    if request.method == "POST":
        file = request.FILES.get("file")

        if not file:
            return render(request, "index.html", {
                "answer": "No file uploaded."
            })

        # Extract text
        text = ""
        pdf = fitz.open(stream=file.read(), filetype="pdf")

        for page in pdf:
            text += page.get_text()

        # Chunk text
        chunks = chunk_text(text)

        # Store chunks + embeddings
        for chunk in chunks:
            embedding = generate_embedding(chunk)

            Document.objects.create(
                file_name=file.name,
                chunk_text=chunk,
                embedding=embedding
            )

        return render(request, "index.html", {
            "answer": "PDF processed and embeddings stored successfully."
        })

    return render(request, "index.html")
import fitz
import os
import requests
from django.shortcuts import render
from sentence_transformers import SentenceTransformer

from .models import Document
from .chunker import chunk_text
from .retriever import retrieve


# 🔹 Load local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAME = "openai/gpt-3.5-turbo"  
# You can change to:
# "openai/gpt-4o"
# "meta-llama/llama-3-70b-instruct"
# etc.


# 🔹 Generate local embedding
def generate_embedding(text):
    return embedding_model.encode(text).tolist()


# 🔹 Call OpenRouter API
def generate_answer_with_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an academic AI tutor."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"OpenRouter Error: {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"]


# 🔹 Home + Ask Question
def ask_page(request):
    books = Document.objects.values_list("file_name", flat=True).distinct()

    if request.method == "POST":
        question = request.POST.get("question")

        if not question:
            return render(request, "index.html", {
                "answer": "Please enter a question.",
                "books": books
            })

        # 1️⃣ Embed question
        query_embedding = generate_embedding(question)

        # 2️⃣ Retrieve relevant chunks
        context_chunks = retrieve(query_embedding)

        if not context_chunks:
            return render(request, "index.html", {
                "question": question,
                "answer": "NOT FOUND",
                "books": books
            })

        context_text = "\n\n".join(context_chunks)

        # 3️⃣ Strict grounded prompt
        prompt = f"""
You are a strict academic assistant.

Your task is to answer the question ONLY using the provided context.

STRICT RULES:
1. Use ONLY the information present in the context.
2. Do NOT use outside knowledge.
3. Do NOT make assumptions.
4. If the answer is not explicitly present in the context, respond exactly with:
NOT FOUND
5. Do NOT explain that the answer comes from context.
6. Keep the answer clear, structured, and concise.
7. If possible, format the answer in bullet points or short paragraphs.
8. Do NOT add examples unless they exist in the context.

CONTEXT:
----------------
{context_text}
----------------

QUESTION:
{question}

FINAL ANSWER:
"""

        try:
            answer_text = generate_answer_with_openrouter(prompt)
        except Exception as e:
            answer_text = f"System Error: {str(e)}"

        return render(request, "index.html", {
            "question": question,
            "answer": answer_text,
            "books": books
        })

    return render(request, "index.html", {"books": books})


# 🔹 Upload PDF
def upload_pdf(request):
    books = Document.objects.values_list("file_name", flat=True).distinct()

    if request.method == "POST":
        file = request.FILES.get("file")

        if not file:
            return render(request, "index.html", {
                "answer": "No file uploaded.",
                "books": books
            })

        # Remove old chunks if same file re-uploaded
        Document.objects.filter(file_name=file.name).delete()

        # Extract text
        text = ""
        pdf = fitz.open(stream=file.read(), filetype="pdf")

        for page in pdf:
            text += page.get_text()

        if not text.strip():
            return render(request, "index.html", {
                "answer": "Could not extract text from PDF.",
                "books": books
            })

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

        books = Document.objects.values_list("file_name", flat=True).distinct()

        return render(request, "index.html", {
            "answer": f"{file.name} processed successfully.",
            "books": books
        })

    return render(request, "index.html", {"books": books})
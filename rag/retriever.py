import numpy as np
from .models import Document

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query_embedding, top_k=3):
    documents = Document.objects.all()
    scored = []

    for doc in documents:
        if not doc.embedding:
            continue

        score = cosine_similarity(query_embedding, doc.embedding)
        scored.append((score, doc.chunk_text))

    scored.sort(reverse=True)

    return [item[1] for item in scored[:top_k]]
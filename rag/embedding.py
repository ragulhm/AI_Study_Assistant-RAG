from google import genai
import os

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values
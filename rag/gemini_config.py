from google import genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

chat_model = genai.GenerativeModel("gemini-2.5-pro")
embedding_model = "models/embedding-001"
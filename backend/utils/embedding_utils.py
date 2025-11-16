from backend.config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
)

from config import LLM_PROVIDER, EMBEDDING_MODEL

def get_embeddings_model():
    """
    Returns the appropriate embeddings model based on config.
    """
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # Uses the default GOOGLE_API_KEY from environment
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

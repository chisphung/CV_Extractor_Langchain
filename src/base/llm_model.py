from langchain_google_genai import ChatGoogleGenerativeAI
from get_token import get_token

def get_llm(model: str = "gemini-2.0-flash", temperature: float = 0.0, max_tokens: int = None, **kwargs):
    """
    Initialize Gemini LLM via Google Generative AI using LangChain.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=2,
        api_key=get_token(),  # <-- pass directly here
        **kwargs
    )
    return llm

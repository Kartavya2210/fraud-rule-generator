"""LLM back-ends: one local (Ollama) and two cloud-hosted (Gemini, Groq).

Every back-end is put in JSON mode. Ollama additionally receives the full
rule JSON schema, which it enforces with grammar-constrained decoding.
"""
from langchain_core.language_models import BaseChatModel

from app import config

PROVIDERS = ("ollama", "gemini", "groq")


def get_llm(provider: str | None = None, model: str | None = None,
            json_schema: dict | None = None) -> tuple[BaseChatModel, str, str]:
    """Return (chat_model, provider, model_name)."""
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = model or config.OLLAMA_MODEL
        llm = ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0,
                         format=json_schema or "json")
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model or config.GEMINI_MODEL
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, response_mime_type="application/json")
    elif provider == "groq":
        from langchain_groq import ChatGroq
        model = model or config.GROQ_MODEL
        llm = ChatGroq(model=model, temperature=0,
                       model_kwargs={"response_format": {"type": "json_object"}})
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")

    return llm, provider, model

"""Provider factory so the whole app can switch between OpenAI and Gemini via one env var."""
from __future__ import annotations

from .config import get_settings


def get_chat_model(temperature: float = 0.0):
    s = get_settings()
    if s.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.chat_model, temperature=temperature)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=s.chat_model, temperature=temperature)


def get_embeddings():
    s = get_settings()
    if s.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedding_model)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=s.embedding_model)

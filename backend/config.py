import os
from functools import lru_cache

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
SQLITE_PATH = os.path.join(BASE_DIR, "chat_history.db")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-3-5-sonnet-latest")

GOOGLE_TEXT_MODEL = os.getenv("GOOGLE_TEXT_MODEL", "gemini-2.5-flash")
GOOGLE_VISION_MODEL = os.getenv("GOOGLE_VISION_MODEL", "gemini-2.5-flash")

GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set. Add it to {ENV_PATH}.")
    return value


def provider_supports_local_vision_uploads(provider: str | None = None) -> bool:
    selected = provider or get_provider()
    return selected in {"openai", "anthropic", "google", "groq"}


@lru_cache(maxsize=8)
def _get_cached_llm(provider: str, vision: bool, temperature: float):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model_name = OPENAI_VISION_MODEL if vision else OPENAI_TEXT_MODEL
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=_require_env_var("OPENAI_API_KEY"),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = ANTHROPIC_VISION_MODEL if vision else ANTHROPIC_TEXT_MODEL
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=_require_env_var("ANTHROPIC_API_KEY"),
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = GOOGLE_VISION_MODEL if vision else GOOGLE_TEXT_MODEL
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=_require_env_var("GEMINI_API_KEY"),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_VISION_MODEL if vision else GROQ_TEXT_MODEL,
            temperature=temperature,
            api_key=_require_env_var("GROQ_API_KEY"),
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. Choose from 'openai', 'anthropic', 'google', or 'groq'."
    )


def get_llm(vision: bool = False, temperature: float = 0.0):
    """
    Instantiate the configured LangChain chat model.
    """
    return _get_cached_llm(get_provider(), vision, temperature)

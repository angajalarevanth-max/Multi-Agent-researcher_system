"""
Application configuration from environment variables and ``.env``.

- ``Settings`` - Azure OpenAI credentials, app name, log level, optional LangSmith toggles.
- ``RetrieverPathsSettings`` - corpus directory and web search flags (loadable without Azure vars).
- ``get_research_sources_dir`` - resolved path to local research files (assessment default or ``RESEARCH_SOURCES_DIR``).
- ``get_llm`` / ``_build_azure_chat_llm`` - cached ``AzureChatOpenAI`` clients.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_openai import AzureChatOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    AZURE_OPENAI_ENDPOINT: str = Field(min_length=8, description="Azure OpenAI resource URL")
    AZURE_OPENAI_API_KEY: str = Field(min_length=8, description="Azure OpenAI API key")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(
        min_length=1, description="Chat deployment name"
    )
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-01")
    AZURE_OPENAI_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)

    APP_NAME: str = Field(default="multi-agent-researcher")
    LOG_LEVEL: str = Field(default="INFO")

    # LangSmith (optional); synced to os.environ via utils.langsmith_setup
    LANGSMITH_TRACING: bool = Field(
        default=False,
        description="Send LangGraph / LLM traces to LangSmith when API key is set.",
    )
    LANGSMITH_API_KEY: str = Field(
        default="",
        description="LangSmith API key (also sets LANGCHAIN_API_KEY for LangChain).",
    )
    LANGSMITH_PROJECT: str = Field(
        default="multi-agent-researcher",
        description="LangSmith / LangChain project name for traces.",
    )
    LANGSMITH_ENDPOINT: str = Field(
        default="",
        description="Optional LangSmith API base URL (empty = cloud default).",
    )


class RetrieverPathsSettings(BaseSettings):
    """Loads only retriever paths from ``.env`` so the corpus resolves without Azure credentials."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    RESEARCH_SOURCES_DIR: str = Field(
        default="",
        description="Path to folder with .txt/.pdf/.html/.csv sources; empty = default assessment pack.",
    )
    WEB_SEARCH_ENABLED: bool = Field(
        default=True,
        description="Augment dataset retrieval with DuckDuckGo web results (second source channel).",
    )
    WEB_SEARCH_MAX_RESULTS: int = Field(default=5, ge=1, le=15)


@lru_cache
def get_settings() -> Settings:
    """Singleton ``Settings`` instance (cached)."""
    return Settings()


@lru_cache
def get_retriever_paths() -> RetrieverPathsSettings:
    """Singleton retriever path settings (cached)."""
    return RetrieverPathsSettings()


@lru_cache(maxsize=8)
def _build_azure_chat_llm(temperature: float) -> AzureChatOpenAI:
    """Cached LLM instances per resolved temperature (settings read at call time)."""
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.AZURE_OPENAI_ENDPOINT.rstrip("/"),
        api_key=s.AZURE_OPENAI_API_KEY,
        azure_deployment=s.AZURE_OPENAI_CHAT_DEPLOYMENT,
        openai_api_version=s.AZURE_OPENAI_API_VERSION,
        temperature=temperature,
    )


def get_research_sources_dir() -> Path:
    """Directory containing research corpus files for the retriever agent."""
    s = get_retriever_paths()
    raw = (s.RESEARCH_SOURCES_DIR or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent.parent / p).resolve()
        return p
    return (
        Path(__file__).resolve().parent.parent
        / "AI_Solutions_Engineer_Tech_Assessment_V3"
        / "materials"
        / "research_pack"
        / "sources"
    )


def get_llm(*, temperature: Optional[float] = None) -> AzureChatOpenAI:
    """
    Azure OpenAI chat client for agents. Credentials and deployment come from ``.env`` via ``Settings``.

    Pass ``temperature`` to override ``AZURE_OPENAI_TEMPERATURE`` from the environment.
    """
    s = get_settings()
    t = s.AZURE_OPENAI_TEMPERATURE if temperature is None else temperature
    return _build_azure_chat_llm(t)

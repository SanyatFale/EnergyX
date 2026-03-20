"""Configuration management for EnergyX."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider: "ollama" or "cerebras"
    llm_provider: str = Field(default="cerebras")

    # Ollama Configuration (local)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")

    # Cerebras Configuration (cloud, OpenAI-compatible)
    cerebras_api_key: str = Field(default="")
    cerebras_model: str = Field(default="llama-3.3-70b")
    cerebras_base_url: str = Field(default="https://api.cerebras.ai/v1")

    # Temperature settings (two-LLM pattern)
    routing_temperature: float = Field(default=0.1)
    synthesis_temperature: float = Field(default=0.7)

    # System Configuration
    max_workers: int = Field(default=4)
    device: Literal["cuda", "cpu"] = Field(default="cuda")
    random_seed: int = Field(default=42)

    # Logging
    log_level: str = Field(default="INFO")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    output_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "outputs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)


# Global settings instance
settings = Settings()


def get_llm(temperature: float = 0.1):
    """Get LLM instance based on configured provider.

    Supports Cerebras (cloud, OpenAI-compatible) and Ollama (local).
    two-temperature LLM pattern.
    """
    if settings.llm_provider == "cerebras":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=settings.cerebras_base_url,
            api_key=settings.cerebras_api_key,
            model=settings.cerebras_model,
            temperature=temperature,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=temperature,
        )

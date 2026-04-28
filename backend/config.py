"""
Application settings — single pydantic-settings class for all configuration.

Reads from (in priority order):
1. init kwargs (for tests)
2. OS environment variables
3. HashiCorp Vault secrets (if VAULT_ADDR + VAULT_TOKEN are set)
4. .env file
5. Field defaults

Per INSTRUCTIONS.md: "Single pydantic-settings class. Typed/validated env vars
at startup. No os.getenv or magic strings."
"""

import logging
import os
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)

# Vault secret path and key-to-field mapping
# Only secrets that are stored in Vault are listed here.
# When Vault is reachable, these values override the ones from .env.
_VAULT_KEY_MAP: dict[str, str] = {
    "JW-SECRET-KEY": "jwt_secret_key",
    "AZURE-OPENAI-KEY": "azure_openai_key",
}


class HashiCorpVaultSource(PydanticBaseSettingsSource):
    """
    Custom settings source that reads secrets from HashiCorp Vault.

    Uses the KV v2 secrets engine at mount point "secret".
    The path "travel-planner" holds all our application secrets.

    If Vault is unreachable or authentication fails, we silently
    fall back to env/.env values — the app never fails to start
    because of a vault issue.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._vault_values: dict[str, Any] | None = None

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        return None, field_name, False

    def _fetch_from_vault(self) -> dict[str, Any]:
        # Return cached values if we've already fetched from vault
        if self._vault_values is not None:
            return self._vault_values

        vault_addr = os.getenv("VAULT_ADDR", "").strip() or None
        vault_token = os.getenv("VAULT_TOKEN", "").strip() or None

        if not vault_addr or not vault_token:
            logger.debug("VAULT_ADDR or VAULT_TOKEN not configured — skipping vault lookup")
            self._vault_values = {}
            return self._vault_values

        try:
            import hvac

            client = hvac.Client(url=vault_addr, token=vault_token)

            if not client.is_authenticated():
                logger.warning("Vault authentication failed — falling back to env/.env")
                self._vault_values = {}
                return self._vault_values

            # Read secrets from KV v2 engine at secret/data/travel-planner
            response = client.secrets.kv.v2.read_secret_version(
                path="travel-planner",
                mount_point="secret",
            )
            secret_data = response.get("data", {}).get("data", {})

            # Map vault keys to settings field names
            values: dict[str, Any] = {}
            for vault_key, field_name in _VAULT_KEY_MAP.items():
                value = secret_data.get(vault_key)
                if value is not None:
                    values[field_name] = value
                    logger.info("Fetched %s from HashiCorp Vault", field_name)

            self._vault_values = values
        except Exception:
            logger.warning(
                "HashiCorp Vault unreachable or secret not found — "
                "falling back to env/.env values"
            )
            self._vault_values = {}

        return self._vault_values

    def __call__(self) -> dict[str, Any]:
        return self._fetch_from_vault()


class Settings(BaseSettings):
    """
    All application configuration in one place.

    Every field is typed and validated at startup.
    Secrets (jwt_secret_key, azure_openai_key) can come from Vault
    or fall back to .env values.
    """

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Core & Server ──────────────────────────────────────
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    allowed_origins: str = Field(default="http://localhost:5173", alias="ALLOWED_ORIGINS")

    # ── Database ─────────────────────────────────────────────
    database_url: str = Field(alias="DATABASE_URL")

    # ── Auth ─────────────────────────────────────────────────
    jwt_secret_key: str = Field(alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=60, alias="JWT_EXPIRATION_MINUTES")

    # ── Azure OpenAI LLMs & Embeddings ──────────────────────
    # Azure OpenAI is our LLM and embedding provider (v1 API).
    # - Strong model: used for final agent synthesis (e.g. Kimi-K2.6-1)
    # - Cheap model:  used for mechanical tasks (arg extraction, query rewrite)
    #                 (e.g. DeepSeek-V3.2-1)
    # - Embedding:    text-embedding-3-small outputs 1536-dim vectors
    #                 (matches vector(1536) column in documents table)
    # - endpoint:     Azure OpenAI resource endpoint with /openai/v1/ suffix
    # - key:          API key for authentication — stored in Vault when possible
    #
    # Why the v1 API (ChatOpenAI) instead of AzureChatOpenAI?
    # - Azure's v1 API allows using ChatOpenAI directly with base_url + api_key
    # - No need for api_version or azure_deployment parameters
    # - Simpler, fewer config fields, same functionality
    # - LangChain docs recommend this approach for v1 API endpoints
    #
    # Two Models, One Agent (per INSTRUCTIONS.md):
    # - Cheap model for mechanical tasks: argument extraction, RAG query rewrite
    # - Strong model for final synthesis: combining tool outputs into a plan
    # - This saves cost — most work is cheap, only the final answer uses the
    #   expensive model
    azure_openai_key: str = Field(alias="AZURE_OPENAI_KEY")
    azure_openai_endpoint: str = Field(
        default="https://hadymahdy44-0734-week4-resource.services.ai.azure.com/openai/v1",
        alias="AZURE_OPENAI_ENDPOINT",
    )
    azure_strong_model: str = Field(
        default="Kimi-K2.6-1", alias="AZURE_STRONG_MODEL"
    )
    azure_cheap_model: str = Field(
        default="DeepSeek-V3.2-1", alias="AZURE_CHEAP_MODEL"
    )
    azure_embedding_model: str = Field(
        default="text-embedding-3-small-1", alias="AZURE_EMBEDDING_MODEL"
    )

    # ── LangSmith Tracing ────────────────────────────────────
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")

    # ── External APIs (agent tools) ──────────────────────────
    weather_api_key: str | None = Field(default=None, alias="WEATHER_API_KEY")
    flight_api_key: str | None = Field(default=None, alias="FLIGHT_API_KEY")
    fx_api_key: str | None = Field(default=None, alias="FX_API_KEY")

    # ── Webhook Delivery ─────────────────────────────────────
    webhook_url: str | None = Field(default=None, alias="WEBHOOK_URL")
    webhook_timeout_sec: int = Field(default=10, alias="WEBHOOK_TIMEOUT_SEC")
    webhook_max_retries: int = Field(default=3, alias="WEBHOOK_MAX_RETRIES")

    # ── RAG & Caching ────────────────────────────────────────
    rag_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")
    api_cache_ttl_sec: int = Field(default=600, alias="API_CACHE_TTL_SEC")

    documents_dir: str = Field(default="../documents", alias="DOCUMENTS_DIR")

    # ── HashiCorp Vault ──────────────────────────────────────
    vault_addr: str | None = Field(default=None, alias="VAULT_ADDR")
    vault_token: str | None = Field(default=None, alias="VAULT_TOKEN")

    # ── Other integrations ───────────────────────────────────
    discord_bot_token: str | None = Field(default=None, alias="DISCORD_BOT_TOKEN")
    resend_api_key: str | None = Field(default=None, alias="RESEND_API_KEY")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority order (highest first):
        # 1. init kwargs (tests)
        # 2. OS env vars
        # 3. HashiCorp Vault
        # 4. .env file
        # 5. file secrets (k8s/docker secrets)
        return (
            init_settings,
            env_settings,
            HashiCorpVaultSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — created once, reused everywhere."""
    return Settings()

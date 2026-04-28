#!/bin/sh
# vault-init.sh — Seeds application secrets into HashiCorp Vault on first boot.
#
# This script runs INSIDE the Vault container (see docker-compose.yml entrypoint).
# Flow: Vault dev server starts → this script waits for it to be ready →
#       writes secrets from env vars into the KV v2 secrets engine.
#
# How the secrets get here:
#   1. .env file defines JWT_SECRET_KEY and AZURE_OPENAI_KEY
#   2. docker-compose.yml passes them as environment vars into the Vault container
#   3. This script reads those env vars and writes them to Vault's KV store
#   4. The FastAPI backend (config.py → HashiCorpVaultSource) reads them back
#      from Vault at startup, overriding the .env values
#
# Why "dev" mode?
#   - Vault dev mode stores everything in memory (no persistent storage needed)
#   - All secrets are wiped when the container restarts — this script re-seeds them
#   - Perfect for development; production would use Raft storage + unseal keys

# Stop on first error — don't silently continue if Vault is unreachable
set -e

# Vault client configuration — the `vault` CLI reads these env vars
# to know where Vault is and how to authenticate.
# ${VAR:-default} syntax: use VAR if set, otherwise use the default value.
export VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
export VAULT_TOKEN="${VAULT_TOKEN:-myroot}"

# Poll Vault until its HTTP API is ready.
# In dev mode, Vault starts unsealed — we just need to wait for the
# HTTP listener to accept connections. The `until` loop retries every
# second until `vault status` returns exit code 0.
# >/dev/null 2>&1 silences both stdout and stderr — we only need the exit code.
echo "Waiting for Vault..."
until vault status >/dev/null 2>&1; do
  sleep 1
done
echo "Vault is ready. Seeding secrets..."

# Write secrets to Vault's KV v2 engine at path "secret/travel-planner".
#
# vault kv put secret/travel-planner KEY=VALUE
#   └─ mount point: "secret"  (KV v2 default mount in dev mode)
#   └─ path:        "travel-planner"
#   └─ data:        key-value pairs written as secret data
#
# The values come from environment variables injected by docker-compose:
#   JWT_SECRET_KEY   → from .env (used by auth service for JWT signing)
#   AZURE_OPENAI_KEY → from .env (used by llm.py for Azure API authentication)
#
# config.py's _VAULT_KEY_MAP maps these Vault keys back to Settings fields:
#   "JW-SECRET-KEY"    → settings.jwt_secret_key
#   "AZURE-OPENAI-KEY" → settings.azure_openai_key
vault kv put secret/travel-planner \
  JW-SECRET-KEY="${JWT_SECRET_KEY}" \
  AZURE-OPENAI-KEY="${AZURE_OPENAI_KEY}"

echo "Vault secrets seeded."

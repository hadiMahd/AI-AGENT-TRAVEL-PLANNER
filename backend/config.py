# Settings class
import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient
VAULT_URL = os.environ["AZURE_KEYVAULT_RESOURCEENDPOINT"]
credential = DefaultAzureCredential()

secret_client = SecretClient(vault_url=VAULT_URL, credential=credential)
secret = secret_client.get_secret("secret-name")

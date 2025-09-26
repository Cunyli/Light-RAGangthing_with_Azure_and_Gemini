#!/usr/bin/env python3
"""
Debug Azure OpenAI URL construction
"""

import os
from dotenv import load_dotenv

def debug_azure_urls():
    """Debug URL construction for Azure OpenAI"""
    
    load_dotenv()
    
    print("🔍 Debugging Azure OpenAI URL Construction")
    print("=" * 60)
    
    # Get current environment variables
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    llm_deployment = os.getenv('AZURE_OPENAI_LLM_DEPLOYMENT')
    embed_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
    
    print(f"API Key: {'✅' if api_key else '❌'}")
    print(f"Endpoint: {endpoint}")
    print(f"LLM Deployment: {llm_deployment}")
    print(f"Embed Deployment: {embed_deployment}")
    print(f"API Version: {api_version}")
    
    # Construct URLs as they would be in our functions
    if endpoint and llm_deployment:
        llm_base_url = f"{endpoint}/openai/deployments/{llm_deployment}"
        print(f"\n🔗 LLM Base URL: {llm_base_url}")
        print(f"   Full LLM URL would be: {llm_base_url}/chat/completions?api-version={api_version}")
    
    if endpoint and embed_deployment:
        embed_base_url = f"{endpoint}/openai/deployments/{embed_deployment}"
        print(f"\n🔗 Embed Base URL: {embed_base_url}")
        print(f"   Full Embed URL would be: {embed_base_url}/embeddings?api-version={api_version}")
    
    # Show expected format for comparison
    print(f"\n📋 Expected Azure OpenAI URL format:")
    print(f"   https://<resource-name>.openai.azure.com/openai/deployments/<deployment-name>/chat/completions?api-version=<api-version>")
    print(f"   https://<resource-name>.openai.azure.com/openai/deployments/<deployment-name>/embeddings?api-version=<api-version>")

if __name__ == "__main__":
    debug_azure_urls()
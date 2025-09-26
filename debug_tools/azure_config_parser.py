#!/usr/bin/env python3
"""
Fixed Azure OpenAI configuration for LightRAG compatibility

This version properly converts Azure OpenAI URLs to the format expected by LightRAG.
"""

import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

def parse_azure_openai_config():
    """Parse Azure OpenAI configuration and convert to LightRAG-compatible format"""
    
    # Load environment variables
    load_dotenv()
    
    # Get Azure OpenAI URLs from environment
    llm_full_url = os.getenv('LLM_BINDING_HOST')
    embedding_full_url = os.getenv('EMBEDDING_BINDING_HOST')
    
    def parse_azure_url(full_url):
        """Parse Azure OpenAI URL and extract base_url components"""
        if not full_url:
            return None, None
        
        print(f"   Parsing URL: {full_url}")
        
        # Example URL: https://lijie-mazglg3v-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview
        parsed = urlparse(full_url)
        
        # Extract base URL (everything up to and including /openai/)
        base_url = f"{parsed.scheme}://{parsed.netloc}/openai/"
        
        # Extract deployment name from path
        path_parts = parsed.path.strip('/').split('/')
        print(f"   Path parts: {path_parts}")
        
        deployment_name = None
        if 'deployments' in path_parts:
            deployment_idx = path_parts.index('deployments') + 1
            if deployment_idx < len(path_parts):
                deployment_name = path_parts[deployment_idx]
        
        print(f"   Extracted base_url: {base_url}")
        print(f"   Extracted deployment: {deployment_name}")
        
        return base_url, deployment_name
    
    # Parse LLM configuration
    llm_base_url, llm_deployment = parse_azure_url(llm_full_url)
    llm_api_key = os.getenv('LLM_BINDING_API_KEY')
    
    # Parse Embedding configuration  
    embed_base_url, embed_deployment = parse_azure_url(embedding_full_url)
    embed_api_key = os.getenv('EMBEDDING_BINDING_API_KEY')
    
    config = {
        # LLM Configuration for LightRAG
        'llm_base_url': llm_base_url,
        'llm_deployment': llm_deployment,
        'llm_api_key': llm_api_key,
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        
        # Embedding Configuration for LightRAG
        'embed_base_url': embed_base_url,
        'embed_deployment': embed_deployment, 
        'embed_api_key': embed_api_key,
        'embed_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'embed_dim': int(os.getenv('EMBEDDING_DIM', '1536')),
    }
    
    print("🔧 Parsed Azure OpenAI Configuration:")
    print(f"   LLM Base URL: {config['llm_base_url']}")
    print(f"   LLM Deployment: {config['llm_deployment']}")
    print(f"   LLM Model: {config['llm_model']}")
    print(f"   Embed Base URL: {config['embed_base_url']}")
    print(f"   Embed Deployment: {config['embed_deployment']}")
    print(f"   Embed Model: {config['embed_model']}")
    print()
    
    return config

if __name__ == "__main__":
    config = parse_azure_openai_config()
    print("✅ Configuration parsed successfully!")
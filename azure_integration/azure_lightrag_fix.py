#!/usr/bin/env python3
"""
Azure OpenAI compatible wrapper for LightRAG

This module provides Azure OpenAI compatible LLM and embedding functions
that work with LightRAG by handling Azure-specific URL and deployment requirements.
"""

import os
import base64
from openai import AsyncAzureOpenAI
import numpy as np
from typing import Any
from dotenv import load_dotenv


async def azure_openai_complete_if_cache(
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs
) -> str:
    """Azure OpenAI LLM completion function for RAGAnything"""
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
    api_version = os.getenv("AZURE_LLM_API_VERSION", "2024-08-01-preview")
    
    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing required Azure OpenAI configuration")
    
    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for msg in history_messages:
        messages.append(msg)
    
    if prompt:
        messages.append({"role": "user", "content": prompt})
    
    # Filter out LightRAG-specific parameters that should not be passed to the API
    lightrag_params = {
        'azure_endpoint', 'api_version', 'azure_deployment', 'hashing_kv', 
        'keyword_extraction', 'enable_cot'
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in lightrag_params}
    
    # Call Azure OpenAI Chat Completions API
    response = await client.chat.completions.create(
        model=deployment,
        messages=messages,
        **filtered_kwargs
    )
    
    return response.choices[0].message.content


async def azure_openai_embed(
    texts: list[str], 
    **kwargs
) -> np.ndarray:
    """Azure OpenAI compatible embedding function for LightRAG"""
    
    load_dotenv()
    
    # Get Azure OpenAI configuration
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  # e.g., https://your-resource.openai.azure.com
    deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')  # e.g., text-embedding-3-small
    # Use embedding-specific API version
    api_version = os.getenv('AZURE_EMBEDDING_API_VERSION', '2023-05-15')
    
    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing required Azure OpenAI embedding configuration")
    
    # Filter out LightRAG-specific parameters that should not be passed to the API
    lightrag_params = {
        'azure_endpoint', 'api_version', 'azure_deployment', 'hashing_kv', 
        'keyword_extraction', 'enable_cot'
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in lightrag_params}
    
    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )
    
    try:
        # Handle single text or list of texts
        if isinstance(texts, str):
            text_input = [texts]
        else:
            text_input = texts
        
        response = await client.embeddings.create(
            model=deployment,
            input=text_input,
            **filtered_kwargs
        )
        
        return np.array([
            np.array(dp.embedding, dtype=np.float32)
            if isinstance(dp.embedding, list)
            else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
            for dp in response.data
        ])
    finally:
        await client.close()

# Add the embedding_dim attribute required by LightRAG
azure_openai_embed.embedding_dim = 1536

# Add function object attributes for compatibility
azure_openai_complete_if_cache.func = azure_openai_complete_if_cache
azure_openai_embed.func = azure_openai_embed


def get_azure_openai_functions():
    """Get Azure OpenAI compatible functions for RAGAnything"""
    return azure_openai_complete_if_cache, azure_openai_embed
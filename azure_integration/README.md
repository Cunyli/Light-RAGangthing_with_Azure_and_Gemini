# Azure Integration for RAGAnything

This directory contains the production-ready Azure OpenAI integration files for RAGAnything.

## Files

### `azure_lightrag_fix.py`
Custom Azure OpenAI functions that provide full compatibility with Azure's deployment-based URL structure and parameter filtering for LightRAG integration.

**Key Features:**
- Compatible with Azure OpenAI API endpoints
- Proper parameter filtering for LightRAG compatibility
- Support for both embedding and LLM functions
- API version separation (embedding: 2023-05-15, LLM: 2024-08-01-preview)
- Built-in retry logic and error handling

### `fixed_azure_custom_example.py`
Production example showing how to use RAGAnything with Azure OpenAI services.

**Usage:**
```bash
# From the project root directory
uv run python azure_integration/fixed_azure_custom_example.py <path_to_document>

# Example:
uv run python azure_integration/fixed_azure_custom_example.py samples/sample.pdf
```

## Configuration

Make sure your `.env` file (in the project root) contains the following Azure OpenAI configuration:

```env
# Azure OpenAI Embedding Configuration
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15

# Azure OpenAI LLM Configuration  
AZURE_OPENAI_LLM_API_KEY=your_llm_api_key
AZURE_OPENAI_LLM_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_LLM_API_VERSION=2024-08-01-preview

# Performance Settings
MAX_ASYNC=1  # Keep low to avoid Azure OpenAI rate limits
```

## Performance Notes

- `MAX_ASYNC=1` is recommended to avoid Azure OpenAI rate limiting
- Processing time depends on document size and Azure OpenAI quota
- Increase MAX_ASYNC only if you have sufficient Azure OpenAI quota

## Integration

To use these functions in your own code:

```python
from azure_integration.azure_lightrag_fix import get_azure_openai_functions

# Get Azure OpenAI compatible functions
llm_func, embedding_func = get_azure_openai_functions()

# Use with RAGAnything
config = RAGAnythingConfig(
    llm_model_func=llm_func,
    embedding_func=embedding_func,
    # ... other config
)
```
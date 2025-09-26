# Azure OpenAI Integration for RAG-Anything

This document describes the Azure OpenAI integration for RAG-Anything after reorganization.

## Directory Structure

### 🚀 Production Files: `azure_integration/`
Contains production-ready Azure OpenAI integration files:
- `azure_lightrag_fix.py` - Core Azure OpenAI compatibility functions
- `fixed_azure_custom_example.py` - Production example script
- `README.md` - Detailed usage documentation

### 🔧 Debug Tools: `debug_tools/`
Contains debugging and testing utilities used during development:
- Various test scripts for individual components
- Debug storage directories
- Legacy integration attempts
- `README.md` - Debug tools documentation

## Quick Start

### 1. Configuration
Ensure your `.env` file contains Azure OpenAI configuration:
```bash
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

# Performance Settings (Important!)
MAX_ASYNC=1  # Keep low to avoid Azure rate limits
```

### 2. Usage
```bash
# Process a document with Azure OpenAI
uv run python azure_integration/fixed_azure_custom_example.py path/to/document.pdf

# Example with sample file
uv run python azure_integration/fixed_azure_custom_example.py samples/sample.pdf
```

### 3. Integration in Your Code
```python
from azure_integration.azure_lightrag_fix import get_azure_openai_functions
from raganything import RAGAnything, RAGAnythingConfig

# Get Azure-compatible functions
llm_func, embedding_func = get_azure_openai_functions()

# Configure RAGAnything
config = RAGAnythingConfig(
    llm_model_func=llm_func,
    embedding_func=embedding_func,
    working_directory="./storage",
    # ... other configuration
)

# Initialize and use
rag = RAGAnything(config)
await rag.process_document_complete("document.pdf")
```

## Key Features

✅ **Full Azure OpenAI Compatibility**
- Deployment-based URL structure support
- API version separation for embedding/LLM
- Proper parameter filtering for LightRAG

✅ **Rate Limiting Handling**
- Built-in retry logic for 429 errors
- Configurable concurrency limits
- Automatic backoff strategies

✅ **Production Ready**
- Error handling and logging
- Clean separation of concerns
- Comprehensive documentation

## Troubleshooting

If you encounter issues:
1. Check your Azure OpenAI quota and rate limits
2. Verify `.env` configuration
3. Use debug tools in `debug_tools/` directory
4. Set `MAX_ASYNC=1` to avoid rate limiting
5. See individual README files for detailed guidance

## Migration Notes

If you were using previous Azure integration files, they have been moved:
- Production files → `azure_integration/`
- Debug/test files → `debug_tools/`
- All import paths updated automatically
- Functionality remains the same
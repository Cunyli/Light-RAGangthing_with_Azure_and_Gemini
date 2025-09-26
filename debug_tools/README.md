# Debug Tools for RAG-Anything Azure Integration

This directory contains various debugging and testing tools used during the development and troubleshooting of Azure OpenAI integration.

## Files Overview

### Configuration and URL Tools
- `azure_config_parser.py` - Parses Azure OpenAI URLs and extracts configuration parameters
- `debug_azure_urls.py` - URL parsing and validation utilities

### Testing Scripts
- `test_azure_functions.py` - Independent test for Azure OpenAI LLM functions
- `test_embedding.py` - Independent test for Azure OpenAI embedding functions
- `debug_azure_integration.py` - Comprehensive step-by-step debug script that tests all components

### Example Scripts (Legacy)
- `azure_openai_example.py` - Early Azure OpenAI integration attempt
- `debug_azure_openai.py` - Debug version with additional logging
- `fixed_azure_example.py` - Intermediate fix attempt

### Storage Directories
- `debug_storage/` - Debug storage used by test scripts
- `debug_storage_text/` - Text-specific debug storage

## Usage

### Quick Debug Test
Run the comprehensive debug script to test all components:
```bash
cd debug_tools
uv run python debug_azure_integration.py
```

### Individual Component Tests
Test embedding function:
```bash
cd debug_tools
uv run python test_embedding.py
```

Test LLM function:
```bash
cd debug_tools
uv run python test_azure_functions.py
```

### URL Configuration Parser
Parse and validate Azure OpenAI URLs:
```bash
cd debug_tools
uv run python azure_config_parser.py
```

## Debug Process History

These tools were used to solve the following issues:
1. **404 Errors**: Incorrect URL formatting and API versions
2. **Parameter Compatibility**: LightRAG passing unsupported parameters to Azure OpenAI
3. **Client Type Issues**: Using AsyncOpenAI instead of AsyncAzureOpenAI
4. **Rate Limiting**: 429 errors due to high concurrency
5. **Historical Data Conflicts**: Storage format incompatibilities

## Notes

- These tools require the same `.env` configuration as the production Azure integration
- Storage directories may be created during testing and can be safely deleted
- Some scripts may be outdated but are kept for reference during troubleshooting
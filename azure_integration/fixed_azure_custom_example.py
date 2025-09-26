#!/usr/bin/env python3
"""
RAGAnything Fixed Azure OpenAI Example with Custom Functions

This example shows how to use RAGAnything with Azure OpenAI
using custom LLM and embedding functions that are fully compatible
with Azure's deployment-based URL structure.
"""

import asyncio
import logging
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom Azure OpenAI functions
from azure_integration.azure_lightrag_fix import get_azure_openai_functions

# Import RAGAnything components
from raganything import RAGAnything, RAGAnythingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_with_custom_azure_openai(file_path: str):
    """Process a document using RAGAnything with custom Azure OpenAI functions"""
    
    try:
        print("🔧 RAGAnything Fixed Azure OpenAI with Custom Functions")
        print("=" * 65)
        print("Processing document with custom Azure OpenAI functions")
        print("=" * 65)
        
        # Get custom Azure OpenAI functions
        azure_llm_func, azure_embed_func = get_azure_openai_functions()
        
        logger.info("✅ Azure OpenAI custom functions loaded successfully")
        logger.info("🔧 Using custom Azure OpenAI functions:")
        logger.info("   LLM Function: azure_openai_complete_if_cache")
        logger.info("   Embedding Function: azure_openai_embed")
        
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir="./rag_storage_custom",
            parser='mineru',
            parse_method='auto',
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True
        )
        
        # Initialize RAGAnything with custom Azure OpenAI functions
        rag = RAGAnything(
            config=config,
            llm_model_func=azure_llm_func,  # Use custom Azure LLM function
            embedding_func=azure_embed_func  # Use custom Azure embedding function
        )
        
        logger.info(f"🚀 Processing document: {file_path}")
        
        # Process the document
        result = await rag.process_document_complete(file_path)
        
        logger.info("✅ Document processing completed successfully!")
        
        # Test queries
        logger.info("\n🔍 Testing queries:")
        
        queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?"
        ]
        
        for query in queries:
            logger.info(f"\n📝 Query: {query}")
            try:
                result = await rag.aquery(query, mode="hybrid")
                print(f"✅ Answer:\n{result}\n")
                logger.info("✅ Query executed successfully!")
            except Exception as e:
                logger.error(f"❌ Query failed: {e}")
                continue
        
        logger.info("✅ All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            await rag.finalize_storages()
            logger.info("✅ RAGAnything finalized successfully")
        except Exception as cleanup_error:
            logger.warning(f"Warning: Failed to finalize RAGAnything: {cleanup_error}")

async def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python fixed_azure_custom_example.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    await process_with_custom_azure_openai(file_path)

if __name__ == "__main__":
    asyncio.run(main())
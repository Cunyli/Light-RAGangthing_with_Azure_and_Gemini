#!/usr/bin/env python3
"""
Debug Azure RAGAnything Integration - Step by step testing
"""

import asyncio
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_azure_functions_basic():
    """Test Azure functions in isolation"""
    logger.info("🔧 Step 1: Testing Azure functions in isolation")
    
    try:
        # Add parent directory to path for import
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from azure_integration.azure_lightrag_fix import azure_openai_complete_if_cache, azure_openai_embed
        logger.info("✅ Azure functions imported successfully")
        
        # Test embedding
        logger.info("📊 Testing embedding function...")
        embedding_result = await azure_openai_embed(["Hello world", "Test embedding"])
        logger.info(f"✅ Embedding successful! Shape: {embedding_result.shape}")
        
        # Test LLM
        logger.info("🤖 Testing LLM function...")
        llm_result = await azure_openai_complete_if_cache(
            "What is 2+2?",
            system_prompt="You are a helpful assistant."
        )
        logger.info(f"✅ LLM successful! Response: {llm_result[:50]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ Azure functions test failed: {e}")
        return False

async def test_raganything_config():
    """Test RAGAnything configuration without processing"""
    logger.info("🔧 Step 2: Testing RAGAnything configuration")
    
    try:
        from raganything import RAGAnything, RAGAnythingConfig
        from azure_integration.azure_lightrag_fix import get_azure_openai_functions
        
        llm_func, embed_func = get_azure_openai_functions()
        
        config = RAGAnythingConfig(
            working_dir="./debug_storage",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=1
        )
        
        logger.info("✅ RAGAnything config created successfully")
        
        # Initialize with functions passed separately
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            embedding_func=embed_func
        )
        logger.info("✅ RAGAnything instance created successfully")
        
        # Clean up
        await rag.finalize_storages()
        logger.info("✅ RAGAnything finalized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAGAnything config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_text_processing():
    """Test processing simple text instead of PDF"""
    logger.info("🔧 Step 3: Testing simple text processing")
    
    try:
        from raganything import RAGAnything, RAGAnythingConfig
        from azure_integration.azure_lightrag_fix import get_azure_openai_functions
        import tempfile
        import os
        
        llm_func, embed_func = get_azure_openai_functions()
        
        config = RAGAnythingConfig(
            working_dir="./debug_storage_text",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
            max_concurrent_files=1
        )
        
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            embedding_func=embed_func
        )
        logger.info("✅ RAGAnything initialized for text processing")
        
        # Create a simple text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a simple test document. It contains basic information for testing.")
            temp_file = f.name
        
        logger.info(f"📄 Created test file: {temp_file}")
        
        # Try to process the simple text file
        logger.info("🚀 Starting text file processing...")
        result = await rag.process_document_complete(temp_file)
        logger.info("✅ Text processing completed!")
        
        # Test a simple query
        logger.info("🔍 Testing simple query...")
        query_result = await rag.aquery("What is this document about?")
        logger.info(f"✅ Query successful: {query_result[:100]}...")
        
        # Clean up
        os.unlink(temp_file)
        await rag.finalize_storages()
        logger.info("✅ Text processing test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main debugging function"""
    logger.info("🔍 Starting Azure RAGAnything Integration Debug")
    logger.info("=" * 60)
    
    # Step 1: Test Azure functions
    step1_success = await test_azure_functions_basic()
    if not step1_success:
        logger.error("❌ Step 1 failed - Azure functions not working")
        sys.exit(1)
    
    # Step 2: Test RAGAnything config
    step2_success = await test_raganything_config()
    if not step2_success:
        logger.error("❌ Step 2 failed - RAGAnything config issues")
        sys.exit(1)
    
    # Step 3: Test simple text processing
    step3_success = await test_simple_text_processing()
    if not step3_success:
        logger.error("❌ Step 3 failed - Text processing issues")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🎉 All debug steps passed successfully!")
    logger.info("The issue might be specific to PDF processing or the specific file.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Debug interrupted by user")
    except Exception as e:
        logger.error(f"❌ Debug failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
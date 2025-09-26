#!/usr/bin/env python3
"""
Fixed RAGAnything example for Azure OpenAI

This version properly supports Azure OpenAI configuration from environment variables.
"""

import asyncio
import argparse
import logging
import os
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig

# Import LightRAG functions
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_azure_openai_config():
    """Get Azure OpenAI configuration from environment variables"""
    config = {
        # LLM Configuration
        'llm_api_key': os.getenv('LLM_BINDING_API_KEY'),
        'llm_base_url': os.getenv('LLM_BINDING_HOST'),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        
        # Embedding Configuration
        'embed_api_key': os.getenv('EMBEDDING_BINDING_API_KEY'),
        'embed_base_url': os.getenv('EMBEDDING_BINDING_HOST'),
        'embed_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'embed_dim': int(os.getenv('EMBEDDING_DIM', '1536')),
    }
    
    # Validate configuration
    missing = []
    for key, value in config.items():
        if not value and 'dim' not in key:
            missing.append(key)
    
    if missing:
        raise ValueError(f"Missing Azure OpenAI configuration: {missing}")
    
    return config

async def process_with_azure_openai(file_path: str, output_dir: str = "./output"):
    """Process document using Azure OpenAI configuration from environment variables"""
    
    try:
        # Get Azure OpenAI configuration
        azure_config = get_azure_openai_config()
        logger.info("✅ Azure OpenAI configuration loaded successfully")
        
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir="./rag_storage",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
        # Define LLM model function for Azure OpenAI
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                azure_config['llm_model'],
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=azure_config['llm_api_key'],
                base_url=azure_config['llm_base_url'],
                **kwargs,
            )
        
        # Define vision model function for Azure OpenAI
        def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",  # Use vision model
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=azure_config['llm_api_key'],
                    base_url=azure_config['llm_base_url'],
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                },
                            ],
                        },
                    ],
                    api_key=azure_config['llm_api_key'],
                    base_url=azure_config['llm_base_url'],
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
        
        # Define embedding function for Azure OpenAI
        embedding_func = EmbeddingFunc(
            embedding_dim=azure_config['embed_dim'],
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=azure_config['embed_model'],
                api_key=azure_config['embed_api_key'],
                base_url=azure_config['embed_base_url'],
            ),
        )
        
        # Initialize RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        
        logger.info(f"🚀 Processing document: {file_path}")
        
        # Process document
        await rag.process_document_complete(
            file_path=file_path, 
            output_dir=output_dir, 
            parse_method="auto"
        )
        
        logger.info("✅ Document processing completed successfully!")
        
        # Test queries
        logger.info("\n🔍 Testing queries:")
        
        queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]
        
        for query in queries:
            logger.info(f"\n📝 Query: {query}")
            try:
                result = await rag.aquery(query, mode="hybrid")
                logger.info(f"✅ Answer: {result}")
            except Exception as e:
                logger.error(f"❌ Query failed: {str(e)}")
        
        # Finalize
        await rag.finalize_storages()
        logger.info("✅ All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="RAGAnything example with Azure OpenAI support")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--output", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file_path):
        logger.error(f"❌ File not found: {args.file_path}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run processing
    asyncio.run(process_with_azure_openai(args.file_path, args.output))

if __name__ == "__main__":
    print("🔧 RAGAnything Azure OpenAI Example")
    print("=" * 40)
    print("Processing document with Azure OpenAI")
    print("=" * 40)
    main()
#!/usr/bin/env python3
"""
Final Azure OpenAI example with LightRAG-compatible configuration

This version properly formats Azure OpenAI URLs for LightRAG compatibility.
"""

import asyncio
import argparse
import logging
import os
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Import our Azure configuration parser
from azure_config_parser import parse_azure_openai_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def process_with_fixed_azure_openai(file_path: str, output_dir: str = "./output"):
    """Process document using LightRAG-compatible Azure OpenAI configuration"""
    
    try:
        # Parse Azure OpenAI configuration
        azure_config = parse_azure_openai_config()
        logger.info("✅ Azure OpenAI configuration parsed successfully")
        
        # Create RAGAnything configuration with new working directory
        config = RAGAnythingConfig(
            working_dir="./rag_storage_new",  # Use new directory to avoid version issues
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
        # Define LLM model function using LightRAG-compatible format
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                azure_config['llm_deployment'],  # Use deployment name as model
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=azure_config['llm_api_key'],
                base_url=azure_config['llm_base_url'],
                **kwargs,
            )
        
        # Define vision model function (if needed)
        def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
            # For vision, we might need to use a different deployment
            model_name = "gpt-4o"  # Vision model deployment name
            
            if messages:
                return openai_complete_if_cache(
                    model_name,
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
                    model_name,
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
        
        # Define embedding function using LightRAG-compatible format
        embedding_func = EmbeddingFunc(
            embedding_dim=azure_config['embed_dim'],
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=azure_config['embed_deployment'],  # Use deployment name as model
                api_key=azure_config['embed_api_key'],
                base_url=azure_config['embed_base_url'],
            ),
        )
        
        logger.info("🔧 Using LightRAG-compatible Azure OpenAI configuration:")
        logger.info(f"   LLM: {azure_config['llm_deployment']} @ {azure_config['llm_base_url']}")
        logger.info(f"   Embedding: {azure_config['embed_deployment']} @ {azure_config['embed_base_url']}")
        
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
                import traceback
                traceback.print_exc()
        
        # Finalize
        await rag.finalize_storages()
        logger.info("✅ All operations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description="RAGAnything with fixed Azure OpenAI support")
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
    asyncio.run(process_with_fixed_azure_openai(args.file_path, args.output))

if __name__ == "__main__":
    print("🔧 RAGAnything Fixed Azure OpenAI Example")
    print("=" * 45)
    print("Processing document with corrected Azure OpenAI")
    print("=" * 45)
    main()
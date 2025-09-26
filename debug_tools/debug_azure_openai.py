#!/usr/bin/env python3
"""
Debug Azure OpenAI configuration - 模拟RAGAnything的实际使用方式
"""

import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import aiohttp

# Load environment variables
load_dotenv()

async def test_direct_url_request():
    """直接使用完整URL测试（模拟RAGAnything的方式）"""
    
    print("=== 测试1: 直接使用完整URL (模拟RAGAnything方式) ===")
    
    # 使用RAGAnything的配置方式
    full_embedding_url = os.getenv("EMBEDDING_BINDING_HOST")
    api_key = os.getenv("EMBEDDING_BINDING_API_KEY")
    
    print(f"完整URL: {full_embedding_url}")
    print(f"API Key: {'*' * 60 + api_key[-4:] if api_key else 'None'}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "input": ["This is a test sentence for embedding."],
        "model": "text-embedding-3-small"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print("发送直接HTTP请求...")
            async with session.post(full_embedding_url, json=payload, headers=headers) as response:
                print(f"状态码: {response.status}")
                print(f"响应头: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ 直接URL请求成功!")
                    print(f"   嵌入维度: {len(result['data'][0]['embedding'])}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 直接URL请求失败!")
                    print(f"   错误内容: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

async def test_azure_client_method():
    """使用Azure OpenAI客户端测试（我们之前成功的方式）"""
    
    print("\n=== 测试2: Azure OpenAI客户端方式 ===")
    
    api_key = os.getenv("EMBEDDING_BINDING_API_KEY")
    binding_host = os.getenv("EMBEDDING_BINDING_HOST")
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")
    
    # 从完整URL提取endpoint
    endpoint = "/".join(binding_host.split("/")[:3])
    
    try:
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        print("使用Azure OpenAI客户端发送请求...")
        response = await client.embeddings.create(
            input="This is a test sentence for embedding.",
            model=deployment
        )
        
        print("✅ Azure客户端请求成功!")
        print(f"   嵌入维度: {len(response.data[0].embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Azure客户端请求失败: {str(e)}")
        return False

async def test_llm_request():
    """测试LLM请求"""
    
    print("\n=== 测试3: LLM请求测试 ===")
    
    api_key = os.getenv("LLM_BINDING_API_KEY")
    full_llm_url = os.getenv("LLM_BINDING_HOST")
    
    print(f"LLM URL: {full_llm_url}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "messages": [{"role": "user", "content": "Say 'Hello World'"}],
        "model": "gpt-4o-mini",
        "max_tokens": 50
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print("发送LLM请求...")
            async with session.post(full_llm_url, json=payload, headers=headers) as response:
                print(f"状态码: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ LLM请求成功!")
                    print(f"   响应: {result['choices'][0]['message']['content']}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ LLM请求失败!")
                    print(f"   错误内容: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ LLM请求异常: {str(e)}")
        return False

async def main():
    """运行所有测试"""
    print("开始调试Azure OpenAI配置...")
    print("=" * 50)
    
    # 测试1: 直接URL方式（RAGAnything使用的方式）
    test1_result = await test_direct_url_request()
    
    # 测试2: Azure客户端方式（我们之前成功的方式）
    test2_result = await test_azure_client_method()
    
    # 测试3: LLM请求
    test3_result = await test_llm_request()
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"   直接URL嵌入请求: {'✅ 成功' if test1_result else '❌ 失败'}")
    print(f"   Azure客户端嵌入: {'✅ 成功' if test2_result else '❌ 失败'}")
    print(f"   LLM请求: {'✅ 成功' if test3_result else '❌ 失败'}")
    
    if test1_result and test2_result and test3_result:
        print("\n🎉 所有测试都成功！配置应该没问题。")
    else:
        print("\n⚠️  有测试失败，需要进一步调查。")

if __name__ == "__main__":
    asyncio.run(main())
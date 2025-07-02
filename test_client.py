#!/usr/bin/env python3
"""
SGLang 客户端测试脚本
用于测试 Qwen3-14b 模型推理服务
"""

import requests
import json
import argparse
import time

def test_completions_api(base_url, prompt, max_tokens=100):
    """测试 completions API"""
    url = f"{base_url}/v1/completions"
    
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"发送请求到: {url}")
        print(f"提示词: {prompt}")
        print("-" * 50)
        
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"]
            
            print(f"生成结果:")
            print(generated_text)
            print("-" * 50)
            print(f"耗时: {end_time - start_time:.2f} 秒")
            print(f"状态: 成功")
            return True
        else:
            print(f"请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return False

def test_chat_completions_api(base_url, message, max_tokens=100):
    """测试 chat completions API"""
    url = f"{base_url}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"发送聊天请求到: {url}")
        print(f"用户消息: {message}")
        print("-" * 50)
        
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            print(f"AI回复:")
            print(generated_text)
            print("-" * 50)
            print(f"耗时: {end_time - start_time:.2f} 秒")
            print(f"状态: 成功")
            return True
        else:
            print(f"请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return False

def check_server_health(base_url):
    """检查服务器健康状态"""
    try:
        url = f"{base_url}/health"
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="测试 SGLang 服务器")
    parser.add_argument("--host", default="localhost", help="服务器主机地址")
    parser.add_argument("--port", default=30000, type=int, help="服务器端口")
    parser.add_argument("--prompt", default="你好，请介绍一下自己。", help="测试提示词")
    parser.add_argument("--max-tokens", default=100, type=int, help="最大生成token数")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print("SGLang 服务器测试")
    print("=" * 60)
    
    # 检查服务器健康状态
    print("1. 检查服务器健康状态...")
    if check_server_health(base_url):
        print("✓ 服务器运行正常")
    else:
        print("✗ 服务器无法访问，请确保服务器已启动")
        return
    
    print("\n2. 测试 completions API...")
    test_completions_api(base_url, args.prompt, args.max_tokens)
    
    print("\n3. 测试 chat completions API...")
    test_chat_completions_api(base_url, args.prompt, args.max_tokens)
    
    print("\n" + "=" * 60)
    print("测试完成")

if __name__ == "__main__":
    main() 
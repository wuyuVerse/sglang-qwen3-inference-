#!/usr/bin/env python3
"""
OpenAI SDK 接口测试 - 专门测试如何避免思考过程输出
基于 SGLang 官方文档的示例 - 类型安全版本
"""

from openai import OpenAI
import json

# 设置 OpenAI 客户端连接到 SGLang 服务器
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def safe_get_content(response) -> str:
    """安全获取响应内容"""
    content = response.choices[0].message.content
    return content if content is not None else ""

def test_enable_thinking_false():
    """测试 enable_thinking=False 参数"""
    print("🔴 测试 1: enable_thinking=False")
    print("-" * 50)
    
    try:
        response = client.chat.completions.create(
            model="default",  # 使用默认模型
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
                {"role": "user", "content": "什么是人工智能？请简洁回答。"}
            ],
            max_tokens=200,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},  # 关键参数
            },
        )
        
        content = safe_get_content(response)
        print(f"问题: 什么是人工智能？")
        print(f"回答: {content}")
        print(f"✅ 成功 - 输出长度: {len(content)} 字符")
        
        # 检查是否包含思考过程标记
        if "<think>" in content or "</think>" in content:
            print("⚠️  警告: 输出仍包含思考标记")
        else:
            print("✅ 无思考标记")
            
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    print()

def test_enable_thinking_true():
    """测试 enable_thinking=True 参数（对比）"""
    print("🟡 测试 2: enable_thinking=True (对比测试)")
    print("-" * 50)
    
    try:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "什么是人工智能？请简洁回答。"}
            ],
            max_tokens=400,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},  # 启用思考
            },
        )
        
        content = safe_get_content(response)
        print(f"问题: 什么是人工智能？")
        print(f"回答: {content[:200]}..." if len(content) > 200 else f"回答: {content}")
        print(f"💭 输出长度: {len(content)} 字符")
        
        # 检查是否包含思考过程标记
        if "<think>" in content or "</think>" in content:
            print("💭 包含思考标记（预期行为）")
        else:
            print("❓ 未包含思考标记")
            
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    print()

def test_different_models():
    """测试不同的模型名称"""
    print("🔵 测试 3: 不同模型名称测试")
    print("-" * 50)
    
    model_names = ["default", "Qwen/Qwen3-4B"]
    
    for model_name in model_names:
        print(f"📝 测试模型: {model_name}")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
                    {"role": "user", "content": "谁发明了电话？"}
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            content = safe_get_content(response)
            print(f"   回答: {content}")
            print(f"   ✅ 成功")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
        print()

def test_multiple_questions():
    """批量测试多个问题"""
    print("🟣 测试 4: 批量问题测试 (enable_thinking=False)")
    print("-" * 50)
    
    questions = [
        "什么是深度学习？",
        "区块链技术的优势是什么？",
        "机器学习和人工智能的区别？",
        "Python 的主要特点？"
    ]
    
    for i, question in enumerate(questions, 1):
        try:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
                    {"role": "user", "content": question}
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            content = safe_get_content(response)
            print(f"4.{i} {question}")
            print(f"    答: {content[:100]}..." if len(content) > 100 else f"    答: {content}")
            
            # 检查思考标记
            has_thinking = any(marker in content for marker in ["<think>", "</think>", "思考:", "用户问", "ASSISTANT:"])
            print(f"    {'⚠️  包含思考标记' if has_thinking else '✅ 干净输出'}")
            
        except Exception as e:
            print(f"4.{i} ❌ 失败: {e}")
        print()

def test_parameter_optimization():
    """测试参数优化组合"""
    print("🟠 测试 5: 参数优化组合测试")
    print("-" * 50)
    
    # 测试不同的参数组合
    test_configs = [
        {
            "name": "配置A: 低创造性",
            "params": {
                "temperature": 0.3,
                "top_p": 0.7,
                "presence_penalty": 2.0,
            }
        },
        {
            "name": "配置B: 推荐参数",
            "params": {
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
            }
        },
        {
            "name": "配置C: 高创造性",
            "params": {
                "temperature": 0.9,
                "top_p": 0.9,
                "presence_penalty": 1.0,
            }
        }
    ]
    
    question = "什么是自然语言处理？"
    
    for config in test_configs:
        print(f"📊 {config['name']}")
        try:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
                    {"role": "user", "content": question}
                ],
                max_tokens=200,
                **config['params'],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            content = safe_get_content(response)
            print(f"   回答: {content[:80]}..." if len(content) > 80 else f"   回答: {content}")
            print(f"   长度: {len(content)} 字符")
            
            # 检查思考标记
            has_thinking = any(marker in content for marker in ["<think>", "</think>", "思考:", "用户问", "ASSISTANT:"])
            print(f"   {'⚠️  包含思考相关内容' if has_thinking else '✅ 干净输出'}")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
        print()

def test_comparison_with_without_enable_thinking():
    """对比测试：带和不带 enable_thinking 参数"""
    print("🔄 测试 6: enable_thinking 参数对比测试")
    print("-" * 50)
    
    question = "什么是机器学习？"
    system_prompt = "你是一个有用的AI助手。"
    
    # 测试不带 enable_thinking 参数
    print("🔹 不使用 enable_thinking 参数:")
    try:
        response1 = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7,
            top_p=0.8,
        )
        
        content1 = safe_get_content(response1)
        print(f"   回答: {content1[:150]}..." if len(content1) > 150 else f"   回答: {content1}")
        print(f"   长度: {len(content1)} 字符")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    print()
    
    # 测试带 enable_thinking=False 参数
    print("🔹 使用 enable_thinking=False:")
    try:
        response2 = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        
        content2 = safe_get_content(response2)
        print(f"   回答: {content2[:150]}..." if len(content2) > 150 else f"   回答: {content2}")
        print(f"   长度: {len(content2)} 字符")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    print()

def main():
    print("=" * 80)
    print("🚀 OpenAI SDK 接口测试 - 避免思考过程输出")
    print("基于 SGLang 官方文档示例 (类型安全版本)")
    print("=" * 80)
    
    # 执行所有测试
    test_enable_thinking_false()
    test_enable_thinking_true()
    test_different_models()
    test_multiple_questions()
    test_parameter_optimization()
    test_comparison_with_without_enable_thinking()
    
    print("=" * 80)
    print("🎉 测试完成")
    print("💡 关键发现:")
    print("   1. enable_thinking=False 是控制思考过程的关键参数")
    print("   2. 配合中文系统提示词效果更好") 
    print("   3. 推荐参数: temperature=0.7, top_p=0.8, presence_penalty=1.5")
    print("   4. max_tokens 建议设置为 150-400 对于简短回答")
    print("   5. extra_body 中的 chat_template_kwargs 是 SGLang 特有功能")
    print("   6. OpenAI SDK 兼容性良好，可直接替换 API 端点")
    print("=" * 80)

if __name__ == "__main__":
    main() 
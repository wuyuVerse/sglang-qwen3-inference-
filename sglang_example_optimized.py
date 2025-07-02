#!/usr/bin/env python3
"""
SGLang 统一测试示例 - 最终优化版本
整合 SGLang Python API 和 OpenAI SDK 的最佳实践
"""

import sglang as sgl
import requests
import json
from openai import OpenAI

# OpenAI SDK 客户端配置
openai_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:30000/v1",
)

def safe_get_content(response) -> str:
    """安全获取响应内容"""
    content = response.choices[0].message.content
    return content if content is not None else ""

@sgl.function
def simple_qa(s, question):
    """简单问答示例 - 中文系统提示词 + 优化参数"""
    s += sgl.system("你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。")  # type: ignore
    s += sgl.user(question)  # type: ignore
    s += sgl.assistant(sgl.gen("answer", temperature=0.7, top_p=0.8, presence_penalty=1.5, stop=["<think>", "</think>", "思考:", "用户:", "ASSISTANT:"]))

def openai_chat_clean(messages, max_tokens=200):
    """OpenAI SDK 清洁调用 - 使用 enable_thinking: False"""
    try:
        response = openai_client.chat.completions.create(
            model="default",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},  # 关键参数
            },
        )
        return safe_get_content(response)
    except Exception as e:
        return f"Exception: {e}"

def chat_api_clean(messages, max_tokens=200):
    """传统 requests 调用 - 使用 enable_thinking: False"""
    url = "http://localhost:30000/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False  # 关键参数：禁用思考过程
            }
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content if content is not None else ""
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Exception: {e}"

def openai_structured_generation(topic):
    """OpenAI SDK 结构化生成 - 推荐方式"""
    try:
        response = openai_client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "你是一个有用的助手，请按照指定格式提供信息，不要显示思考过程。"},
                {"role": "user", "content": f"请简要介绍{topic}，包括定义和主要特点。请分别用'定义：'和'特点：'开头，每部分用一段话说明。"}
            ],
            max_tokens=300,
            temperature=0.5,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return safe_get_content(response)
    except Exception as e:
        return f"Exception: {e}"

@sgl.function
def structured_generation(s, topic):
    """结构化生成示例 - SGLang API（备选方案）"""
    s += sgl.system("你是一个有用的助手，请按照指定格式提供信息，不要显示思考过程。")  # type: ignore
    s += sgl.user(f"请简要介绍{topic}，包括定义和主要特点。请分别用'定义：'和'特点：'开头。")  # type: ignore
    s += sgl.assistant("定义：")  # type: ignore
    s += sgl.assistant(sgl.gen("definition", temperature=0.5, top_p=0.8, presence_penalty=1.5, stop=["<think>", "</think>", "\n特点：", "\n\n"]))
    s += sgl.assistant("\n特点：")  # type: ignore
    s += sgl.assistant(sgl.gen("features", temperature=0.5, top_p=0.8, presence_penalty=1.5, stop=["<think>", "</think>", "\n定义：", "\n\n"]))

@sgl.function
def creative_writing(s, prompt):
    """创作示例 - 中文系统提示词版"""
    s += sgl.system("你是一个富有创意的作家，请直接提供创作内容，不要显示思考过程。")  # type: ignore
    s += sgl.user(prompt)  # type: ignore
    s += sgl.assistant(sgl.gen("content", temperature=0.8, top_p=0.9, presence_penalty=1.5, stop=["<think>", "</think>", "---", "END"]))

@sgl.function
def code_generation(s, task):
    """代码生成示例 - 中文系统提示词版"""
    s += sgl.system("你是一个编程专家，请直接提供代码，不要显示思考过程和解释。")  # type: ignore
    s += sgl.user(f"请编写代码：{task}")  # type: ignore
    s += sgl.assistant("```python\n")  # type: ignore
    s += sgl.assistant(sgl.gen("code", temperature=0.3, top_p=0.8, presence_penalty=1.5, stop=["```", "<think>", "</think>", "END"]))
    s += sgl.assistant("\n```")  # type: ignore

def main():
    # 设置后端地址
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    
    print("=" * 80)
    print("🚀 SGLang 统一测试示例（最终优化版）")
    print("整合 SGLang Python API 和 OpenAI SDK 的最佳实践")
    print("=" * 80)
    
    # 示例1: SGLang Python API 问答
    print("\n1. 🎯 SGLang Python API 问答")
    print("-" * 50)
    try:
        state = simple_qa.run(question="什么是人工智能？")
        print(f"问题: 什么是人工智能？")
        print(f"回答: {state['answer'].strip()}")
        print("✅ SGLang API 成功 - 输出干净")
    except Exception as e:
        print(f"❌ SGLang API 失败: {e}")
    
    # 示例2: OpenAI SDK 测试（推荐方式）
    print("\n2. 🔥 OpenAI SDK 测试（推荐 - enable_thinking: False）")
    print("-" * 50)
    try:
        result = openai_chat_clean([
            {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
            {"role": "user", "content": "什么是量子计算？请简洁回答。"}
        ])
        print(f"问题: 什么是量子计算？")
        print(f"回答: {result.strip()}")
        print("✅ OpenAI SDK 成功")
    except Exception as e:
        print(f"❌ OpenAI SDK 失败: {e}")
    
    # 示例3: 传统 requests 调用对比
    print("\n3. 💬 传统 requests 调用对比")
    print("-" * 50)
    try:
        result = chat_api_clean([
            {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
            {"role": "user", "content": "什么是机器学习？请简洁回答。"}
        ])
        print(f"问题: 什么是机器学习？")
        print(f"回答: {result.strip()}")
        print("✅ requests 调用成功")
    except Exception as e:
        print(f"❌ requests 调用失败: {e}")
    
    # 示例4: 结构化生成（OpenAI SDK - 推荐）
    print("\n4. 📊 结构化生成测试（OpenAI SDK）")
    print("-" * 50)
    try:
        result = openai_structured_generation("深度学习")
        print(f"主题: 深度学习")
        print(f"结果: {result}")
        print("✅ OpenAI SDK 结构化生成成功")
    except Exception as e:
        print(f"❌ OpenAI SDK 结构化生成失败: {e}")
    
    # 示例4B: SGLang API 结构化生成（备选方案）
    print("\n4B. 📊 SGLang API 结构化生成（备选方案）")
    print("-" * 50)
    try:
        state = structured_generation.run(topic="机器学习")
        print(f"主题: 机器学习")
        print(f"定义: {state['definition'].strip()}")
        print(f"特点: {state['features'].strip()}")
        print("✅ SGLang API 结构化生成成功")
    except Exception as e:
        print(f"❌ SGLang API 结构化生成失败: {e}")
    
    # 示例5: 创意写作（SGLang API）
    print("\n5. ✍️ 创意写作测试")
    print("-" * 50)
    try:
        state = creative_writing.run(prompt="写一首关于春天的四行诗")
        print(f"要求: 写一首关于春天的四行诗")
        content = state['content'].strip()
        print(f"作品: {content}")
        print("✅ SGLang 创意写作成功")
    except Exception as e:
        print(f"❌ SGLang 创意写作失败: {e}")
    
    # 示例6: 代码生成（SGLang API）
    print("\n6. 💻 代码生成测试")
    print("-" * 50)
    try:
        state = code_generation.run(task="实现冒泡排序")
        print(f"任务: 实现冒泡排序")
        print(f"代码:\n```python\n{state['code'].strip()}\n```")
        print("✅ SGLang 代码生成成功")
    except Exception as e:
        print(f"❌ SGLang 代码生成失败: {e}")
    
    # 示例7: OpenAI SDK 批量测试
    print("\n7. 🔥 OpenAI SDK 批量测试（推荐方式）")
    print("-" * 50)
    
    # 多种问题类型的测试
    test_questions = [
        "什么是深度学习？",
        "谁发明了电话？", 
        "解释一下区块链技术"
    ]
    
    for i, q in enumerate(test_questions, 1):
        try:
            result = openai_chat_clean([
                {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
                {"role": "user", "content": q}
            ])
            print(f"7.{i} {q}")
            print(f"    答: {result[:80]}..." if len(result) > 80 else f"    答: {result}")
            
            # 检查是否包含思考过程标记
            has_thinking = any(marker in result for marker in ["<think>", "</think>", "思考:", "用户问", "ASSISTANT:"])
            print(f"    {'⚠️  包含思考标记' if has_thinking else '✅ 干净输出'}")
        except Exception as e:
            print(f"    ❌ 失败: {e}")
    
    # 示例8: 参数对比测试
    print("\n8. ⚖️ enable_thinking 参数对比测试")
    print("-" * 50)
    
    test_question = "什么是自然语言处理？"
    
    # 测试不带 enable_thinking=False 的情况
    print("🔸 不使用 enable_thinking=False:")
    try:
        # 创建一个不使用 enable_thinking=False 的临时函数
        payload_without_enable_thinking = {
            "model": "default",
            "messages": [
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": test_question}
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
        }
        
        response = requests.post("http://localhost:30000/v1/chat/completions", 
                               json=payload_without_enable_thinking, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            result1 = content if content is not None else ""
        else:
            result1 = f"Error: {response.status_code}"
            
        print(f"    回答: {result1[:100]}..." if len(result1) > 100 else f"    回答: {result1}")
        has_thinking1 = "<think>" in result1 or "</think>" in result1
        print(f"    {'包含思考过程' if has_thinking1 else '干净输出'}")
    except Exception as e:
        print(f"    ❌ 失败: {e}")
    
    print("\n🔸 使用 enable_thinking=False:")
    try:
        result2 = openai_chat_clean([
            {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答用户问题，不要显示思考过程。"},
            {"role": "user", "content": test_question}
        ], max_tokens=300)
        print(f"    回答: {result2[:100]}..." if len(result2) > 100 else f"    回答: {result2}")
        has_thinking2 = "<think>" in result2 or "</think>" in result2
        print(f"    {'包含思考过程' if has_thinking2 else '✅ 干净输出'}")
    except Exception as e:
        print(f"    ❌ 失败: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 统一测试完成")
    print("🏆 最佳实践总结：")
    print("   1. 🔥 推荐使用 OpenAI SDK + enable_thinking: False")
    print("   2. 💡 配合中文系统提示词：'不要显示思考过程'")
    print("   3. ⚙️  推荐参数: temperature=0.7, top_p=0.8, presence_penalty=1.5")
    print("   4. 📏 max_tokens 建议设置为 150-400 对于简短回答")
    print("   5. 🛠️  SGLang API 添加 stop 序列: ['<think>', '</think>', '思考:', '用户:', 'ASSISTANT:']")
    print("   6. 🎯 OpenAI SDK 在 extra_body 中使用 chat_template_kwargs")
    print("   7. ✨ OpenAI SDK 提供更好的类型安全性和易用性")
    print("=" * 80)

if __name__ == "__main__":
    main() 
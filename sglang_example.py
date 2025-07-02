#!/usr/bin/env python3
"""
SGLang Python API 使用示例
展示如何使用 SGLang 的前端语言进行复杂推理任务
"""

import sglang as sgl

@sgl.function
def multi_turn_chat(s, question):
    """多轮对话示例"""
    s += sgl.system("你是一个有用的AI助手，请用中文回答问题。")  # type: ignore
    s += sgl.user(question)  # type: ignore
    s += sgl.assistant(sgl.gen("answer", max_tokens=200))

@sgl.function
def chain_of_thought(s, problem):
    """思维链推理示例"""
    s += sgl.system("你是一个逻辑推理专家。请一步步分析问题。")  # type: ignore
    s += sgl.user(f"请分析以下问题：{problem}")  # type: ignore
    s += sgl.assistant("让我一步步分析这个问题：\n\n")  # type: ignore
    s += sgl.assistant(sgl.gen("thinking", max_tokens=300, stop="\n\n结论："))
    s += sgl.assistant("\n\n结论：")  # type: ignore
    s += sgl.assistant(sgl.gen("conclusion", max_tokens=100))

@sgl.function
def structured_output(s, topic):
    """结构化输出示例"""
    s += sgl.system("请按照指定格式输出信息。")  # type: ignore
    s += sgl.user(f"请介绍一下{topic}，包括定义、特点和应用。")  # type: ignore
    s += sgl.assistant("## 定义\n")  # type: ignore
    s += sgl.assistant(sgl.gen("definition", max_tokens=100, stop="\n\n"))
    s += sgl.assistant("\n\n## 特点\n")  # type: ignore
    s += sgl.assistant(sgl.gen("features", max_tokens=150, stop="\n\n"))
    s += sgl.assistant("\n\n## 应用\n")  # type: ignore
    s += sgl.assistant(sgl.gen("applications", max_tokens=150))

@sgl.function
def batch_generation(s, prompts):
    """批量生成示例"""
    results = []
    for i, prompt in enumerate(prompts):
        s += sgl.user(f"请回答：{prompt}")  # type: ignore
        s += sgl.assistant(sgl.gen(f"answer_{i}", max_tokens=100))
        results.append(s[f"answer_{i}"])
    return results

def main():
    # 设置后端地址
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    
    print("=" * 60)
    print("SGLang Python API 示例")
    print("=" * 60)
    
    # 示例1: 多轮对话
    print("\n1. 多轮对话示例:")
    print("-" * 30)
    try:
        state = multi_turn_chat.run(question="什么是人工智能？")
        print(f"问题: 什么是人工智能？")
        print(f"回答: {state['answer']}")
    except Exception as e:
        print(f"执行失败: {e}")
    
    # 示例2: 思维链推理
    print("\n2. 思维链推理示例:")
    print("-" * 30)
    try:
        problem = "如果一个篮子里有10个苹果，拿走3个，又放入5个，最后有多少个苹果？"
        state = chain_of_thought.run(problem=problem)
        print(f"问题: {problem}")
        print(f"分析过程: {state['thinking']}")
        print(f"结论: {state['conclusion']}")
    except Exception as e:
        print(f"执行失败: {e}")
    
    # 示例3: 结构化输出
    print("\n3. 结构化输出示例:")
    print("-" * 30)
    try:
        state = structured_output.run(topic="机器学习")
        print(f"主题: 机器学习")
        print(f"定义: {state['definition']}")
        print(f"特点: {state['features']}")
        print(f"应用: {state['applications']}")
    except Exception as e:
        print(f"执行失败: {e}")
    
    # 示例4: 批量生成
    print("\n4. 批量生成示例:")
    print("-" * 30)
    try:
        prompts = [
            "什么是深度学习？",
            "Python有什么优势？", 
            "如何学习编程？"
        ]
        state = batch_generation.run(prompts=prompts)
        for i, prompt in enumerate(prompts):
            print(f"问题{i+1}: {prompt}")
            print(f"回答{i+1}: {state[f'answer_{i}']}")
            print()
    except Exception as e:
        print(f"执行失败: {e}")
    
    print("=" * 60)
    print("示例完成")

if __name__ == "__main__":
    main() 
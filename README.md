# SGLang Qwen3 推理服务

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![SGLang](https://img.shields.io/badge/SGLang-0.4.0%2B-green.svg)](https://github.com/sgl-project/sglang)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

基于 [SGLang](https://github.com/sgl-project/sglang) 的 Qwen3 模型高性能推理服务

</div>

## 🚀 项目简介

这是一个基于 SGLang 框架的 Qwen3 模型推理服务项目，提供了完整的模型部署、量化优化和API服务解决方案。

### ✨ 主要特性

- **🔥 高性能推理**：基于 SGLang 的优化后端，支持 RadixAttention、零开销调度等加速技术
- **💾 智能量化**：支持多种量化方案（INT4/INT8/FP8），适配不同硬件配置
- **🌐 标准API**：兼容 OpenAI API 格式，无缝集成现有应用
- **⚙️ 灵活配置**：YAML配置文件 + 命令行参数，支持预设和自定义配置
- **📱 多种接口**：支持 HTTP API、Python SDK、SGLang 前端语言
- **🔧 易于部署**：完整的启动脚本和测试工具，开箱即用

### 🎯 支持的模型

- Qwen3-4B / Qwen3-14B
- 其他 SGLang 支持的模型（Llama、Gemma、Mistral 等）

## 📋 系统要求

### 硬件要求

| 模型 | 最小显存 | 推荐显存 | 推荐配置 |
|------|----------|----------|----------|
| Qwen3-4B | 6GB | 12GB | RTX 3080+ |
| Qwen3-14B | 12GB | 28GB | RTX 4090+ |

### 软件要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.8+
- **CUDA**: 11.8+ 或 12.0+
- **GPU**: NVIDIA GPU (支持 CUDA)

## 🛠️ 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/wuyuVerse/sglang-qwen3-inference-.git
cd sglang-qwen3-inference
```

### 2. 安装依赖

#### 方法 A: 使用 uv (推荐)

```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

#### 方法 B: 使用 pip

```bash
pip install -r requirements.txt
# 或者手动安装核心依赖
pip install sglang[all] torch transformers accelerate fastapi uvicorn
```

### 3. 启动服务

#### 一键启动 (推荐)

```bash
# 平衡配置 - 适合大多数场景
python launch_server.py --preset balanced

# 高性能配置 - 适合高端GPU
python launch_server.py --preset high_performance

# 内存优化配置 - 适合显存不足
python launch_server.py --preset memory_optimized
```

#### 自定义启动

```bash
# 使用 TorchAO INT4 量化
python launch_server.py --torchao-config int4wo-64

# 指定模型路径
python launch_server.py --model-path /path/to/qwen3-model --preset balanced

# 多GPU部署
python launch_server.py --preset balanced --tp-size 2
```

### 4. 测试服务

```bash
# 基础连接测试
python test_client.py

# 自定义测试
python test_client.py --prompt "介绍一下深度学习" --max-tokens 200
```

## 📚 详细使用指南

### 🎛️ 配置选择指南

#### 预设配置对比

| 预设 | 量化方法 | 内存节省 | 推理速度 | 适用场景 |
|------|----------|----------|----------|----------|
| `high_performance` | 无量化 | 0% | 100% | 高端GPU，追求最佳性能 |
| `balanced` | INT4 | ~50% | 85-95% | **推荐配置**，性能与内存平衡 |
| `memory_optimized` | INT4+优化 | ~60% | 75-85% | 显存不足，优先可用性 |
| `ultra_low_memory` | INT8 | ~75% | 60-75% | 低端GPU，最大兼容性 |

#### 硬件配置建议

```bash
# RTX 3060 (8GB)
python launch_server.py --preset ultra_low_memory --model-path /path/to/qwen3-4b

# RTX 3080/4080 (10-16GB)  
python launch_server.py --preset memory_optimized

# RTX 4090 (24GB)
python launch_server.py --preset balanced

# A100/H100 (40GB+)
python launch_server.py --preset high_performance
```

### 🔧 高级配置

#### 使用配置文件

```bash
# 使用默认配置文件
python launch_server.py --config server_config.yaml

# 使用示例配置文件
python launch_server.py --config config_examples.yaml

# 配置文件 + 命令行覆盖
python launch_server.py --config server_config.yaml --torchao-config int4wo-128
```

#### TorchAO 量化配置

```bash
# INT4 权重量化 (推荐)
python launch_server.py --torchao-config int4wo-64   # 更快
python launch_server.py --torchao-config int4wo-128  # 更准确

# INT8 量化
python launch_server.py --torchao-config int8wo

# FP8 量化 (需要 H100/H200)
python launch_server.py --torchao-config fp8dq-per_tensor
```

### 🌐 API 使用

#### HTTP API

**Completions API**

```bash
curl -X POST "http://localhost:30000/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "prompt": "介绍一下人工智能的发展历程",
       "max_tokens": 200,
       "temperature": 0.7
     }'
```

**Chat Completions API**

```bash
curl -X POST "http://localhost:30000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "messages": [
         {"role": "user", "content": "什么是机器学习？"}
       ],
       "max_tokens": 200,
       "temperature": 0.7
     }'
```

#### Python SDK

```python
import requests

# 简单对话
response = requests.post("http://localhost:30000/v1/chat/completions", 
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

#### SGLang 前端语言

```python
# 运行高级示例
python sglang_example.py

# 或者自定义使用
import sglang as sgl

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def simple_chat(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))

state = simple_chat.run(question="什么是深度学习？")
print(state["answer"])
```

## 📁 项目结构

```
sglang-qwen3-inference/
├── 📄 README.md                    # 项目文档
├── 📄 pyproject.toml               # 项目配置和依赖
├── 📄 uv.lock                      # 依赖锁定文件
├── 📄 .gitignore                   # Git忽略文件
├── 📄 .python-version              # Python版本指定
│
├── 🚀 launch_server.py             # 服务器启动脚本 (主入口)
├── ⚙️ server_config.yaml           # 默认配置文件  
├── ⚙️ config_examples.yaml         # 配置示例文件
│
├── 🧪 test_client.py               # HTTP API 测试客户端
├── 🧪 test_openai_sdk_clean.py     # OpenAI SDK 兼容测试
├── 🧪 sglang_example.py            # SGLang 前端语言示例
├── 🧪 sglang_example_optimized.py  # 优化版示例
│
├── 📱 main.py                      # 简单启动入口
└── 📂 .venv/                       # 虚拟环境 (uv创建)
```

## 🔍 示例和测试

### 基础功能测试

```bash
# 1. 启动服务器
python launch_server.py --preset balanced

# 2. 等待服务器启动完成 (看到 "Listening on http://0.0.0.0:30000")

# 3. 在新终端中测试
python test_client.py --prompt "介绍一下Python编程语言"
```

### 高级功能示例

```bash
# SGLang 前端语言示例 (支持多轮对话、结构化输出等)
python sglang_example.py

# OpenAI SDK 兼容性测试
python test_openai_sdk_clean.py

# 优化版示例 (批量处理、并发测试等)
python sglang_example_optimized.py
```

## ⚠️ 故障排除

### 常见问题

**Q: 启动时显示 "CUDA out of memory"**
```bash
# 解决方案1: 使用更激进的量化配置
python launch_server.py --preset ultra_low_memory

# 解决方案2: 减少内存分配
python launch_server.py --preset balanced --mem-fraction-static 0.7

# 解决方案3: 切换到更小的模型
python launch_server.py --model-path /path/to/qwen3-4b --preset balanced
```

**Q: 推理速度很慢**
```bash
# 解决方案1: 启用编译加速
python launch_server.py --preset balanced --enable-torch-compile

# 解决方案2: 使用更少量化
python launch_server.py --preset high_performance

# 解决方案3: 检查是否使用了GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: API 请求超时**
```bash
# 增加超时时间，检查服务器状态
curl http://localhost:30000/health

# 检查服务器日志
python launch_server.py --preset balanced  # 查看输出日志
```

### 性能优化建议

1. **选择合适的量化配置**：根据硬件选择预设配置
2. **启用编译加速**：添加 `--enable-torch-compile` 参数
3. **调整内存分配**：根据实际显存调整 `--mem-fraction-static`
4. **使用多GPU**：大模型可以使用 `--tp-size` 进行张量并行

## 🤝 贡献指南

我们欢迎各种形式的贡献！

1. **报告问题**：在 [Issues](https://github.com/wuyuVerse/sglang-qwen3-inference-/issues) 中报告 bug 或提出功能建议
2. **提交代码**：Fork 项目，创建分支，提交 Pull Request
3. **完善文档**：改进 README、添加示例或教程
4. **分享经验**：分享使用经验和最佳实践

### 开发环境搭建

```bash
# 克隆项目
git clone https://github.com/wuyuVerse/sglang-qwen3-inference-.git
cd sglang-qwen3-inference

# 安装开发依赖
uv sync --dev

# 运行测试
python -m pytest tests/  # (如果有测试的话)
```

## 📜 许可证

本项目采用 [Apache 2.0 License](LICENSE) 许可证。

## 🙏 致谢

- [SGLang](https://github.com/sgl-project/sglang) - 强大的 LLM 服务框架
- [Qwen](https://github.com/QwenLM/Qwen) - 优秀的开源语言模型
- [TorchAO](https://github.com/pytorch/torchao) - 高效的模型量化库

## 📞 联系方式

- **项目主页**: https://github.com/wuyuVerse/sglang-qwen3-inference-
- **问题反馈**: [GitHub Issues](https://github.com/wuyuVerse/sglang-qwen3-inference-/issues)
- **邮箱**: 1074275896@qq.com

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！⭐**

</div>

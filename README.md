# SGLang Qwen3-14b 推理项目

本项目使用 [SGLang](https://github.com/sgl-project/sglang) 框架部署和加速 Qwen3-14b 模型的推理服务。

## 项目简介

SGLang 是一个快速的大语言模型和视觉语言模型服务框架，通过协同设计后端运行时和前端语言，使模型交互更快速、更可控。

### 主要特性

- **快速后端运行时**: 提供高效的推理服务，支持 RadixAttention、零开销 CPU 调度、张量并行等优化
- **灵活前端语言**: 提供直观的编程接口，支持链式生成、高级提示、控制流等
- **广泛模型支持**: 支持 Qwen、Llama、Gemma、Mistral 等多种生成模型

## 环境要求

- Python >= 3.8
- CUDA 11.8+ 或 12.0+ (用于 GPU 加速)
- 足够的 GPU 内存 (Qwen3-14b 模型约需要 28GB+ 显存)

## 安装依赖

### 方法1: 使用 uv (推荐)

```bash
# 安装项目依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 方法2: 使用 pip

```bash
pip install sglang[all] torch transformers accelerate fastapi uvicorn requests
```

## 使用说明

### 1. 启动 SGLang 服务器

#### 快速启动 (使用量化预设)

```bash
# 平衡配置 (推荐) - INT4量化，适合大多数场景
python launch_server.py --preset balanced

# 高性能配置 - 无量化，适合高端GPU
python launch_server.py --preset high_performance

# 内存优化配置 - 激进量化，适合显存不足
python launch_server.py --preset memory_optimized

# 超低内存配置 - 最大压缩，适合低端GPU
python launch_server.py --preset ultra_low_memory
```

#### 自定义量化配置

```bash
# 使用TorchAO INT4量化 (推荐)
python launch_server.py --torchao-config int4wo-64

# 使用FP8量化 (需要H100/H200)
python launch_server.py --torchao-config fp8dq-per_tensor

# 使用传统量化方法
python launch_server.py --quantization fp8

# 多卡部署
python launch_server.py --preset balanced --tp-size 4
```

#### 使用配置文件

```bash
# 使用默认配置文件
python launch_server.py --config server_config.yaml

# 使用自定义配置文件
python launch_server.py --config config_examples.yaml

# 命令行参数会覆盖配置文件设置
python launch_server.py --config server_config.yaml --torchao-config int8wo
```

#### 启动参数说明

**基础配置:**
- `--model-path`: 模型路径 (默认: /data/local_disk0/wuyu/model/qwen/Qwen3-14b)
- `--host`: 服务器主机地址 (默认: 0.0.0.0)
- `--port`: 服务器端口 (默认: 30000)
- `--preset`: 量化预设 (balanced, high_performance, memory_optimized, ultra_low_memory)

**量化配置:**
- `--torchao-config`: TorchAO量化方法 (int4wo-64, int4wo-128, int8wo, fp8dq-per_tensor等)
- `--quantization`: 传统量化方法 (fp8, awq, gptq, bitsandbytes等)
- `--kv-cache-dtype`: KV缓存数据类型 (auto, fp8_e5m2, int8)

**性能优化:**
- `--tp-size`: 张量并行度 (默认: 1)
- `--mem-fraction-static`: 静态内存分配比例 (默认: 0.9)
- `--enable-torch-compile`: 启用torch编译加速
- `--enable-flashinfer`: 启用FlashInfer加速

### 2. 测试服务器

服务器启动后，使用测试脚本验证服务是否正常：

```bash
# 基本测试
python test_client.py

# 自定义测试
python test_client.py --host localhost --port 30000 --prompt "你好，介绍一下Python编程语言。"
```

### 3. 量化配置选择指南

#### 根据硬件选择配置

| GPU型号 | 显存大小 | 推荐配置 | 量化方法 | 预期性能 |
|---------|----------|----------|----------|----------|
| RTX 3060 | 8GB | `ultra_low_memory` | int8wo | 可运行，较慢 |
| RTX 3080 | 12GB | `memory_optimized` | int4wo-64 | 良好性能 |
| RTX 4090 | 24GB | `balanced` | int4wo-128 | 优秀性能 |
| A100 40GB | 40GB | `balanced` | int4wo-128 | 优秀性能 |
| A100 80GB | 80GB | `high_performance` | 无量化 | 最佳性能 |
| H100 80GB | 80GB | `fp8_hopper`* | fp8dq-per_tensor | 最佳性能+节省内存 |

*注：H100 配置需要在 `server_config.yaml` 中手动配置或使用 `config_examples.yaml`

#### 量化方法对比

| 量化方法 | 内存节省 | 推理速度 | 精度保持 | 硬件要求 |
|----------|----------|----------|----------|----------|
| 无量化 | 0% | 100% | 100% | 高显存 |
| int4wo-128 | ~50% | 85-95% | 95-98% | 通用 |
| int4wo-64 | ~50% | 80-90% | 90-95% | 通用 |
| int8wo | ~25% | 70-80% | 98-99% | 通用 |
| fp8dq-per_tensor | ~25% | 90-100% | 95-98% | H100/H200 |

#### 场景推荐

- **开发调试**: `high_performance` - 追求最高精度和稳定性  
- **生产部署**: `balanced` - 平衡性能、内存和精度  
- **资源受限**: `memory_optimized` - 优先考虑可部署性  
- **边缘计算**: `ultra_low_memory` - 最大化内存节省  

#### 实际使用示例

```bash
# 根据你的GPU选择对应配置
# RTX 4090 用户
python launch_server.py --preset balanced

# H100 用户 (使用FP8优化)
python launch_server.py --config config_examples.yaml --preset fp8_hopper_example

# 自定义微调
python launch_server.py --preset balanced --mem-fraction-static 0.8 --tp-size 2
```

### 4. 使用 SGLang Python API

运行高级功能示例：

```bash
python sglang_example.py
```

该示例包含：
- 多轮对话
- 思维链推理
- 结构化输出
- 批量生成

### 4. HTTP API 使用

#### Completions API

```bash
curl -X POST "http://localhost:30000/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "prompt": "你好，请介绍一下自己。",
       "max_tokens": 100,
       "temperature": 0.7
     }'
```

#### Chat Completions API

```bash
curl -X POST "http://localhost:30000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "messages": [
         {"role": "user", "content": "什么是人工智能？"}
       ],
       "max_tokens": 100,
       "temperature": 0.7
     }'
```

## 性能优化建议

### 1. 硬件配置

- **单卡部署**: 至少需要 A100 80GB 或类似级别的 GPU
- **多卡部署**: 可以使用张量并行 (`--tp` 参数) 分布到多张 GPU

### 2. 内存优化

```bash
# 调整静态内存分配比例
python launch_server.py --mem-fraction-static 0.8

# 启用内存优化选项
python launch_server.py --enable-flashinfer
```

### 3. 批处理优化

- 使用连续批处理提高吞吐量
- 适当调整 `max_tokens` 和 `temperature` 参数

## 项目结构

```
sglang-qwen3-inference/
├── launch_server.py      # 量化启动脚本 (支持YAML配置)
├── server_config.yaml    # 主配置文件 (包含量化预设)
├── config_examples.yaml  # 配置示例文件 (各种场景)
├── test_client.py        # 客户端测试脚本
├── sglang_example.py     # SGLang API 使用示例
├── pyproject.toml        # 项目依赖配置
├── README.md            # 项目说明文档
└── .venv/               # 虚拟环境 (uv 创建)
```

### 配置文件说明

- **server_config.yaml**: 主配置文件，包含所有支持的参数和4个内置量化预设
- **config_examples.yaml**: 详细的配置示例，包含6种不同硬件场景的优化配置
- **launch_server.py**: 增强的启动脚本，支持YAML配置文件和命令行参数

## 故障排除

### 常见问题

1. **显存不足**
   ```
   解决方案: 减少 mem-fraction-static 值或使用更多 GPU
   ```

2. **模型加载失败**
   ```
   检查模型路径是否正确，确保模型文件完整
   ```

3. **服务器无法访问**
   ```
   检查防火墙设置，确保端口未被占用
   ```

4. **推理速度慢**
   ```
   启用 FlashInfer 优化，调整批处理大小
   ```

## 相关链接

- [SGLang 官方文档](https://docs.sglang.ai/)
- [SGLang GitHub 仓库](https://github.com/sgl-project/sglang)
- [Qwen 模型文档](https://qwen.readthedocs.io/)

## 许可证

本项目遵循 Apache 2.0 许可证。

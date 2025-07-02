#!/usr/bin/env python3
"""
SGLang 服务器启动脚本 - 支持YAML配置和量化部署
用于部署 Qwen3-14b 模型推理服务，支持多种量化方式
"""

import argparse
import subprocess
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

class SGLangServerLauncher:
    def __init__(self):
        self.config_file = None
        self.config = {}
        self.default_config_path = "server_config.yaml"
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
            return {}
        except yaml.YAMLError as e:
            print(f"错误: 解析配置文件失败: {e}")
            sys.exit(1)
    
    def apply_quantization_preset(self, config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """应用量化预设配置"""
        if 'quantization_presets' not in config or preset_name not in config['quantization_presets']:
            print(f"警告: 量化预设 '{preset_name}' 不存在，使用默认配置")
            return config
            
        preset = config['quantization_presets'][preset_name]
        print(f"应用量化预设: {preset_name}")
        
        # 应用预设到相应的配置section
        for key, value in preset.items():
            if key == 'torchao_config':
                config.setdefault('quantization', {})['torchao_config'] = value
            elif key == 'dtype':
                config.setdefault('model', {})['dtype'] = value
            elif key == 'kv_cache_dtype':
                config.setdefault('quantization', {})['kv_cache_dtype'] = value
            elif key == 'mem_fraction_static':
                config.setdefault('memory', {})['mem_fraction_static'] = value
            elif key == 'chunked_prefill_size':
                config.setdefault('memory', {})['chunked_prefill_size'] = value
            elif key == 'max_running_requests':
                config.setdefault('memory', {})['max_running_requests'] = value
            elif key in ['enable_torch_compile', 'enable_flashinfer', 'disable_cuda_graph']:
                config.setdefault('optimization', {})[key] = value
                
        return config
    
    def build_command(self, args: argparse.Namespace, config: Dict[str, Any]) -> List[str]:
        """构建SGLang启动命令"""
        cmd = ["python", "-m", "sglang.launch_server"]
        
        # 模型配置
        model_config = config.get('model', {})
        model_path = args.model_path or model_config.get('model_path')
        if model_path:
            cmd.extend(["--model-path", model_path])
        
        if args.trust_remote_code or model_config.get('trust_remote_code'):
            cmd.append("--trust-remote-code")
            
        dtype = args.dtype or model_config.get('dtype')
        if dtype:
            cmd.extend(["--dtype", dtype])
        
        # 服务器配置
        server_config = config.get('server', {})
        host = args.host or server_config.get('host', "0.0.0.0")
        port = args.port or server_config.get('port', 30000)
        cmd.extend(["--host", host, "--port", str(port)])
        
        # 量化配置 (核心部分)
        quant_config = config.get('quantization', {})
        
        # TorchAO 量化配置 (优先级最高)
        torchao_config = args.torchao_config or quant_config.get('torchao_config')
        if torchao_config:
            cmd.extend(["--torchao-config", torchao_config])
            print(f"启用 TorchAO 量化: {torchao_config}")
        
        # 传统量化配置
        quantization_method = args.quantization or quant_config.get('method')
        if quantization_method and not torchao_config:
            cmd.extend(["--quantization", quantization_method])
            print(f"启用量化方法: {quantization_method}")
        
        # KV缓存量化
        kv_cache_dtype = args.kv_cache_dtype or quant_config.get('kv_cache_dtype')
        if kv_cache_dtype and kv_cache_dtype != "auto":
            cmd.extend(["--kv-cache-dtype", kv_cache_dtype])
        
        # 内存管理配置
        memory_config = config.get('memory', {})
        mem_fraction = args.mem_fraction_static or memory_config.get('mem_fraction_static')
        if mem_fraction:
            cmd.extend(["--mem-fraction-static", str(mem_fraction)])
            
        chunked_prefill = args.chunked_prefill_size or memory_config.get('chunked_prefill_size')
        if chunked_prefill:
            cmd.extend(["--chunked-prefill-size", str(chunked_prefill)])
            
        max_requests = args.max_running_requests or memory_config.get('max_running_requests')
        if max_requests:
            cmd.extend(["--max-running-requests", str(max_requests)])
            
        context_length = args.context_length or memory_config.get('context_length')
        if context_length:
            cmd.extend(["--context-length", str(context_length)])
        
        # 并行配置
        parallel_config = config.get('parallel', {})
        tp_size = args.tp_size or parallel_config.get('tp_size', 1)
        if tp_size > 1:
            cmd.extend(["--tp", str(tp_size)])
            
        # 性能优化配置
        opt_config = config.get('optimization', {})
        
        if args.enable_torch_compile or opt_config.get('enable_torch_compile'):
            cmd.append("--enable-torch-compile")
            
        if args.enable_flashinfer or opt_config.get('enable_flashinfer'):
            cmd.append("--enable-flashinfer")
            
        if args.disable_cuda_graph or opt_config.get('disable_cuda_graph'):
            cmd.append("--disable-cuda-graph")
            
        # 注意力配置
        attention_config = config.get('attention', {})
        if args.enable_dp_attention or attention_config.get('enable_dp_attention'):
            cmd.append("--enable-dp-attention")
            
        attention_backend = args.attention_backend or attention_config.get('attention_backend')
        if attention_backend:
            cmd.extend(["--attention-backend", attention_backend])
        
        # 分布式配置
        dist_config = config.get('distributed', {})
        dist_timeout = args.dist_timeout or dist_config.get('dist_timeout')
        if dist_timeout:
            cmd.extend(["--dist-timeout", str(dist_timeout)])
        
        return cmd
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description="启动 SGLang 服务器 - 支持多种量化配置",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
量化配置示例:
  1. 高性能 (内存充足):     --preset high_performance
  2. 平衡配置 (推荐):       --preset balanced  
  3. 内存优化:             --preset memory_optimized
  4. 超低内存:             --preset ultra_low_memory
  5. 自定义TorchAO:        --torchao-config int4wo-64
  6. 自定义传统量化:        --quantization fp8
  
支持的TorchAO量化配置:
  - int4wo-64, int4wo-128: INT4权重量化
  - int8wo, int8dq: INT8量化  
  - fp8wo, fp8dq-per_tensor: FP8量化
  - gemlite-4-64: GemLite内核优化
            """
        )
        
        # 配置文件相关
        parser.add_argument("--config", "-c", 
                           help="YAML配置文件路径 (默认: server_config.yaml)")
        parser.add_argument("--preset", 
                           choices=["high_performance", "balanced", "memory_optimized", "ultra_low_memory"],
                           help="使用预定义的量化配置预设")
        
        # 基础模型配置
        parser.add_argument("--model-path", 
                           default="/data/local_disk0/wuyu/model/qwen/Qwen3-4B",
                           help="模型路径")
        parser.add_argument("--trust-remote-code", action="store_true",
                           help="信任远程代码")
        parser.add_argument("--dtype", 
                           choices=["auto", "float16", "bfloat16", "float32"],
                           help="模型数据类型")
        
        # 服务器配置
        parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
        parser.add_argument("--port", default=30000, type=int, help="服务器端口")
        
        # 量化配置 (核心)
        quant_group = parser.add_argument_group("量化配置")
        quant_group.add_argument("--torchao-config",
                                choices=["int4wo-64", "int4wo-128", "int8wo", "int8dq", 
                                        "fp8wo", "fp8dq-per_tensor", "fp8dq-per_row",
                                        "gemlite-4-64", "gemlite-8-64"],
                                help="TorchAO量化配置 (推荐)")
        quant_group.add_argument("--quantization",
                                choices=["fp8", "awq", "gptq", "bitsandbytes", "marlin"],
                                help="传统量化方法")
        quant_group.add_argument("--kv-cache-dtype",
                                choices=["auto", "fp8_e5m2", "fp8_e4m3", "int8"],
                                help="KV缓存数据类型")
        
        # 内存管理
        memory_group = parser.add_argument_group("内存管理")
        memory_group.add_argument("--mem-fraction-static", type=float,
                                 help="静态内存分配比例 (0.8-0.9)")
        memory_group.add_argument("--chunked-prefill-size", type=int,
                                 help="分块预填充大小")
        memory_group.add_argument("--max-running-requests", type=int,
                                 help="最大并发请求数")
        memory_group.add_argument("--context-length", type=int,
                                 help="最大上下文长度")
        
        # 并行配置
        parallel_group = parser.add_argument_group("并行配置")
        parallel_group.add_argument("--tp-size", "--tp", type=int, default=1,
                                   help="张量并行度")
        
        # 性能优化
        opt_group = parser.add_argument_group("性能优化")
        opt_group.add_argument("--enable-torch-compile", action="store_true",
                              help="启用torch编译加速")
        opt_group.add_argument("--enable-flashinfer", action="store_true",
                              help="启用FlashInfer加速")
        opt_group.add_argument("--disable-cuda-graph", action="store_true",
                              help="禁用CUDA Graph")
        
        # 注意力配置
        attn_group = parser.add_argument_group("注意力配置")
        attn_group.add_argument("--enable-dp-attention", action="store_true",
                               help="启用数据并行注意力")
        attn_group.add_argument("--attention-backend",
                               choices=["flashinfer", "triton", "torch_naive"],
                               help="注意力后端")
        
        # 分布式配置
        dist_group = parser.add_argument_group("分布式配置")
        dist_group.add_argument("--dist-timeout", type=int,
                               help="分布式超时时间(秒)")
        
        return parser
    
    def print_config_summary(self, cmd: List[str], config: Dict[str, Any]):
        """打印配置摘要"""
        print("=" * 60)
        print("SGLang 服务器启动配置摘要")
        print("=" * 60)
        
        # 提取关键配置信息
        model_path = None
        host, port = "0.0.0.0", "30000"
        quantization_info = "无"
        tp_size = "1"
        
        for i, arg in enumerate(cmd):
            if arg == "--model-path" and i+1 < len(cmd):
                model_path = cmd[i+1]
            elif arg == "--host" and i+1 < len(cmd):
                host = cmd[i+1]
            elif arg == "--port" and i+1 < len(cmd):
                port = cmd[i+1]
            elif arg == "--torchao-config" and i+1 < len(cmd):
                quantization_info = f"TorchAO: {cmd[i+1]}"
            elif arg == "--quantization" and i+1 < len(cmd):
                quantization_info = f"传统量化: {cmd[i+1]}"
            elif arg == "--tp" and i+1 < len(cmd):
                tp_size = cmd[i+1]
        
        print(f"模型路径: {model_path}")
        print(f"服务地址: http://{host}:{port}")
        print(f"量化配置: {quantization_info}")
        print(f"张量并行: {tp_size}")
        
        # 显示优化选项
        optimizations = []
        if "--enable-torch-compile" in cmd:
            optimizations.append("Torch编译")
        if "--enable-flashinfer" in cmd:
            optimizations.append("FlashInfer")
        if "--enable-dp-attention" in cmd:
            optimizations.append("数据并行注意力")
        
        if optimizations:
            print(f"启用优化: {', '.join(optimizations)}")
        
        print("\n执行命令:")
        print(" ".join(cmd))
        print("=" * 60)
    
    def run(self):
        """主运行函数"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # 加载配置文件
        config_path = args.config or self.default_config_path
        self.config = self.load_config(config_path)
        
        # 应用量化预设
        if args.preset:
            self.config = self.apply_quantization_preset(self.config, args.preset)
        
        # 检查模型路径
        model_path = args.model_path or self.config.get('model', {}).get('model_path')
        if model_path and not os.path.exists(model_path):
            print(f"错误: 模型路径不存在: {model_path}")
            sys.exit(1)
        
        # 构建启动命令
        cmd = self.build_command(args, self.config)
        
        # 打印配置摘要
        self.print_config_summary(cmd, self.config)
        
        # 启动服务器
        try:
            print("\n正在启动 SGLang 服务器...")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n启动失败: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n用户中断，服务器关闭")
        except Exception as e:
            print(f"\n未知错误: {e}")
            sys.exit(1)

def main():
    launcher = SGLangServerLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 
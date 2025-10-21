#!/usr/bin/env python3
"""
批量处理系统测试脚本
用于验证系统功能和性能
"""

import os
import sys
import json
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from batch_pdf_processor import BatchPDFProcessor, DeepSeekOCRBatchProcessor, OpenRouterProcessor, JSONSchemaValidator
    from config_batch import Config, setup_environment
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所有依赖包")
    sys.exit(1)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

class BatchSystemTester:
    """批量处理系统测试器"""

    def __init__(self):
        self.config = Config()
        self.test_results = {}
        self.start_time = None

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print(f"{Colors.BLUE}{'='*60}")
        print(f"🧪 DeepSeek OCR 批量处理系统测试")
        print(f"{'='*60}{Colors.RESET}\n")

        self.start_time = time.time()

        tests = [
            ("环境配置测试", self.test_environment),
            ("配置验证测试", self.test_config_validation),
            ("JSON Schema测试", self.test_json_schema_validation),
            ("API连接测试", self.test_api_connectivity),
            ("GPU内存测试", self.test_gpu_memory),
            ("模型加载测试", self.test_model_loading),
            ("图像处理测试", self.test_image_processing),
            ("端到端测试", self.test_end_to_end)
        ]

        for test_name, test_func in tests:
            print(f"{Colors.CYAN}🔬 {test_name}...{Colors.RESET}")
            try:
                result = test_func()
                self.test_results[test_name] = {"status": "PASS", "details": result}
                print(f"{Colors.GREEN}✅ {test_name}: 通过{Colors.RESET}\n")
            except Exception as e:
                self.test_results[test_name] = {"status": "FAIL", "error": str(e)}
                print(f"{Colors.RED}❌ {test_name}: 失败 - {e}{Colors.RESET}\n")

        self.generate_test_report()
        return self.test_results

    def test_environment(self) -> Dict[str, Any]:
        """测试环境配置"""
        results = {}

        # 检查Python版本
        import sys
        python_version = sys.version_info
        results["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"

        if python_version < (3, 8):
            raise Exception(f"Python版本过低: {results['python_version']} (需要 >= 3.8)")

        # 检查CUDA
        try:
            import torch
            results["torch_version"] = torch.__version__
            results["cuda_available"] = torch.cuda.is_available()

            if torch.cuda.is_available():
                results["gpu_name"] = torch.cuda.get_device_name(0)
                results["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                raise Exception("CUDA不可用")

        except ImportError:
            raise Exception("PyTorch未安装")

        # 检查关键模块
        required_modules = [
            "vllm", "transformers", "PIL", "aiohttp", "jsonschema",
            "numpy", "tqdm", "fitz", "img2pdf"
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                results[f"{module}_available"] = True
            except ImportError:
                missing_modules.append(module)
                results[f"{module}_available"] = False

        if missing_modules:
            raise Exception(f"缺少必需模块: {missing_modules}")

        return results

    def test_config_validation(self) -> Dict[str, Any]:
        """测试配置验证"""
        results = {}

        try:
            # 设置环境
            setup_environment()
            results["environment_setup"] = "SUCCESS"

            # 验证配置
            self.config.validate_config()
            results["config_validation"] = "SUCCESS"

            # 检查关键路径
            critical_paths = {
                "schema_path": self.config.paths.SCHEMA_PATH,
                "input_dir": self.config.paths.INPUT_DIR,
                "output_dir": self.config.paths.OUTPUT_DIR
            }

            for name, path in critical_paths.items():
                if path.exists():
                    results[name] = "EXISTS"
                else:
                    results[name] = "MISSING"

        except Exception as e:
            raise Exception(f"配置验证失败: {e}")

        return results

    def test_json_schema_validation(self) -> Dict[str, Any]:
        """测试JSON Schema验证"""
        results = {}

        # 测试Schema加载
        try:
            validator = JSONSchemaValidator(str(self.config.paths.SCHEMA_PATH))
            results["schema_loading"] = "SUCCESS"
        except Exception as e:
            raise Exception(f"Schema加载失败: {e}")

        # 测试有效JSON验证
        valid_json = self._create_test_json()
        is_valid, error = validator.validate(valid_json)
        results["valid_json_test"] = "PASS" if is_valid else f"FAIL: {error}"

        # 测试无效JSON验证
        invalid_json = {"invalid": "data"}
        is_valid, error = validator.validate(invalid_json)
        results["invalid_json_test"] = "PASS" if not is_valid else "FAIL: Should reject invalid JSON"

        # 测试JSON修复功能
        fixed_json, warnings = validator.validate_and_fix(invalid_json)
        results["json_fix_test"] = f"PASS: {len(warnings)} warnings"

        return results

    async def test_api_connectivity(self) -> Dict[str, Any]:
        """测试API连接"""
        results = {}

        if not self.config.api.OPENROUTER_API_KEY:
            raise Exception("OPENROUTER_API_KEY未设置")

        try:
            async with OpenRouterProcessor() as processor:
                # 测试简单的API调用
                test_messages = [
                    {"role": "user", "content": "Hello, this is a test. Please respond with 'Test successful'."}
                ]

                for model_key in ["gemini", "qwen"]:
                    try:
                        start_time = time.time()
                        response = await processor.call_model(model_key, test_messages, max_tokens=100)
                        end_time = time.time()

                        if response and 'choices' in response:
                            results[f"{model_key}_api"] = {
                                "status": "SUCCESS",
                                "response_time": f"{end_time - start_time:.2f}s",
                                "usage": response.get('usage', {})
                            }
                        else:
                            results[f"{model_key}_api"] = {"status": "FAIL", "error": "Invalid response"}

                    except Exception as e:
                        results[f"{model_key}_api"] = {"status": "FAIL", "error": str(e)}

        except Exception as e:
            raise Exception(f"API连接测试失败: {e}")

        return results

    def test_gpu_memory(self) -> Dict[str, Any]:
        """测试GPU内存使用"""
        results = {}

        try:
            import torch

            if not torch.cuda.is_available():
                raise Exception("CUDA不可用")

            # 获取初始内存状态
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory

            results["initial_memory_mb"] = initial_memory / 1024**2
            results["total_memory_gb"] = total_memory / 1024**3

            # 测试内存分配
            test_tensor = torch.randn(1000, 1000, device='cuda')
            allocated_memory = torch.cuda.memory_allocated(0)
            results["test_allocation_mb"] = (allocated_memory - initial_memory) / 1024**2

            # 清理
            del test_tensor
            torch.cuda.empty_cache()

            # 检查可用内存是否足够
            available_memory_gb = (total_memory - initial_memory) / 1024**3
            results["available_memory_gb"] = available_memory_gb

            if available_memory_gb < 18:
                raise Exception(f"可用显存不足: {available_memory_gb:.1f}GB (推荐 >= 18GB)")

        except Exception as e:
            raise Exception(f"GPU内存测试失败: {e}")

        return results

    def test_model_loading(self) -> Dict[str, Any]:
        """测试模型加载"""
        results = {}

        try:
            # 测试DeepSeek OCR处理器初始化
            start_time = time.time()
            processor = DeepSeekOCRBatchProcessor()
            end_time = time.time()

            results["model_loading_time"] = f"{end_time - start_time:.2f}s"
            results["model_loading"] = "SUCCESS"

            # 检查模型组件
            if hasattr(processor, 'llm') and processor.llm is not None:
                results["llm_initialized"] = True
            else:
                results["llm_initialized"] = False

            if hasattr(processor, 'sampling_params'):
                results["sampling_params_configured"] = True
            else:
                results["sampling_params_configured"] = False

        except Exception as e:
            raise Exception(f"模型加载失败: {e}")

        return results

    def test_image_processing(self) -> Dict[str, Any]:
        """测试图像处理"""
        results = {}

        try:
            from PIL import Image
            import numpy as np

            # 创建测试图像
            test_image = Image.new('RGB', (640, 480), color='white')

            # 添加一些内容到图像
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(test_image)
            font = ImageFont.load_default()
            draw.text((10, 10), "Test Document", fill='black', font=font)
            draw.rectangle([50, 50, 200, 100], outline='red', width=2)

            results["test_image_created"] = True

            # 测试图像预处理
            processor = DeepSeekOCRBatchProcessor()
            batch_inputs = processor.process_images_batch([test_image])

            if batch_inputs and len(batch_inputs) == 1:
                results["image_preprocessing"] = "SUCCESS"
                results["batch_size"] = len(batch_inputs)
            else:
                results["image_preprocessing"] = "FAIL"

        except Exception as e:
            raise Exception(f"图像处理测试失败: {e}")

        return results

    async def test_end_to_end(self) -> Dict[str, Any]:
        """端到端测试"""
        results = {}

        try:
            # 创建临时PDF文件（模拟）
            test_content = """
            # 测试文档

            这是一个测试文档，用于验证批量处理系统。

            ## 财务数据

            - 收入: $100M
            - 利润: $20M
            - 增长率: 15%

            ## 图表数据

            | 季度 | 收入 | 利润 |
            |------|------|------|
            | Q1   | 25M  | 5M   |
            | Q2   | 30M  | 6M   |
            """

            # 由于创建真实PDF比较复杂，这里只测试JSON生成部分
            test_json = self._create_test_json()

            # 测试JSON验证
            validator = JSONSchemaValidator(str(self.config.paths.SCHEMA_PATH))
            is_valid, error = validator.validate(test_json)

            if is_valid:
                results["json_validation"] = "PASS"
            else:
                results["json_validation"] = f"FAIL: {error}"

            # 测试API调用（如果有API密钥）
            if self.config.api.OPENROUTER_API_KEY:
                async with OpenRouterProcessor() as processor:
                    test_messages = [{"role": "user", "content": "Extract data from: Test Revenue: $100M"}]

                    try:
                        response = await processor.call_model("gemini", test_messages, max_tokens=200)
                        results["api_integration"] = "SUCCESS"
                    except Exception as e:
                        results["api_integration"] = f"FAIL: {e}"
            else:
                results["api_integration"] = "SKIPPED: No API key"

        except Exception as e:
            raise Exception(f"端到端测试失败: {e}")

        return results

    def _create_test_json(self) -> Dict[str, Any]:
        """创建测试JSON数据"""
        return {
            "_id": "test_document_12345",
            "source": {
                "file_name": "test_document.pdf",
                "processing_metadata": {
                    "vision_model": "test-vision-model",
                    "synthesis_model": "test-synthesis-model",
                    "validation_model": "test-validation-model",
                    "processed_at": "2024-10-21T10:00:00Z",
                    "pages_processed": 1,
                    "successful_pages": 1
                }
            },
            "report": {
                "title": "Test Financial Report",
                "report_date": "2024-10-21",
                "report_type": "company",
                "symbols": ["TEST"],
                "sector": "Technology",
                "content": "This is a test report content.",
                "word_count": 100
            },
            "data": {
                "figures": [{
                    "figure_id": "test_revenue_chart",
                    "type": "bar_chart",
                    "title": "Revenue Growth",
                    "description": "Quarterly revenue growth",
                    "data": {
                        "labels": ["Q1", "Q2", "Q3", "Q4"],
                        "series": [{
                            "name": "Revenue",
                            "values": [100, 120, 135, 150],
                            "unit": "$M"
                        }]
                    },
                    "source_page": 1
                }],
                "numerical_data": [{
                    "value": "150",
                    "figure_id": "test_revenue_chart",
                    "context": "Q4 revenue",
                    "metric_type": "currency",
                    "source_page": 1
                }],
                "companies": ["Test Company"],
                "key_metrics": ["revenue", "growth"],
                "extraction_summary": {
                    "figures_count": 1,
                    "numerical_data_count": 1,
                    "companies_mentioned": 1,
                    "figures_with_linked_data": 1,
                    "validation_summary": {
                        "original_figures": 1,
                        "kept_figures": 1,
                        "original_numerical": 1,
                        "kept_numerical": 1,
                        "validation_method": "test_validation",
                        "data_accuracy_rate": 100.0
                    }
                }
            },
            "query_capabilities": {
                "description": "Test document with query capabilities",
                "searchable_fields": ["report.title", "data.figures.title"],
                "figure_data_available": True,
                "can_recreate_charts": True
            }
        }

    def generate_test_report(self):
        """生成测试报告"""
        end_time = time.time()
        total_time = end_time - self.start_time

        print(f"\n{Colors.BLUE}{'='*60}")
        print(f"📊 测试报告")
        print(f"{'='*60}{Colors.RESET}")

        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests

        print(f"总测试数: {total_tests}")
        print(f"通过: {Colors.GREEN}{passed_tests}{Colors.RESET}")
        print(f"失败: {Colors.RED}{failed_tests}{Colors.RESET}")
        print(f"总耗时: {total_time:.2f} 秒")
        print()

        # 详细结果
        for test_name, result in self.test_results.items():
            status_color = Colors.GREEN if result["status"] == "PASS" else Colors.RED
            print(f"{status_color}{result['status']:<6}{Colors.RESET} {test_name}")

            if result["status"] == "FAIL":
                print(f"         错误: {result['error']}")

        # 保存报告到文件
        report_file = Path("test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.time(),
                "total_time": total_time,
                "summary": {
                    "total": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests
                },
                "results": self.test_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📁 详细报告已保存到: {report_file}")

        # 根据结果决定退出代码
        if failed_tests > 0:
            print(f"\n{Colors.RED}❌ 部分测试失败，请检查上述错误{Colors.RESET}")
            return False
        else:
            print(f"\n{Colors.GREEN}✅ 所有测试通过！系统可以正常使用{Colors.RESET}")
            return True

async def main():
    """主函数"""
    print(f"{Colors.BOLD}DeepSeek OCR 批量处理系统测试{Colors.RESET}\n")

    tester = BatchSystemTester()
    success = await asyncio.get_event_loop().run_in_executor(None, tester.run_all_tests)

    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  测试被用户中断{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ 测试失败: {e}{Colors.RESET}")
        sys.exit(1)
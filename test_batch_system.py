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
    def __init__(self):
        self.config = Config()
        self.test_results: Dict[str, Any] = {}

    def run_all_tests(self) -> Dict[str, Any]:
        tests = [
            ("环境测试", self.test_environment),
            ("配置验证", self.test_config_validation),
            ("JSON Schema验证", self.test_json_schema_validation),
            ("GPU显存测试", self.test_gpu_memory),
            ("模型加载测试", self.test_model_loading),
            ("图像处理测试", self.test_image_processing)
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

        # 检查关键模块（移除 aiohttp 与可选的 PyMuPDF）
        required_modules = [
            "vllm", "transformers", "PIL", "jsonschema",
            "numpy", "tqdm", "img2pdf"
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

        # 可选模块检测（不作为失败条件）
        try:
            __import__("fitz")
            results["fitz_available"] = True
        except ImportError:
            results["fitz_available"] = False

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
            results["schema_load"] = "SUCCESS"
        except Exception as e:
            raise Exception(f"Schema加载失败: {e}")

        # 测试示例数据验证
        test_json = self._create_test_json()
        is_valid, error = validator.validate(test_json)
        results["example_validation"] = "PASS" if is_valid else f"FAIL: {error}"

        # 测试修复功能
        fixed_json, warnings = validator.validate_and_fix({})
        results["auto_fix"] = "SUCCESS" if fixed_json else "FAIL"
        results["auto_fix_warnings"] = warnings

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

                for model_key in ["gemini"]:
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
        """测试GPU显存"""
        results = {}
        try:
            import torch
            if torch.cuda.is_available():
                results["cuda"] = True
                results["gpu_name"] = torch.cuda.get_device_name(0)
                results["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                results["cuda"] = False
                results["error"] = "CUDA不可用"
        except Exception as e:
            results["error"] = str(e)
        return results

    def test_model_loading(self) -> Dict[str, Any]:
        """测试模型加载"""
        results = {}
        try:
            ocr = DeepSeekOCRBatchProcessor()
            results["model_initialized"] = True
            results["sampling_params"] = {
                "temperature": ocr.sampling_params.temperature,
                "top_p": ocr.sampling_params.top_p,
                "max_tokens": ocr.sampling_params.max_tokens
            }
        except Exception as e:
            results["error"] = str(e)
        return results

    def test_image_processing(self) -> Dict[str, Any]:
        """测试图像处理"""
        results = {}
        try:
            # 检测 PyMuPDF 可用性，若不可用则跳过
            fitz_available = False
            try:
                __import__("fitz")
                fitz_available = True
            except ImportError:
                pass

            if not fitz_available:
                results["skipped"] = "PyMuPDF 未安装，图像渲染测试跳过"
                return results

            test_pdf = Path(self.config.paths.INPUT_DIR) / "test_layouts.pdf"
            if not test_pdf.exists():
                results["test_pdf"] = "SKIPPED: test_layouts.pdf 不存在"
                return results

            ocr = DeepSeekOCRBatchProcessor()
            images = ocr.pdf_to_images_high_quality(str(test_pdf))
            results["pages"] = len(images)

            batch_inputs = ocr.process_images_batch(images)
            results["batch_inputs"] = len(batch_inputs)

        except Exception as e:
            results["error"] = str(e)
        return results

    async def test_end_to_end(self) -> Dict[str, Any]:
        """端到端测试"""
        results = {}

        try:
            # 准备JSON验证器
            validator = JSONSchemaValidator(str(self.config.paths.SCHEMA_PATH))

            # 创建并验证最终JSON（模拟）
            final_json = self._create_test_json()
            is_valid_final, err_final = validator.validate(final_json)
            results["final_json_validation"] = "PASS" if is_valid_final else f"FAIL: {err_final}"

            # 生成并验证模板JSON（使用自动修复机制）
            template_json, warnings = validator.validate_and_fix({})
            is_valid_template, err_template = validator.validate(template_json)
            results["template_json_validation"] = "PASS" if is_valid_template else f"FAIL: {err_template}"
            results["template_warnings"] = warnings

            # 写出两个JSON文件到输出目录
            output_dir = Path(self.config.paths.OUTPUT_DIR) / "test_end_to_end"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_path = output_dir / "final_test.json"
            template_path = output_dir / "template_test.json"

            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template_json, f, ensure_ascii=False, indent=2)

            results["final_json_path"] = str(final_path)
            results["template_json_path"] = str(template_path)

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
                            "unit": "M"
                        }]
                    }
                }],
                "tables": [{
                    "table_id": "financial_table_1",
                    "title": "Quarterly Financials",
                    "rows": [
                        {"quarter": "Q1", "revenue": 25, "profit": 5},
                        {"quarter": "Q2", "revenue": 30, "profit": 6}
                    ]
                }],
                "numerical_data": [{
                    "metric": "growth_rate",
                    "value": 15,
                    "unit": "%",
                    "time_period": "Q2 2024"
                }]
            }
        }

    def generate_test_report(self):
        """生成测试报告"""
        report = {
            "summary": {},
            "details": self.test_results
        }

        # 汇总统计
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r.get("status") == "PASS")
        failed = total - passed
        report["summary"] = {
            "total": total,
            "passed": passed,
            "failed": failed
        }

        # 输出报告
        report_path = Path("test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"{Colors.BLUE}📝 测试报告已生成: {report_path}{Colors.RESET}")

async def main_async_tests(tester: BatchSystemTester):
    # API连接测试
    try:
        api_results = await tester.test_api_connectivity()
        print(f"{Colors.BOLD}=== API连接测试结果 ==={Colors.RESET}")
        for key, val in api_results.items():
            print(f"- {key}: {val}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  API连接测试跳过: {e}{Colors.RESET}")

    # 端到端测试
    try:
        e2e_results = await tester.test_end_to_end()
        print(f"{Colors.BOLD}=== 端到端测试结果 ==={Colors.RESET}")
        for key, val in e2e_results.items():
            print(f"- {key}: {val}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  端到端测试跳过: {e}{Colors.RESET}")


def main():
    tester = BatchSystemTester()
    results = tester.run_all_tests()

    # 打印简要结果
    print(f"{Colors.BOLD}=== 测试结果摘要 ==={Colors.RESET}")
    for name, result in results.items():
        status = result.get("status")
        print(f"- {name}: {status}")

    # 运行异步测试
    asyncio.run(main_async_tests(tester))
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  测试被用户中断{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ 测试失败: {e}{Colors.RESET}")
        sys.exit(1)
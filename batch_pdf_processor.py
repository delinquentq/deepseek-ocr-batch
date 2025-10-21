#!/usr/bin/env python3
"""
批量PDF处理系统 - 针对RTX 3090 24G优化
基于DeepSeek OCR + OpenRouter双模型的完整处理流程

功能:
1. 批量PDF → Markdown + 图像提取
2. OpenRouter API调用 (Gemini 2.5 Flash + Qwen3-VL-30B)
3. 严格JSON Schema验证
4. 数据库兼容性检查
"""

import os
import sys
import json
import time
import hashlib
import asyncio
import aiohttp
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# DeepSeek OCR imports
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# JSON Schema validation
import jsonschema
from jsonschema import validate, ValidationError

# Configuration
from config import MODEL_PATH, PROMPT, CROP_MODE

# 针对RTX 3090 24G的优化配置
@dataclass
class Config:
    # DeepSeek OCR配置
    BATCH_SIZE: int = 4  # 每批处理页数
    MAX_CONCURRENCY: int = 6  # 降低并发数以节省显存
    GPU_MEMORY_UTILIZATION: float = 0.75  # 为后续LLM调用保留显存
    NUM_WORKERS: int = 8  # 图像预处理线程数

    # OpenRouter API配置
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # 测试模型配置
    MODELS = {
        "gemini": "google/gemini-2.5-flash",
        "qwen": "qwen/qwen-2.5-vl-72b-instruct"
    }

    # 文件路径配置
    INPUT_DIR: str = "input_pdfs"
    OUTPUT_DIR: str = "output_results"
    TEMP_DIR: str = "temp_processing"

    # 处理配置
    PDF_DPI: int = 144
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 300

    # JSON Schema路径
    SCHEMA_PATH: str = "json schema.json"

# 全局配置实例
config = Config()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RESET = '\033[0m'

class DeepSeekOCRBatchProcessor:
    """DeepSeek OCR批量处理器"""

    def __init__(self):
        self.model_registry_initialized = False
        self.llm = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化DeepSeek OCR模型"""
        try:
            if not self.model_registry_initialized:
                ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
                self.model_registry_initialized = True

            # 针对3090优化的模型配置
            self.llm = LLM(
                model=MODEL_PATH,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                enforce_eager=False,
                trust_remote_code=True,
                max_model_len=8192,
                swap_space=0,
                max_num_seqs=config.MAX_CONCURRENCY,
                tensor_parallel_size=1,
                gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
                disable_mm_preprocessor_cache=True
            )

            # 配置采样参数
            logits_processors = [NoRepeatNGramLogitsProcessor(
                ngram_size=20, window_size=50,
                whitelist_token_ids={128821, 128822}
            )]

            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )

            logger.info(f"{Colors.GREEN}✓ DeepSeek OCR模型初始化完成{Colors.RESET}")

        except Exception as e:
            logger.error(f"{Colors.RED}✗ DeepSeek OCR模型初始化失败: {e}{Colors.RESET}")
            raise

    def pdf_to_images_high_quality(self, pdf_path: str, dpi: int = None) -> List[Image.Image]:
        """高质量PDF转图像"""
        if dpi is None:
            dpi = config.PDF_DPI

        images = []
        try:
            pdf_document = fitz.open(pdf_path)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                Image.MAX_IMAGE_PIXELS = None

                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background

                images.append(img)

            pdf_document.close()
            logger.info(f"PDF转换完成: {len(images)}页")
            return images

        except Exception as e:
            logger.error(f"PDF转换失败: {e}")
            raise

    def process_images_batch(self, images: List[Image.Image]) -> List[Dict]:
        """批量处理图像"""
        batch_inputs = []

        def process_single_image(image):
            prompt_in = PROMPT
            return {
                "prompt": prompt_in,
                "multi_modal_data": {
                    "image": DeepseekOCRProcessor().tokenize_with_images(
                        images=[image], bos=True, eos=True, cropping=CROP_MODE
                    )
                },
            }

        # 使用线程池处理图像预处理
        with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc=f"{Colors.CYAN}预处理图像{Colors.RESET}"
            ))

        return batch_inputs

    def extract_figures_and_markdown(self, outputs_list: List, images: List[Image.Image],
                                   output_dir: str) -> Tuple[str, List[str]]:
        """提取图表和生成Markdown"""
        os.makedirs(f"{output_dir}/images", exist_ok=True)

        contents = ""
        figure_paths = []

        for idx, (output, img) in enumerate(zip(outputs_list, images)):
            content = output.outputs[0].text

            # 处理结束标记
            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')

            # 提取图表引用
            matches_ref, matches_images, matches_other = self._extract_references(content)

            # 保存图表
            image_draw = img.copy()
            result_image = self._process_image_with_refs(image_draw, matches_ref, idx, output_dir)

            # 更新图表路径引用
            for img_idx, match_image in enumerate(matches_images):
                figure_path = f"images/{idx}_{img_idx}.jpg"
                figure_paths.append(os.path.join(output_dir, figure_path))
                content = content.replace(match_image, f"![Figure {idx}-{img_idx}]({figure_path})\n")

            # 清理其他引用
            for match_other in matches_other:
                content = content.replace(match_other, '').replace(
                    '\\coloneqq', ':='
                ).replace('\\eqqcolon', '=:')

            page_separator = f'\n\n--- Page {idx + 1} ---\n\n'
            contents += content + page_separator

        return contents, figure_paths

    def _extract_references(self, text: str) -> Tuple[List, List, List]:
        """提取文本中的引用"""
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)

        matches_image = []
        matches_other = []

        for match in matches:
            if '<|ref|>image<|/ref|>' in match[0]:
                matches_image.append(match[0])
            else:
                matches_other.append(match[0])

        return matches, matches_image, matches_other

    def _process_image_with_refs(self, image: Image.Image, refs: List,
                               page_idx: int, output_dir: str) -> Image.Image:
        """处理图像引用并保存图表"""
        image_width, image_height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        font = ImageFont.load_default()
        img_idx = 0

        for ref in refs:
            try:
                result = self._extract_coordinates_and_label(ref, image_width, image_height)
                if result and result[0] == 'image':
                    label_type, points_list = result

                    for points in points_list:
                        x1, y1, x2, y2 = points
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)

                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            figure_path = f"{output_dir}/images/{page_idx}_{img_idx}.jpg"
                            cropped.save(figure_path)
                            img_idx += 1
                        except Exception as e:
                            logger.warning(f"图表提取失败: {e}")
            except Exception as e:
                logger.warning(f"引用处理失败: {e}")
                continue

        return img_draw

    def _extract_coordinates_and_label(self, ref_text: Tuple,
                                     image_width: int, image_height: int) -> Optional[Tuple]:
        """提取坐标和标签"""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
            return (label_type, cor_list)
        except Exception as e:
            logger.warning(f"坐标提取失败: {e}")
            return None

    def process_pdf(self, pdf_path: str, output_dir: str) -> Tuple[str, List[str]]:
        """处理单个PDF文件"""
        logger.info(f"{Colors.BLUE}开始处理PDF: {pdf_path}{Colors.RESET}")

        try:
            # 1. PDF转图像
            images = self.pdf_to_images_high_quality(pdf_path)

            # 2. 批量处理
            batch_inputs = self.process_images_batch(images)

            # 3. OCR推理
            logger.info(f"{Colors.CYAN}开始OCR推理...{Colors.RESET}")
            outputs_list = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)

            # 4. 提取结果
            markdown_content, figure_paths = self.extract_figures_and_markdown(
                outputs_list, images, output_dir
            )

            # 5. 保存Markdown
            pdf_name = Path(pdf_path).stem
            markdown_path = os.path.join(output_dir, f"{pdf_name}.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"{Colors.GREEN}✓ PDF处理完成: {pdf_path}{Colors.RESET}")
            return markdown_path, figure_paths

        except Exception as e:
            logger.error(f"{Colors.RED}✗ PDF处理失败 {pdf_path}: {e}{Colors.RESET}")
            raise

class OpenRouterProcessor:
    """OpenRouter API处理器"""

    def __init__(self):
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY环境变量未设置")

        self.session = None
        self.headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "DeepSeek OCR Batch Processor"
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
            headers=self.headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def call_model(self, model_name: str, messages: List[Dict],
                        max_tokens: int = 4000) -> Dict:
        """调用OpenRouter模型"""
        payload = {
            "model": config.MODELS[model_name],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9
        }

        for attempt in range(config.MAX_RETRIES):
            try:
                async with self.session.post(
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.warning(f"API调用失败 (尝试 {attempt + 1}/{config.MAX_RETRIES}): "
                                     f"状态码 {response.status}, 错误: {error_text}")

            except Exception as e:
                logger.warning(f"API调用异常 (尝试 {attempt + 1}/{config.MAX_RETRIES}): {e}")

                if attempt < config.MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)

        raise Exception(f"API调用失败，已重试 {config.MAX_RETRIES} 次")

class JSONSchemaValidator:
    """JSON Schema验证器"""

    def __init__(self, schema_path: str):
        self.schema = self._load_schema(schema_path)

    def _load_schema(self, schema_path: str) -> Dict:
        """加载JSON Schema"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"{Colors.GREEN}✓ JSON Schema加载完成{Colors.RESET}")
            return schema
        except Exception as e:
            logger.error(f"{Colors.RED}✗ JSON Schema加载失败: {e}{Colors.RESET}")
            raise

    def validate(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """验证JSON数据"""
        try:
            validate(instance=data, schema=self.schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"验证异常: {e}"

    def validate_and_fix(self, data: Dict) -> Tuple[Dict, List[str]]:
        """验证并尝试修复JSON数据"""
        warnings = []

        # 确保必需字段存在
        required_fields = {
            "_id": self._generate_id(),
            "source": {},
            "report": {},
            "data": {
                "figures": [],
                "numerical_data": [],
                "companies": [],
                "key_metrics": [],
                "extraction_summary": {}
            },
            "query_capabilities": {}
        }

        # 填充缺失的字段
        for key, default_value in required_fields.items():
            if key not in data:
                data[key] = default_value
                warnings.append(f"添加缺失字段: {key}")

        # 验证数据类型
        is_valid, error_msg = self.validate(data)
        if not is_valid:
            warnings.append(f"Schema验证失败: {error_msg}")

        return data, warnings

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()

class BatchPDFProcessor:
    """批量PDF处理主类"""

    def __init__(self):
        self.ocr_processor = DeepSeekOCRBatchProcessor()
        self.validator = JSONSchemaValidator(config.SCHEMA_PATH)
        self._setup_directories()

    def _setup_directories(self):
        """设置目录结构"""
        for dir_path in [config.INPUT_DIR, config.OUTPUT_DIR, config.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    async def process_single_pdf(self, pdf_path: str) -> Dict:
        """处理单个PDF并生成JSON"""
        pdf_name = Path(pdf_path).stem
        output_dir = os.path.join(config.OUTPUT_DIR, pdf_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. DeepSeek OCR处理
            logger.info(f"{Colors.BLUE}步骤1: DeepSeek OCR处理{Colors.RESET}")
            markdown_path, figure_paths = self.ocr_processor.process_pdf(pdf_path, output_dir)

            # 2. 读取Markdown内容
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # 3. 双模型处理和对比
            logger.info(f"{Colors.BLUE}步骤2: 双模型处理{Colors.RESET}")
            results = await self._process_with_dual_models(
                markdown_content, figure_paths, pdf_name
            )

            # 4. 选择最佳结果
            best_result = self._select_best_result(results)

            # 5. JSON验证和修复
            logger.info(f"{Colors.BLUE}步骤3: JSON验证{Colors.RESET}")
            validated_result, warnings = self.validator.validate_and_fix(best_result)

            if warnings:
                logger.warning(f"JSON修复警告: {warnings}")

            # 6. 保存最终结果
            result_path = os.path.join(output_dir, f"{pdf_name}_final.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(validated_result, f, indent=2, ensure_ascii=False)

            logger.info(f"{Colors.GREEN}✓ PDF处理完成: {pdf_path}{Colors.RESET}")
            return validated_result

        except Exception as e:
            logger.error(f"{Colors.RED}✗ PDF处理失败 {pdf_path}: {e}{Colors.RESET}")
            traceback.print_exc()
            raise

    async def _process_with_dual_models(self, markdown_content: str,
                                       figure_paths: List[str], pdf_name: str) -> Dict:
        """双模型处理和对比"""
        results = {}

        # 准备提示词
        extraction_prompt = self._build_extraction_prompt(markdown_content, figure_paths)

        async with OpenRouterProcessor() as processor:
            # 并行调用两个模型
            tasks = []
            for model_key in ["gemini", "qwen"]:
                task = self._call_model_with_prompt(
                    processor, model_key, extraction_prompt, pdf_name
                )
                tasks.append(task)

            # 等待所有任务完成
            model_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for model_key, result in zip(["gemini", "qwen"], model_results):
                if isinstance(result, Exception):
                    logger.error(f"{model_key}模型调用失败: {result}")
                    results[model_key] = None
                else:
                    results[model_key] = result

        return results

    async def _call_model_with_prompt(self, processor: OpenRouterProcessor,
                                    model_key: str, prompt: str, pdf_name: str) -> Dict:
        """调用单个模型"""
        logger.info(f"{Colors.CYAN}调用{model_key}模型...{Colors.RESET}")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的金融文档数据提取专家。请严格按照提供的JSON schema格式输出结果。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = await processor.call_model(model_key, messages, max_tokens=8000)

            # 提取JSON内容
            content = response['choices'][0]['message']['content']
            json_result = self._extract_json_from_response(content)

            logger.info(f"{Colors.GREEN}✓ {model_key}模型处理完成{Colors.RESET}")
            return {
                "model": model_key,
                "result": json_result,
                "raw_response": content,
                "usage": response.get('usage', {})
            }

        except Exception as e:
            logger.error(f"{Colors.RED}✗ {model_key}模型调用失败: {e}{Colors.RESET}")
            raise

    def _build_extraction_prompt(self, markdown_content: str, figure_paths: List[str]) -> str:
        """构建数据提取提示词"""

        # 读取JSON schema作为参考
        with open(config.SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_content = f.read()

        prompt = f"""
# 任务：金融文档数据提取

## 输入材料：
1. **Markdown内容**：
```markdown
{markdown_content[:10000]}...  # 限制长度避免token超限
```

2. **图表文件数量**：{len(figure_paths)}个图表已提取

## 输出要求：
请严格按照以下JSON Schema格式提取数据：

```json
{schema_content}
```

## 关键要求：
1. **图表数据完整性**：确保所有图表的data字段包含完整的原始数据，能够直接重建图表
2. **数值精确性**：所有数值保持原始格式，不要进行单位转换
3. **关联性**：通过figure_id建立数值数据与图表的关联
4. **类型严格性**：严格遵循schema中的数据类型定义

## 特别注意：
- figures.data字段必须包含完整的labels和series数据
- numerical_data中每个数据点都要关联到具体的figure_id或标明为null
- 所有必需字段都必须存在，不能为空
- 输出纯JSON格式，不要包含其他文本

请开始提取：
"""
        return prompt

    def _extract_json_from_response(self, response: str) -> Dict:
        """从响应中提取JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except:
            # 尝试提取JSON代码块
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 尝试查找第一个完整的JSON对象
            start_idx = response.find('{')
            if start_idx != -1:
                # 简单的括号匹配
                bracket_count = 0
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            return json.loads(response[start_idx:i+1])

            raise ValueError("无法从响应中提取有效的JSON")

    def _select_best_result(self, results: Dict) -> Dict:
        """选择最佳模型结果"""
        # 简单的选择策略：优先选择无异常的结果
        for model_key in ["gemini", "qwen"]:  # gemini优先
            if results.get(model_key) is not None:
                result_data = results[model_key]["result"]

                # 简单的质量检查
                if self._validate_result_quality(result_data):
                    logger.info(f"{Colors.GREEN}选择{model_key}模型结果{Colors.RESET}")
                    return result_data

        # 如果都有问题，返回第一个可用的
        for model_key, result in results.items():
            if result is not None:
                logger.warning(f"{Colors.YELLOW}使用{model_key}模型结果（质量可能有问题）{Colors.RESET}")
                return result["result"]

        raise ValueError("所有模型都失败了")

    def _validate_result_quality(self, data: Dict) -> bool:
        """验证结果质量"""
        try:
            # 基本结构检查
            required_fields = ["_id", "source", "report", "data", "query_capabilities"]
            if not all(field in data for field in required_fields):
                return False

            # 数据完整性检查
            if not data.get("data", {}).get("figures"):
                return False

            # 图表数据检查
            for figure in data["data"]["figures"]:
                if not figure.get("data"):
                    return False

            return True
        except:
            return False

    async def process_batch(self, pdf_paths: List[str]) -> List[Dict]:
        """批量处理PDF文件"""
        logger.info(f"{Colors.BLUE}开始批量处理 {len(pdf_paths)} 个PDF文件{Colors.RESET}")

        results = []
        failed_files = []

        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.info(f"{Colors.CYAN}处理进度: {i}/{len(pdf_paths)} - {Path(pdf_path).name}{Colors.RESET}")
                result = await self.process_single_pdf(pdf_path)
                results.append(result)

            except Exception as e:
                logger.error(f"{Colors.RED}文件处理失败: {pdf_path} - {e}{Colors.RESET}")
                failed_files.append(pdf_path)
                continue

        # 输出处理摘要
        logger.info(f"{Colors.GREEN}批量处理完成！{Colors.RESET}")
        logger.info(f"成功处理: {len(results)} 个文件")
        logger.info(f"失败文件: {len(failed_files)} 个")

        if failed_files:
            logger.warning(f"失败文件列表: {failed_files}")

        return results

async def main():
    """主函数"""
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}DeepSeek OCR 批量处理系统{Colors.RESET}")
    print(f"{Colors.BLUE}针对RTX 3090 24G优化{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    # 检查环境
    if not config.OPENROUTER_API_KEY:
        logger.error(f"{Colors.RED}请设置OPENROUTER_API_KEY环境变量{Colors.RESET}")
        return

    # 查找PDF文件
    pdf_files = list(Path(config.INPUT_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"{Colors.YELLOW}在 {config.INPUT_DIR} 目录中未找到PDF文件{Colors.RESET}")
        return

    logger.info(f"找到 {len(pdf_files)} 个PDF文件")

    # 初始化处理器
    processor = BatchPDFProcessor()

    # 开始批量处理
    start_time = time.time()
    results = await processor.process_batch([str(f) for f in pdf_files])
    end_time = time.time()

    # 输出最终统计
    print(f"\n{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"{Colors.GREEN}处理完成！{Colors.RESET}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功处理: {len(results)} 个文件")
    print(f"平均处理时间: {(end_time - start_time) / len(pdf_files):.2f} 秒/文件")
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}")

if __name__ == "__main__":
    # 设置环境变量
    if torch.version.cuda == '11.8':
        os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
    os.environ['VLLM_USE_V1'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 运行主程序
    asyncio.run(main())
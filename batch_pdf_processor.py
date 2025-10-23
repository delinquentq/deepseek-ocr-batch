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
from openai import AsyncOpenAI
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# DeepSeek OCR imports
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False
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
# 预加载 .env 环境变量（确保在定义 Config 之前）
try:
    from config_batch import setup_environment as _setup_environment
    _setup_environment()
except Exception:
    try:
        base_dir = Path(__file__).resolve().parent
        env_path = base_dir / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if (k not in os.environ) or (os.environ.get(k, "") == ""):
                            os.environ[k] = v
    except Exception:
        pass

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
        "gemini": "google/gemini-2.5-flash"
    }

    # 文件路径配置（使用脚本目录的绝对路径，避免CWD差异）
    BASE_DIR: str = str(Path(__file__).resolve().parent)
    INPUT_DIR: str = str(Path(BASE_DIR) / "input_pdfs")
    OUTPUT_DIR: str = str(Path(BASE_DIR) / "output_results")
    TEMP_DIR: str = str(Path(BASE_DIR) / "temp_processing")

    # 处理配置
    PDF_DPI: int = 144
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 300

    # JSON Schema路径（绝对路径）
    SCHEMA_PATH: str = str(Path(BASE_DIR) / "json schema.json")

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
        """高质量PDF转图像（支持在未安装PyMuPDF时给出提示）"""
        if dpi is None:
            dpi = config.PDF_DPI

        images = []
        if not HAS_PYMUPDF:
            logger.warning("未检测到 PyMuPDF(fitz)，跳过 PDF 渲染。若模型可直接处理PDF，此步骤可忽略。")
            raise RuntimeError("PyMuPDF 未安装，无法将 PDF 转为图片。")

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
            if '<｜end of sentence｜>' in content:
                content = content.replace('<｜end of sentence｜>', '')

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
    """OpenRouter API处理器 - 使用OpenAI SDK"""

    def __init__(self):
        # 动态读取环境变量，避免模块级配置在进程启动时固定
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY环境变量未设置")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.OPENROUTER_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=config.MAX_RETRIES,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "DeepSeek OCR Batch Processor"
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.client.close()
        except Exception:
            pass

    async def call_model(self, model_name: str, messages: List[Dict],
                        max_tokens: int = 4000) -> Dict:
        """调用OpenRouter模型 (OpenAI SDK)"""
        try:
            # 优先使用 JSON 强格式输出
            response = await self.client.chat.completions.create(
                model=config.MODELS[model_name],
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"}
            )
            return response.model_dump()
        except Exception as e:
            logger.warning(f"JSON 强格式输出失败，回退到普通输出: {e}")
            # 回退到非 JSON 强格式
            response = await self.client.chat.completions.create(
                model=config.MODELS[model_name],
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9
            )
            return response.model_dump()

    async def collect_stream_content(self, model_name: str, messages: List[Dict],
                                     max_tokens: int = 4000) -> str:
        """以流式方式调用模型并收集完整文本，默认启用JSON强格式输出"""
        content = ""
        try:
            stream = await self.client.chat.completions.create(
                model=config.MODELS[model_name],
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
                stream=True
            )
            async for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        content += delta.content
                except Exception:
                    # 容错处理，忽略单个流事件中的异常
                    pass
            return content
        except Exception as e:
            logger.warning(f"流式调用失败，回退到非流式: {e}")
            # 回退到非流式普通调用，优先尝试JSON强格式
            try:
                resp = await self.client.chat.completions.create(
                    model=config.MODELS[model_name],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    response_format={"type": "json_object"}
                )
                return resp.choices[0].message.content
            except Exception as e2:
                logger.warning(f"JSON 强格式失败，回退到普通输出: {e2}")
                resp2 = await self.client.chat.completions.create(
                    model=config.MODELS[model_name],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9
                )
                return resp2.choices[0].message.content

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
        """验证并尝试修复JSON数据（适配 v1.3.1 schema）"""
        warnings: List[str] = []

        # 补齐顶层必需字段
        if "schema_version" not in data:
            data["schema_version"] = "1.3.1"
            warnings.append("添加缺失字段: schema_version")

        if "doc" not in data:
            data["doc"] = {
                "doc_id": self._generate_id(),
                "title": "",
                "source_uri": "",
                "language": "zh",
                "timestamps": {
                    "ingested_at": datetime.now().isoformat(),
                    "extracted_at": datetime.now().isoformat()
                },
                "extraction_run": {
                    "vision_model": "DeepSeek OCR",
                    "synthesis_model": "google/gemini-2.5-flash",
                    "pipeline_steps": ["ocr", "figures_parallel", "llm_synthesis"],
                    "processing_metadata": {
                        "pages_processed": 0,
                        "successful_pages": 0,
                        "notes": "auto-filled"
                    }
                }
            }
            warnings.append("添加缺失字段: doc")

        for key in ["passages", "entities", "data"]:
            if key not in data:
                data[key] = [] if key != "data" else {
                    "figures": [],
                    "tables": [],
                    "numerical_data": [],
                    "claims": [],
                    "relations": [],
                    "extraction_summary": {
                        "figures_count": 0,
                        "tables_count": 0,
                        "numerical_data_count": 0
                    }
                }
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
        """处理单个PDF并生成JSON（单模型 + 图像并发）"""
        pdf_path_obj = Path(pdf_path).resolve()
        # 依据输入目录名定位根目录，避免工作目录差异导致 relative_to 失败
        input_root_name = Path(config.INPUT_DIR).name
        rel_parent = Path()
        for parent in pdf_path_obj.parents:
            if parent.name == input_root_name:
                rel = pdf_path_obj.relative_to(parent)
                rel_parent = rel.parent
                break
        pdf_name = pdf_path_obj.stem
        output_dir = str(Path(config.OUTPUT_DIR) / rel_parent / pdf_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. DeepSeek OCR处理
            logger.info(f"{Colors.BLUE}步骤1: DeepSeek OCR处理{Colors.RESET}")
            markdown_path, figure_paths = self.ocr_processor.process_pdf(pdf_path, output_dir)

            # 2. 读取Markdown内容（完整，无截断）
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            page_count = self._count_pages_from_markdown(markdown_content)

            # 3. 并行处理每张提取图像（视觉JSON）
            logger.info(f"{Colors.BLUE}步骤2: 并行图像结构化{Colors.RESET}")
            figures_json = await self._process_figures_in_parallel(figure_paths, markdown_content)

            # 4. 单模型综合提取（优先流式+JSON强格式）
            logger.info(f"{Colors.BLUE}步骤3: 单模型综合提取{Colors.RESET}")
            results = await self._process_with_single_model(
                markdown_content, figures_json, pdf_name
            )

            best_result = results["gemini"]

            # 5. JSON验证和修复
            logger.info(f"{Colors.BLUE}步骤4: JSON验证{Colors.RESET}")
            # 补充页数到 doc.extraction_run.processing_metadata
            if "doc" in best_result and "extraction_run" in best_result["doc"]:
                md = best_result["doc"].get("extraction_run", {}).get("processing_metadata", {})
                md.setdefault("pages_processed", page_count)
                md.setdefault("successful_pages", page_count)
                # 增加基于输入相对子路径的分类元信息
                try:
                    # 记录相对子路径（用于按日期/刊物分类）
                    md["input_relative_path"] = str(rel_parent)
                    parts = list(rel_parent.parts)
                    # 简单日期格式 YYYY.MM.DD
                    import re
                    if len(parts) >= 1 and re.match(r"^\d{4}\.\d{2}\.\d{2}$", parts[0]):
                        md["date"] = parts[0]
                        if len(parts) >= 2:
                            md["publication"] = parts[1]
                    else:
                        # 若首层不是日期，则作为刊物名；若次层是日期则记录
                        md["publication"] = parts[0] if parts else md.get("publication")
                        if len(parts) >= 2 and re.match(r"^\d{4}\.\d{2}\.\d{2}$", parts[1]):
                            md["date"] = parts[1]
                except Exception:
                    pass
                best_result["doc"]["extraction_run"]["processing_metadata"] = md

            validated_result, warnings = self.validator.validate_and_fix(best_result)
            if warnings:
                logger.warning(f"JSON修复警告: {warnings}")

            # 6. 模板报告生成（保持一致性）
            template_report = self._generate_template_report(validated_result, markdown_content, figures_json, page_count)

            # 7. 保存最终结果（两份JSON）
            schema_path = os.path.join(output_dir, f"{pdf_name}_final_schema.json")
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(validated_result, f, indent=2, ensure_ascii=False)

            template_path = os.path.join(output_dir, f"{pdf_name}_template_report.json")
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template_report, f, indent=2, ensure_ascii=False)

            logger.info(f"{Colors.GREEN}✓ PDF处理完成: {pdf_path}{Colors.RESET}")
            return validated_result

        except Exception as e:
            logger.error(f"{Colors.RED}✗ PDF处理失败 {pdf_path}: {e}{Colors.RESET}")
            traceback.print_exc()

    async def _process_with_single_model(self, markdown_content: str,
                                         figures_json: List[Dict], pdf_name: str) -> Dict:
        """单模型（gemini）处理"""
        results: Dict[str, Any] = {}
        extraction_prompt = self._build_extraction_prompt(markdown_content, figures_json)

        async with OpenRouterProcessor() as processor:
            try:
                result = await self._call_model_with_prompt(
                    processor, "gemini", extraction_prompt, pdf_name
                )
                results["gemini"] = result["result"]
            except Exception as e:
                logger.error(f"gemini模型调用失败: {e}")
                results["gemini"] = None
        return results

    async def _call_model_with_prompt(self, processor: OpenRouterProcessor,
                                      model_key: str, prompt: str, pdf_name: str) -> Dict:
        """调用单个模型（流式优先）"""
        logger.info(f"{Colors.CYAN}调用{model_key}模型...{Colors.RESET}")
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的金融文档数据提取专家。请严格按照提供的JSON schema格式输出结果。输出必须是纯JSON，不包含其他文本。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        try:
            content = await processor.collect_stream_content(model_key, messages, max_tokens=8000)
            json_result = self._extract_json_from_response(content)
            return {
                "model": model_key,
                "result": json_result,
                "raw_response": content,
                "usage": {}
            }
        except Exception:
            response = await processor.call_model(model_key, messages, max_tokens=8000)
            content = response['choices'][0]['message']['content']
            json_result = self._extract_json_from_response(content)
            return {
                "model": model_key,
                "result": json_result,
                "raw_response": content,
                "usage": response.get('usage', {})
            }

    def _build_extraction_prompt(self, markdown_content: str, figures_json: List[Dict]) -> str:
        """构建综合提取提示词：完整markdown + 预提取图表JSON"""
        with open(config.SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_content = f.read()

        figures_block = json.dumps(figures_json, ensure_ascii=False, indent=2)
        prompt = f"""
# 任务：金融文档数据结构化提取（严格JSON Schema）

## 输入材料：
1. Markdown全文（已去除长度限制）：
```markdown
{markdown_content}
```

2. 预提取图表JSON（每图一个对象，供综合）：
```json
{figures_block}
```

## 输出要求：
- 输出必须是一个完整JSON对象，严格符合以下Schema：
```json
{schema_content}
```
- 请综合Markdown文本与图表JSON，填充 figures/tables/numerical_data/claims/relations 等字段。
- figures 节点应整合并规范化预提取结果；如需修正请直接修正为规范格式。
- 所有必需字段必须存在；单位与数值不做换算；确保 page 信息与原始页数一致。

仅输出最终JSON，不要包含多余说明。
"""
        return prompt

    async def _process_figures_in_parallel(self, figure_paths: List[str], markdown_content: str) -> List[Dict]:
        """并行处理每张图像：调用视觉模型提取单图JSON"""
        if not figure_paths:
            return []

        semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
        async with OpenRouterProcessor() as processor:
            tasks = [self._process_single_figure_request(processor, p, semaphore) for p in figure_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        figures: List[Dict] = []
        for p, r in zip(figure_paths, results):
            if isinstance(r, Exception):
                logger.warning(f"图像处理失败: {p} - {r}")
                continue
            try:
                fig_obj = self._extract_json_from_response(r)
            except Exception:
                # 容错：若返回非JSON，构造保底结构
                fig_obj = {
                    "figure_id": hashlib.md5(p.encode()).hexdigest(),
                    "title": "",
                    "page": self._infer_page_from_image_path(p),
                    "type": "other",
                    "series": [],
                    "provenance": {"page": self._infer_page_from_image_path(p)},
                    "image_ref": p
                }
            # 补充保底字段
            fig_obj.setdefault("figure_id", hashlib.md5((p+str(datetime.now().timestamp())).encode()).hexdigest())
            fig_obj.setdefault("provenance", {"page": self._infer_page_from_image_path(p)})
            fig_obj.setdefault("image_ref", p)
            figures.append(fig_obj)
        return figures

    async def _process_single_figure_request(self, processor: OpenRouterProcessor, image_path: str, semaphore: asyncio.Semaphore) -> str:
        """单图像请求：携带图像并要求输出符合 data.figures 项结构的JSON"""
        async with semaphore:
            b64 = self._encode_image_to_base64(image_path)
            page_idx = self._infer_page_from_image_path(image_path)
            system_text = (
                "你将看到一张图像，请仅输出一个JSON对象，结构与 schema中 'data.figures' 的 items 完全一致。"
                "必须包含 figure_id/title/page/type/series/provenance 等字段。不要输出解释文本。"
            )
            user_content = [
                {"type": "input_text", "text": f"请识别此图的结构化数据，并将 page 设为 {page_idx}."},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
            ]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": user_content}
            ]
            # 流式优先，JSON强格式
            try:
                return await processor.collect_stream_content("gemini", messages, max_tokens=1500)
            except Exception:
                resp = await processor.call_model("gemini", messages, max_tokens=1500)
                return resp['choices'][0]['message']['content']

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _infer_page_from_image_path(self, image_path: str) -> int:
        # 约定文件名 images/{page_idx}_{img_idx}.jpg
        try:
            name = Path(image_path).name
            page_str = name.split('_')[0]
            return int(page_str) + 1  # 存储的是0基，输出为1基
        except Exception:
            return 1

    def _count_pages_from_markdown(self, markdown: str) -> int:
        return len(re.findall(r"--- Page \d+ ---", markdown))

    def _generate_template_report(self, final_schema_json: Dict, markdown_content: str,
                                  figures_json: List[Dict], page_count: int) -> Dict:
        """从最终Schema结果与原始内容生成模板报告JSON，保持一致性"""
        # symbol来源：doc.tickers 或 entities
        symbol = "UNKNOWN"
        try:
            tickers = final_schema_json.get("doc", {}).get("tickers", [])
            if isinstance(tickers, list) and tickers:
                symbol = tickers[0]
        except Exception:
            pass

        # 按页拆分markdown
        pages_blocks = re.split(r"\n\n--- Page \d+ ---\n\n", markdown_content)
        pages = []
        for idx in range(page_count):
            md_text = pages_blocks[idx] if idx < len(pages_blocks) else ""
            charts_for_page = [f for f in figures_json if f.get("provenance", {}).get("page") == (idx+1)]
            page_item = {
                "header": {
                    "title": final_schema_json.get("doc", {}).get("title", f"Page {idx+1}"),
                    "rating": "HOLD"
                },
                "main_content": [
                    {"type": "text", "content": md_text}
                ] + [
                    {"type": "chart", "chart_id": f.get("figure_id", f"chart_{idx}_{i}")}
                    for i, f in enumerate(charts_for_page)
                ]
            }
            pages.append(page_item)

        # 构造 charts 列表
        charts = []
        for f in figures_json:
            chart_id = f.get("figure_id")
            chart_type = f.get("type", "other")
            title = f.get("title", "")
            series = f.get("series", [])
            # 转换为模板的 datasets/labels
            labels = []
            datasets = []
            # 简化：若存在第一条序列的 values 与 labels_norm 则使用
            if series:
                # 尝试从 axes.x.labels_norm 获取labels
                axes = f.get("axes", {})
                x = axes.get("x", {}) if isinstance(axes, dict) else {}
                labels = x.get("labels_norm") or x.get("labels_raw") or []
                for s in series:
                    datasets.append({
                        "label": s.get("name", "series"),
                        "values": s.get("values", []),
                        "color": "#007bff"
                    })
            charts.append({
                "id": chart_id,
                "type": chart_type if chart_type in ["bar","line","pie","area","scatter","waterfall","composed","table"] else "bar",
                "title": title,
                "confidence": 0.9,
                "data": (
                    {"columns": labels, "rows": []} if chart_type == "table" else
                    {"labels": labels, "datasets": datasets}
                ),
                "options": {
                    "source": final_schema_json.get("doc", {}).get("source_uri", ""),
                    "height_estimate": 240,
                    "page_break": "avoid"
                }
            })

        return {
            "symbol": symbol,
            "pages": pages,
            "charts": charts
        }

    def _select_best_result(self, results: Dict) -> Dict:
        """选择最佳模型结果（单模型）"""
        return results.get("gemini") or {}

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
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
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
    # 设置环境变量（尊重 .env 已配置的值）
    if torch.version.cuda == '11.8' and not os.environ.get("TRITON_PTXAS_PATH"):
        os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
    os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '0')
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", '0')

    # 运行主程序
    asyncio.run(main())
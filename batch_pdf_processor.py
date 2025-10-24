#!/usr/bin/env python3
"""
批量PDF处理系统 - 针对RTX 4090 48G极速优化
基于DeepSeek OCR + OpenRouter的完整处理流程

功能:
1. 批量PDF → Markdown + 图像提取（极速并发）
2. OpenRouter API调用 (Gemini 2.5 Flash 极速模式)
3. 严格JSON Schema v1.3.1验证
4. 分离输出目录（OCR结果 vs JSON报告）
"""

import os
import sys
import json
import time
import hashlib
import asyncio
import base64
from openai import AsyncOpenAI
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from collections import defaultdict
import copy

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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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
# 使用优化后的 config_batch 配置
try:
    from config_batch import Config as BatchConfig, setup_environment
    setup_environment()
    batch_config = BatchConfig()

    # 创建兼容的配置对象（使用普通类而非dataclass）
    class Config:
        """配置类 - 从 config_batch 加载"""
        # 硬件配置 - RTX 4090 48G 极速优化
        BATCH_SIZE = batch_config.hardware.BATCH_SIZE
        MAX_CONCURRENCY = batch_config.hardware.MAX_CONCURRENCY
        GPU_MEMORY_UTILIZATION = batch_config.hardware.GPU_MEMORY_UTILIZATION
        NUM_WORKERS = batch_config.hardware.NUM_WORKERS

        # OpenRouter API配置
        OPENROUTER_API_KEY = batch_config.api.OPENROUTER_API_KEY
        OPENROUTER_BASE_URL = batch_config.api.OPENROUTER_BASE_URL
        MODELS = batch_config.api.MODELS

        # 新增：批量config引用（用于访问config.api.XXX）
        api = batch_config.api
        hardware = batch_config.hardware
        ocr = batch_config.ocr
        processing = batch_config.processing
        validation = batch_config.validation

        # 文件路径配置
        BASE_DIR = str(batch_config.paths.BASE_DIR)
        INPUT_DIR = str(batch_config.paths.INPUT_DIR)
        OUTPUT_DIR = str(batch_config.paths.OUTPUT_DIR)  # OCR结果（MD+图像）
        OUTPUT_REPORT_DIR = str(batch_config.paths.OUTPUT_REPORT_DIR)  # JSON报告
        TEMP_DIR = str(batch_config.paths.TEMP_DIR)

        # 处理配置
        PDF_DPI = batch_config.ocr.PDF_DPI
        MAX_RETRIES = batch_config.api.MAX_RETRIES
        REQUEST_TIMEOUT = batch_config.api.REQUEST_TIMEOUT

        # 并发配置
        MAX_CONCURRENT_PDFS = batch_config.processing.MAX_CONCURRENT_PDFS
        MAX_CONCURRENT_API_CALLS = batch_config.processing.MAX_CONCURRENT_API_CALLS

        # JSON Schema路径
        SCHEMA_PATH = str(batch_config.paths.SCHEMA_PATH)

        # 验证配置
        STRICT_SCHEMA_VALIDATION = batch_config.validation.STRICT_SCHEMA_VALIDATION

    config = Config()

except ImportError:
    # 降级到默认配置
    print("警告: 无法导入 config_batch，使用默认配置")

    class Config:
        """配置类 - 默认配置"""
        BATCH_SIZE = 12
        MAX_CONCURRENCY = 16
        GPU_MEMORY_UTILIZATION = 0.90
        NUM_WORKERS = 24
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        MODELS = {"gemini": "google/gemini-2.5-flash"}
        BASE_DIR = str(Path(__file__).resolve().parent)
        INPUT_DIR = str(Path(BASE_DIR) / "input_pdfs")
        OUTPUT_DIR = str(Path(BASE_DIR) / "output_results")
        OUTPUT_REPORT_DIR = str(Path(BASE_DIR) / "output_report")
        TEMP_DIR = str(Path(BASE_DIR) / "temp_processing")
        PDF_DPI = 144
        MAX_RETRIES = 5
        REQUEST_TIMEOUT = 600
        MAX_CONCURRENT_PDFS = 6
        MAX_CONCURRENT_API_CALLS = 12
        SCHEMA_PATH = str(Path(BASE_DIR) / "json schema.json")
        STRICT_SCHEMA_VALIDATION = True

        class ProcessingDefaults:
            FIGURE_MAX_DIMENSION = 1024
            FIGURE_ENABLE_DENOISE = True
            FIGURE_JPEG_QUALITY = 70
            FIGURE_WEBP_QUALITY = 60
            FIGURE_TEXT_FALLBACK_MAX_CHARS = 2000

        class APIDefaults:
            LLM_MAX_TOKENS = 8000
            LLM_MAX_TOKENS_IMAGE = 1536

        processing = ProcessingDefaults()
        api = APIDefaults()

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


@dataclass
class PDFProcessingJob:
    pdf_path: str
    pdf_name: str
    pdf_name_clean: str
    date_str: Optional[str]
    publication: str
    ocr_output_dir: str
    json_output_dir: str
    rel_parent: Path
    markdown_content: str
    figure_paths: List[str]


class DeepSeekOCRBatchProcessor:
    """DeepSeek OCR批量处理器"""

    def __init__(self):
        self.model_registry_initialized = False
        self.llm = None
        self._compressed_image_cache: Dict[str, str] = {}
        self._temp_figure_dir = Path(config.TEMP_DIR) / "processed_figures"
        self._temp_figure_dir.mkdir(parents=True, exist_ok=True)
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
        """高质量PDF转图像（支持直接处理PDF）"""
        if dpi is None:
            dpi = config.PDF_DPI

        images = []

        # 如果没有 PyMuPDF，尝试让 vLLM 直接处理 PDF
        if not HAS_PYMUPDF:
            logger.warning("未检测到 PyMuPDF(fitz)，尝试让模型直接处理PDF文件...")
            try:
                # 尝试直接将 PDF 作为图像处理
                # vLLM 的某些版本支持直接处理 PDF
                from PIL import Image as PILImage

                # 尝试用 PIL 打开 PDF（某些情况下可行）
                try:
                    img = PILImage.open(pdf_path)
                    images.append(img)
                    logger.info(f"成功直接加载PDF: {pdf_path}")
                    return images
                except Exception:
                    pass

                # 如果 PIL 失败，尝试使用 pdf2image（如果安装了）
                try:
                    from pdf2image import convert_from_path
                    logger.info("使用 pdf2image 转换PDF...")
                    images = convert_from_path(pdf_path, dpi=dpi)
                    logger.info(f"PDF转换完成: {len(images)}页")
                    return images
                except ImportError:
                    logger.error("pdf2image 未安装。请安装: pip install pdf2image")
                except Exception as e:
                    logger.error(f"pdf2image 转换失败: {e}")

                # 最后的降级方案：跳过此文件
                logger.error(f"无法处理PDF文件 {pdf_path}，需要安装 PyMuPDF 或 pdf2image")
                logger.info("安装方法: pip install PyMuPDF  或  pip install pdf2image")
                raise RuntimeError("PDF处理失败：缺少必要的PDF处理库")

            except Exception as e:
                logger.error(f"PDF处理失败: {e}")
                raise

        # 使用 PyMuPDF 处理
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

    def _preprocess_and_save_figure(self, cropped: Image.Image, page_idx: int,
                                    figure_idx: int, output_dir: str) -> Tuple[str, str]:
        """对裁剪图像进行统一缩放、降噪，并保存压缩版本"""
        processed = cropped.convert("RGB")

        max_dim = getattr(config.processing, "FIGURE_MAX_DIMENSION", 1024)
        resampling_attr = getattr(Image, "Resampling", Image)
        resample_filter = getattr(resampling_attr, "LANCZOS", Image.LANCZOS)
        processed.thumbnail((max_dim, max_dim), resample_filter)

        if getattr(config.processing, "FIGURE_ENABLE_DENOISE", True):
            try:
                processed = processed.filter(ImageFilter.MedianFilter(size=3))
            except Exception as exc:  # pragma: no cover - pillow差异兼容
                logger.debug(f"降噪处理失败，跳过: {exc}")

        temp_dir = self._temp_figure_dir
        temp_dir.mkdir(parents=True, exist_ok=True)

        webp_quality = getattr(config.processing, "FIGURE_WEBP_QUALITY", 60)
        jpeg_quality = getattr(config.processing, "FIGURE_JPEG_QUALITY", 70)

        temp_path = temp_dir / f"{page_idx}_{figure_idx}.webp"
        try:
            processed.save(temp_path, format="WEBP", quality=webp_quality, method=6)
        except Exception:
            processed.save(temp_path, format="WEBP", quality=webp_quality)

        output_path = Path(output_dir) / "images" / f"{page_idx}_{figure_idx}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)

        display_path = str(output_path)
        compressed_path = str(temp_path)
        self._compressed_image_cache[display_path] = compressed_path

        return display_path, compressed_path

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
                            self._preprocess_and_save_figure(cropped, page_idx, img_idx, output_dir)
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

    def get_preprocessed_image_path(self, image_path: str) -> Optional[str]:
        """获取压缩后的图像路径"""
        compressed_path = self._compressed_image_cache.get(image_path)
        if compressed_path and Path(compressed_path).exists():
            return compressed_path
        return None

    def process_pdf(self, pdf_path: str, output_dir: str) -> Tuple[str, List[str]]:
        """处理单个PDF文件"""
        logger.info(f"{Colors.BLUE}开始处理PDF: {pdf_path}{Colors.RESET}")

        try:
            self._compressed_image_cache.clear()
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
                        max_tokens: int = 4000,
                        response_format: Optional[Dict] = None,
                        tools: Optional[List[Dict]] = None,
                        tool_choice: Optional[Any] = None) -> Dict:
        """调用OpenRouter模型 (支持结构化输出和函数调用)"""
        request_payload: Dict[str, Any] = {
            "model": config.MODELS[model_name],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # 确保输出稳定
            "top_p": 0.9
        }

        if response_format is not None:
            request_payload["response_format"] = response_format
        if tools is not None:
            request_payload["tools"] = tools
        if tool_choice is not None:
            request_payload["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**request_payload)
        return response.model_dump()

class JSONSchemaValidator:
    """JSON Schema验证器 - 严格模式"""

    def __init__(self, schema_path: str):
        self.schema = self._load_schema(schema_path)
        self.strict_mode = config.STRICT_SCHEMA_VALIDATION

    def _load_schema(self, schema_path: str) -> Dict:
        """加载JSON Schema"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"{Colors.GREEN}✓ JSON Schema v{schema.get('properties', {}).get('schema_version', {}).get('const', 'unknown')} 加载完成{Colors.RESET}")
            return schema
        except Exception as e:
            logger.error(f"{Colors.RED}✗ JSON Schema加载失败: {e}{Colors.RESET}")
            raise

    def validate(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """严格验证JSON数据"""
        try:
            # 使用 jsonschema 进行严格验证
            validate(instance=data, schema=self.schema)

            # 额外的严格检查
            if self.strict_mode:
                errors = []

                # 检查 schema_version
                if data.get("schema_version") != "1.3.1":
                    errors.append(f"schema_version 必须为 '1.3.1'，当前为 '{data.get('schema_version')}'")

                # 检查必需的顶层字段
                required_fields = ["schema_version", "doc", "passages", "entities", "data"]
                for field in required_fields:
                    if field not in data:
                        errors.append(f"缺少必需字段: {field}")

                # 检查 doc 的必需子字段
                if "doc" in data:
                    doc_required = ["doc_id", "title", "timestamps", "extraction_run"]
                    for field in doc_required:
                        if field not in data["doc"]:
                            errors.append(f"doc 缺少必需字段: {field}")

                if errors:
                    return False, "; ".join(errors)

            return True, None
        except ValidationError as e:
            return False, f"Schema验证失败: {e.message} (路径: {'.'.join(str(p) for p in e.path)})"
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
        self._figure_api_semaphore: Optional[asyncio.Semaphore] = None
        self._setup_directories()

    def _setup_directories(self):
        """设置目录结构 - 包含分离的输出目录"""
        for dir_path in [config.INPUT_DIR, config.OUTPUT_DIR, config.OUTPUT_REPORT_DIR, config.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"确保目录存在: {dir_path}")

    def _get_max_api_concurrency(self) -> int:
        processing_config = getattr(config, "processing", None)
        candidate = None
        if processing_config is not None:
            candidate = getattr(processing_config, "MAX_CONCURRENT_API_CALLS", None)
        if candidate is None:
            candidate = getattr(config, "MAX_CONCURRENT_API_CALLS", getattr(config, "MAX_CONCURRENCY", 1))
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            value = 1
        return max(1, value)

    async def _process_pdf_stage_a(self, pdf_path: str) -> PDFProcessingJob:
        """阶段A：OCR与资源准备，返回后续处理所需的数据"""
        pdf_path_obj = Path(pdf_path).resolve()
        input_dir_obj = Path(config.INPUT_DIR).resolve()

        try:
            rel_path = pdf_path_obj.relative_to(input_dir_obj)
            rel_parent = rel_path.parent
        except ValueError:
            input_root_name = input_dir_obj.name
            rel_parent = Path()
            for parent in pdf_path_obj.parents:
                if parent.name == input_root_name:
                    rel = pdf_path_obj.relative_to(parent)
                    rel_parent = rel.parent
                    break

        pdf_name = pdf_path_obj.stem
        date_str: Optional[str] = None
        publication = str(rel_parent) if rel_parent != Path('.') else "unknown"

        date_match = re.search(r'_(\d{4}-\d{2}-\d{2})$', pdf_name)
        if date_match:
            date_str = date_match.group(1)
            pdf_name_clean = pdf_name[:date_match.start()]
        else:
            pdf_name_clean = pdf_name

        if date_str:
            ocr_output_dir = str(Path(config.OUTPUT_DIR) / date_str / publication / pdf_name_clean)
            json_output_dir = str(Path(config.OUTPUT_REPORT_DIR) / date_str / publication / pdf_name_clean)
        else:
            ocr_output_dir = str(Path(config.OUTPUT_DIR) / rel_parent / pdf_name_clean)
            json_output_dir = str(Path(config.OUTPUT_REPORT_DIR) / rel_parent / pdf_name_clean)

        os.makedirs(ocr_output_dir, exist_ok=True)
        os.makedirs(json_output_dir, exist_ok=True)

        logger.info(f"OCR输出目录: {ocr_output_dir}")
        logger.info(f"JSON输出目录: {json_output_dir}")
        logger.info(f"{Colors.BLUE}阶段A: DeepSeek OCR处理 - {pdf_path_obj.name}{Colors.RESET}")

        markdown_path, figure_paths = self.ocr_processor.process_pdf(pdf_path, ocr_output_dir)
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        logger.info(f"{Colors.GREEN}阶段A完成: {pdf_path_obj.name} (图像数量: {len(figure_paths)}){Colors.RESET}")

        return PDFProcessingJob(
            pdf_path=str(pdf_path_obj),
            pdf_name=pdf_name,
            pdf_name_clean=pdf_name_clean,
            date_str=date_str,
            publication=publication,
            ocr_output_dir=ocr_output_dir,
            json_output_dir=json_output_dir,
            rel_parent=rel_parent,
            markdown_content=markdown_content,
            figure_paths=figure_paths,
        )

    def _merge_model_and_aggregated_results(
        self,
        model_result: Optional[Dict],
        aggregated_result: Optional[Dict]
    ) -> Dict:
        if not aggregated_result and not model_result:
            return {}
        if not aggregated_result:
            return copy.deepcopy(model_result or {})
        if not model_result:
            return copy.deepcopy(aggregated_result)

        merged = copy.deepcopy(aggregated_result)
        for key, value in model_result.items():
            if value in (None, ""):
                continue
            existing = merged.get(key)
            if existing in (None, [], {}):
                merged[key] = copy.deepcopy(value)
                continue
            if isinstance(existing, dict) and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_value in (None, ""):
                        continue
                    if sub_key not in existing or existing[sub_key] in (None, [], {}):
                        existing[sub_key] = copy.deepcopy(sub_value)
        return merged

    async def _process_pdf_stage_b(self, job: PDFProcessingJob) -> Dict:
        """阶段B：执行所有需要API的步骤并输出最终JSON"""
        pdf_name = job.pdf_name
        logger.info(f"{Colors.BLUE}阶段B: API推理与结构化处理 - {pdf_name}{Colors.RESET}")

        page_count = self._count_pages_from_markdown(job.markdown_content)

        logger.info(f"{Colors.BLUE}阶段B-1: 图表识别{Colors.RESET}")
        figures_data = await self._extract_figures_data_parallel(job.figure_paths)
        logger.info(f"{Colors.CYAN}阶段B-1完成: 识别 {len(figures_data)} 张图表{Colors.RESET}")

        logger.info(f"{Colors.BLUE}阶段B-2: 文本与图表联合提取{Colors.RESET}")
        model_results = await self._process_with_single_model_simplified(
            job.markdown_content,
            pdf_name,
            page_count,
            job.date_str or "",
            job.publication,
            figures_data,
        )
        best_model_result = self._select_best_result(model_results)

        markdown_pages = self._split_markdown_pages(job.markdown_content, page_count)
        page_tasks = self._build_page_tasks(markdown_pages, figures_data)
        page_payloads = await self._extract_page_payloads(
            pdf_name,
            page_tasks,
            required_fields=[
                "passages",
                "entities",
                "tables",
                "numerical_data",
                "claims",
                "relations"
            ]
        )

        aggregated_result = self._aggregate_page_results(
            page_payloads,
            figures_data,
            pdf_name,
            page_count,
            job.date_str,
            job.publication,
            job.markdown_content
        )

        best_result = self._merge_model_and_aggregated_results(best_model_result, aggregated_result)
        if not best_result:
            best_result = aggregated_result or best_model_result or {}

        if not best_result:
            raise ValueError(f"{pdf_name} 未生成有效的结构化结果")

        logger.info(f"{Colors.BLUE}阶段B-3: 基础验证与补全{Colors.RESET}")

        if "schema_version" not in best_result:
            best_result["schema_version"] = "1.3.1"

        if "doc" in best_result and "extraction_run" in best_result["doc"]:
            md = best_result["doc"].get("extraction_run", {}).get("processing_metadata", {})
            md.setdefault("pages_processed", page_count)
            md.setdefault("successful_pages", page_count)
            try:
                md["input_relative_path"] = str(job.rel_parent)
                parts = list(job.rel_parent.parts)
                if len(parts) >= 1 and re.match(r"^\d{4}\.\d{2}\.\d{2}$", parts[0]):
                    md["date"] = parts[0]
                    if len(parts) >= 2:
                        md["publication"] = parts[1]
                else:
                    md["publication"] = parts[0] if parts else md.get("publication")
                    if len(parts) >= 2 and re.match(r"^\d{4}\.\d{2}\.\d{2}$", parts[1]):
                        md["date"] = parts[1]
            except Exception:
                pass
            best_result["doc"]["extraction_run"]["processing_metadata"] = md

        is_valid, error_msg = self.validator.validate(best_result)
        if not is_valid:
            logger.warning(f"JSON验证警告: {error_msg}")
            if "doc" not in best_result:
                best_result["doc"] = self._create_minimal_doc(
                    pdf_name,
                    page_count,
                    job.date_str,
                    job.publication
                )
            if "passages" not in best_result:
                best_result["passages"] = []
            if "entities" not in best_result:
                best_result["entities"] = []
            if "data" not in best_result:
                best_result["data"] = self._create_minimal_data()

        output_path = os.path.join(job.json_output_dir, f"{job.pdf_name_clean}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)

        logger.info(f"{Colors.GREEN}✓ 保存JSON: {output_path}{Colors.RESET}")
        logger.info(f"{Colors.GREEN}✓ PDF处理完成: {job.pdf_path}{Colors.RESET}")
        logger.info(f"  - OCR结果: {job.ocr_output_dir}")
        logger.info(f"  - JSON报告: {job.json_output_dir}")

        return best_result

    async def process_single_pdf(self, pdf_path: str) -> Dict:
        job = await self._process_pdf_stage_a(pdf_path)
        return await self._process_pdf_stage_b(job)

    async def _extract_figures_data_parallel(self, figure_paths: List[str]) -> List[Dict]:
        """并行提取所有图表的数据（使用视觉模型识别图表内容）"""
        if not figure_paths:
            return []

        if self._figure_api_semaphore is None:
            self._figure_api_semaphore = asyncio.Semaphore(self._get_max_api_concurrency())

        semaphore = self._figure_api_semaphore

        async with OpenRouterProcessor() as processor:
            tasks = [
                self._extract_single_figure_data(processor, img_path, semaphore)
                for img_path in figure_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        figures_data = []
        for img_path, result in zip(figure_paths, results):
            if isinstance(result, Exception):
                logger.warning(f"图表识别失败 {img_path}: {result}")
                continue
            if result:
                figures_data.append(result)

        return figures_data

    def _split_markdown_pages(self, markdown: str, page_count: int) -> List[str]:
        """根据分页标记拆分Markdown文本"""
        page_pattern = re.compile(r"--- Page (\d+) ---")
        matches = list(page_pattern.finditer(markdown))
        if not matches:
            return [markdown.strip()]

        pages: List[str] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            pages.append(markdown[start:end].strip())

        while len(pages) < page_count:
            pages.append("")

        return pages

    def _build_page_tasks(self, markdown_pages: List[str], figures_data: List[Dict]) -> List[Dict]:
        """构建分页任务，附带该页图表摘要"""
        figures_by_page: Dict[int, List[Dict]] = defaultdict(list)
        for fig in figures_data:
            page_idx = fig.get("page")
            if isinstance(page_idx, int):
                figures_by_page[page_idx].append(fig)

        tasks: List[Dict] = []
        for idx, page_text in enumerate(markdown_pages):
            tasks.append({
                "page": idx + 1,
                "markdown": page_text,
                "figures": figures_by_page.get(idx + 1, [])
            })
        return tasks

    def _summarize_figures_for_page(self, figures: List[Dict]) -> str:
        if not figures:
            return "无"
        lines = []
        for fig in figures:
            fig_type = fig.get("type", "unknown")
            title = fig.get("title") or fig.get("description") or "无标题"
            lines.append(f"- {fig_type}: {title[:80]}")
        return "\n".join(lines)

    async def _extract_page_payloads(
        self,
        pdf_name: str,
        page_tasks: List[Dict],
        required_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """分页调用LLM，仅发送单页Markdown获取结构化片段"""
        if not page_tasks:
            return []

        fields = required_fields or [
            "passages", "entities", "tables", "numerical_data", "claims", "relations"
        ]

        semaphore = asyncio.Semaphore(
            max(1, min(getattr(config, "MAX_CONCURRENT_API_CALLS", config.MAX_CONCURRENCY), len(page_tasks)))
        )

        async with OpenRouterProcessor() as processor:
            tasks = [
                self._call_single_page_extraction(processor, pdf_name, task, fields, semaphore)
                for task in page_tasks
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        payloads: List[Dict] = []
        for task, response in zip(page_tasks, responses):
            if isinstance(response, Exception):
                logger.error(f"分页提取失败: 第{task.get('page')}页 - {response}")
                payloads.append(self._build_fallback_page_payload(task))
                continue

            if not isinstance(response, dict) or not response:
                payloads.append(self._build_fallback_page_payload(task))
                continue

            response.setdefault("page", task.get("page", 1))
            for field in fields:
                response.setdefault(field, [])
            response.setdefault("doc_metadata", {})
            payloads.append(response)

        return payloads

    async def _call_single_page_extraction(
        self,
        processor: OpenRouterProcessor,
        pdf_name: str,
        task: Dict,
        required_fields: List[str],
        semaphore: asyncio.Semaphore
    ) -> Dict:
        async with semaphore:
            page_no = task.get("page", 1)
            markdown = task.get("markdown", "")
            figure_summary = self._summarize_figures_for_page(task.get("figures", []))
            fields_text = ", ".join(required_fields)

            prompt = f"""
你是金融文档结构化抽取助手。请仅基于下方Markdown提取第 {page_no} 页的结构化信息。

文档名称: {pdf_name}
页码: {page_no}

## 本页Markdown
```markdown
{markdown}
```

## 本页关联图表
{figure_summary}

## 输出要求
- 返回一个JSON对象，必须包含字段: page, {fields_text}, doc_metadata
- page: 当前页码（整数）
- passages: 数组。每个元素需提供 text，允许包含 section、labels、entities（实体名称列表）、passage_index（整数，用于跨字段引用）
- entities: 数组。每个元素包含 name（必填）及可选 type、ticker、aliases、country 等
- tables: 数组。包含 title、headers（或 columns）、rows，以及可选的描述信息
- numerical_data: 数组。每个元素包含 context、value（或 value_text）、unit、metric_type，可选 passage_index、entity（名称）
- claims: 数组。包含 text、label（guidance_up/guidance_down/risk/outlook/strategy/other之一），可提供 sentiment、passage_index、related_entities
- relations: 数组。包含 subject、predicate、object，可附带 provenance（如 passage_index）
- doc_metadata: 可选字典，仅在本页包含文档级信息（如标题、tickers、report_type）时填写

仅输出JSON，不要额外解释。确保所有内容仅来源于本页。
"""

            messages = [
                {
                    "role": "system",
                    "content": "你是一名擅长结构化金融报告的分析师，只输出JSON，不提供任何解释。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = await processor.call_model(
                "gemini",
                messages,
                max_tokens=getattr(config.api, "LLM_MAX_TOKENS", 6000)
            )
            content = response['choices'][0]['message']['content']
            parsed = self._extract_json_from_response(content)
            if not isinstance(parsed, dict):
                return {}
            return parsed

    def _build_fallback_page_payload(self, task: Dict) -> Dict:
        text = (task.get("markdown") or "").strip()
        passages = []
        if text:
            passages.append({"text": text, "section": "auto"})
        return {
            "page": task.get("page", 1),
            "passages": passages,
            "entities": [],
            "tables": [],
            "numerical_data": [],
            "claims": [],
            "relations": [],
            "doc_metadata": {}
        }

    def _build_figure_request_payload(self, page_idx: int, *, prompt_body: str,
                                      b64: Optional[str], text_override: Optional[str]) -> List[Dict[str, Any]]:
        """构造视觉模型的多模态请求载荷，支持文本兜底"""
        payload: List[Dict[str, Any]] = [{"type": "text", "text": prompt_body}]

        max_chars = getattr(config.processing, "FIGURE_TEXT_FALLBACK_MAX_CHARS", 2000)
        if text_override:
            sanitized = text_override.strip()
            if len(sanitized) > max_chars:
                sanitized = sanitized[:max_chars]
            payload.append({
                "type": "text",
                "text": f"图像预处理关键点摘要（第{page_idx}页）:\n{sanitized}"
            })
        elif b64:
            payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        else:
            raise ValueError("缺少图像数据或文本替代内容，无法构造请求")

        return payload

    async def _extract_single_figure_data(self, processor: OpenRouterProcessor,
                                         image_path: str, semaphore: asyncio.Semaphore) -> Dict:
        """提取单个图表的数据（视觉识别）"""
        async with semaphore:
            try:
                b64, text_override = self._encode_image_to_base64(image_path)
                page_idx = self._infer_page_from_image_path(image_path)

                messages = [
                    {
                        "role": "system",
                        "content": "你是专业的图表数据提取专家。请识别图表类型（柱状图/折线图/饼图/表格等），并提取其中的所有数据。"
                    },
                    {
                        "role": "user",
                        "content": self._build_figure_request_payload(
                            page_idx,
                            prompt_body=f"""请分析这张图表（第{page_idx}页），提取以下信息并以JSON格式输出：

1. 图表类型（type）：bar/line/pie/area/scatter/heatmap/waterfall/combo/other
2. 图表标题（title）
3. 坐标轴信息（axes）：
   - x轴：类型、标签
   - y轴：单位、范围
4. 数据系列（series）：每个系列包含name、unit、values数组

输出格式示例：
{{
  "type": "bar",
  "title": "Revenue Growth",
  "page": {page_idx},
  "axes": {{
    "x": {{"type": "category", "labels": ["Q1", "Q2", "Q3", "Q4"]}},
    "y": {{"unit": "USD million", "range": {{"min": 0, "max": 100}}}}
  }},
  "series": [
    {{"name": "Revenue", "unit": "USD million", "values": [20, 30, 45, 60]}}
  ]
}}

仅输出JSON，不要其他文字。""",
                            b64=b64,
                            text_override=text_override
                        )
                    }
                ]

                resp = await processor.call_model(
                    "gemini",
                    messages,
                    max_tokens=getattr(config.api, "LLM_MAX_TOKENS_IMAGE", 1536)
                )
                content = resp['choices'][0]['message']['content']

                # 提取JSON
                figure_data = self._extract_json_from_response(content)
                if figure_data:
                    # 确保包含必需字段
                    figure_data["page"] = page_idx
                    figure_data["figure_id"] = hashlib.md5(
                        f"{image_path}_{page_idx}".encode()
                    ).hexdigest()[:16]
                    # 添加provenance（必需字段）
                    if "provenance" not in figure_data:
                        figure_data["provenance"] = {"page": page_idx}
                    return figure_data

            except Exception as e:
                logger.error(f"提取图表数据失败 {image_path}: {e}")
                return None

    async def _process_with_single_model_simplified(self, markdown_content: str,
                                                     pdf_name: str, page_count: int,
                                                     date_str: str, publication: str,
                                                     figures_data: List[Dict]) -> Dict:
        """简化的单模型处理（整合文本和图表数据）"""
        results: Dict[str, Any] = {}
        extraction_prompt = self._build_simplified_extraction_prompt(
            markdown_content, pdf_name, page_count, date_str, publication, figures_data
        )

        async with OpenRouterProcessor() as processor:
            try:
                result = await self._call_model_with_prompt(
                    processor, "gemini", extraction_prompt, pdf_name
                )
                # 将识别的图表数据整合到结果中
                if result["result"]:
                    if "data" not in result["result"]:
                        result["result"]["data"] = {}
                    # 直接使用视觉识别的图表数据
                    result["result"]["data"]["figures"] = figures_data
                results["gemini"] = result["result"]
            except Exception as e:
                logger.error(f"gemini模型调用失败: {e}")
                results["gemini"] = {}
        return results

    def _aggregate_page_results(
        self,
        page_payloads: List[Dict],
        figures_data: List[Dict],
        pdf_name: str,
        page_count: int,
        date_str: Optional[str],
        publication: str,
        markdown_content: str
    ) -> Dict:
        """聚合页级结果，补齐Schema要求的全局结构"""
        final_result: Dict[str, Any] = {
            "schema_version": "1.3.1",
            "doc": self._create_minimal_doc(pdf_name, page_count, date_str, publication),
            "passages": [],
            "entities": [],
            "data": self._create_minimal_data()
        }

        final_result["data"]["figures"] = figures_data or []

        # 解析文档级元数据（先从整体Markdown，再融合页级补充）
        doc_metadata = self._extract_doc_metadata(markdown_content, pdf_name, date_str)
        for payload in page_payloads:
            meta = payload.get("doc_metadata")
            if isinstance(meta, dict):
                doc_metadata = self._merge_doc_metadata(doc_metadata, meta)

        doc_obj = final_result["doc"]
        for key, value in doc_metadata.items():
            if key in {"title", "report_type", "sector", "language", "source_uri"} and value:
                doc_obj[key] = value
            elif key == "report_date" and value:
                doc_obj["report_date"] = value
            elif key == "word_count" and isinstance(value, int):
                doc_obj["word_count"] = value
            elif key == "full_text" and value:
                doc_obj["full_text"] = value
            elif key == "tickers":
                tickers = sorted({t.strip().upper() for t in value if isinstance(t, str) and t.strip()})
                if tickers:
                    doc_obj["tickers"] = tickers
                    doc_obj["symbols"] = tickers

        # 构建实体索引
        entity_map: Dict[str, Dict] = {}
        alias_index: Dict[str, str] = {}
        doc_tickers = set(doc_obj.get("tickers", []))

        for payload in page_payloads:
            for entity in payload.get("entities", []):
                normalized_name = self._normalize_entity_name(entity.get("name"))
                if not normalized_name:
                    continue

                key = normalized_name.lower()
                existing = entity_map.get(key)
                if not existing:
                    entity_id = entity.get("entity_id") or hashlib.md5(
                        f"{normalized_name}|{publication}".encode()
                    ).hexdigest()[:16]
                    entity_obj = {
                        "entity_id": entity_id,
                        "name": normalized_name
                    }
                    for optional in ["type", "ticker", "isin", "lei", "country"]:
                        if entity.get(optional):
                            entity_obj[optional] = entity[optional]
                    aliases = self._ensure_unique_list(entity.get("aliases"))
                    if aliases:
                        entity_obj["aliases"] = aliases

                    entity_map[key] = entity_obj
                    final_result["entities"].append(entity_obj)
                    self._register_entity_aliases(alias_index, entity_obj)
                else:
                    for optional in ["type", "ticker", "isin", "lei", "country"]:
                        if optional not in existing and entity.get(optional):
                            existing[optional] = entity[optional]
                    aliases = self._ensure_unique_list(entity.get("aliases"))
                    if aliases:
                        existing.setdefault("aliases", [])
                        for alias in aliases:
                            if alias not in existing["aliases"]:
                                existing["aliases"].append(alias)
                    self._register_entity_aliases(alias_index, existing)

                ticker_value = entity.get("ticker")
                if ticker_value:
                    doc_tickers.add(ticker_value.upper())

        if doc_tickers:
            tickers_sorted = sorted(doc_tickers)
            doc_obj["tickers"] = tickers_sorted
            doc_obj["symbols"] = tickers_sorted

        # 处理页级片段
        passage_index_map: Dict[int, Dict[int, str]] = defaultdict(dict)
        figures_index = {fig.get("figure_id"): fig for fig in final_result["data"].get("figures", []) if fig.get("figure_id")}
        table_ids: set = set()
        num_seen_keys: set = set()
        num_id_set: set = set()
        claim_labels = {"guidance_up", "guidance_down", "risk", "outlook", "strategy", "other"}
        sentiment_labels = {"pos", "neg", "neu"}

        for payload in page_payloads:
            page_no = payload.get("page") or 1
            passages = payload.get("passages", [])

            for idx, passage in enumerate(passages):
                text = passage.get("text") or passage.get("content")
                if not text:
                    continue

                passage_id = passage.get("passage_id") or hashlib.md5(
                    f"{pdf_name}|{page_no}|{idx}|{text}".encode()
                ).hexdigest()[:16]

                passage_obj: Dict[str, Any] = {
                    "passage_id": passage_id,
                    "page": page_no,
                    "text": text.strip()
                }

                if passage.get("section"):
                    passage_obj["section"] = passage["section"]

                labels = self._ensure_unique_list(passage.get("labels"))
                if labels:
                    passage_obj["labels"] = labels

                entity_refs = []
                for entity_ref in passage.get("entities", []):
                    resolved = self._resolve_entity_reference(entity_ref, entity_map, alias_index)
                    if resolved:
                        entity_refs.append(resolved)
                if entity_refs:
                    passage_obj["entities"] = entity_refs

                final_result["passages"].append(passage_obj)

                passage_index = passage.get("passage_index")
                if isinstance(passage_index, int):
                    passage_index_map[page_no][passage_index] = passage_id
                passage_index_map[page_no][idx] = passage_id

            # 表格
            for idx, table in enumerate(payload.get("tables", [])):
                headers = table.get("headers") or table.get("columns") or []
                rows = table.get("rows") or table.get("data") or []
                if not headers and not rows:
                    continue

                table_id = table.get("table_id") or hashlib.md5(
                    f"{pdf_name}|table|{page_no}|{idx}".encode()
                ).hexdigest()[:16]

                if table_id in table_ids:
                    continue
                table_ids.add(table_id)

                table_obj: Dict[str, Any] = {
                    "table_id": table_id,
                    "title": table.get("title") or f"Table {page_no}-{idx+1}",
                    "page": page_no,
                    "headers": headers,
                    "rows": rows,
                    "provenance": table.get("provenance") or {"page": page_no}
                }
                if table.get("description"):
                    table_obj["description"] = table["description"]

                final_result["data"]["tables"].append(table_obj)

            # 数值数据
            for idx, num in enumerate(payload.get("numerical_data", [])):
                context = num.get("context") or num.get("label")
                numeric_value, raw_text, is_percentage = self._coerce_to_number(
                    num.get("value"), num.get("value_text")
                )
                if context is None or numeric_value is None:
                    continue

                unit = num.get("unit") or self._infer_unit_from_value_text(raw_text, is_percentage)
                metric_type = self._infer_metric_type(
                    num.get("metric_type"), unit, context, raw_text, is_percentage
                )

                key = (context, numeric_value, unit, page_no)
                if key in num_seen_keys:
                    continue
                num_seen_keys.add(key)

                num_id = num.get("num_id") or hashlib.md5(
                    f"{pdf_name}|num|{page_no}|{context}|{numeric_value}".encode()
                ).hexdigest()[:16]

                provenance = num.get("provenance") or {"page": page_no}
                provenance.setdefault("page", page_no)

                passage_ref = num.get("passage_index")
                passage_id = None
                if isinstance(passage_ref, int):
                    passage_id = self._lookup_passage_id(passage_index_map, page_no, passage_ref)
                if not passage_id and passage_index_map.get(page_no):
                    passage_id = next(iter(passage_index_map[page_no].values()), None)
                if passage_id:
                    provenance.setdefault("passage_id", passage_id)

                num_obj: Dict[str, Any] = {
                    "num_id": num_id,
                    "context": context,
                    "metric_type": metric_type,
                    "unit": unit or "unitless",
                    "value": numeric_value,
                    "value_text": raw_text or str(num.get("value")),
                    "provenance": provenance
                }

                if num.get("scale"):
                    num_obj["scale"] = num["scale"]
                if num.get("multiplier") is not None:
                    num_obj["multiplier"] = num["multiplier"]
                if num.get("rounding") is not None:
                    num_obj["rounding"] = num["rounding"]
                if num.get("as_of"):
                    num_obj["as_of"] = num["as_of"]
                if num.get("fiscal_period"):
                    num_obj["fiscal_period"] = num["fiscal_period"]

                entity_ref = self._resolve_entity_reference(
                    num.get("entity") or num.get("entity_name") or num.get("entity_id"),
                    entity_map,
                    alias_index
                )
                if entity_ref:
                    num_obj["entity_id"] = entity_ref

                final_result["data"]["numerical_data"].append(num_obj)
                num_id_set.add(num_id)

            # 主张（claims）
            for idx, claim in enumerate(payload.get("claims", [])):
                text = claim.get("text") or claim.get("claim")
                if not text:
                    continue

                passage_ref = claim.get("passage_id")
                if not passage_ref:
                    passage_index = claim.get("passage_index")
                    passage_ref = self._lookup_passage_id(passage_index_map, page_no, passage_index) if isinstance(passage_index, int) else None
                    if not passage_ref and passage_index_map.get(page_no):
                        passage_ref = next(iter(passage_index_map[page_no].values()), None)
                if not passage_ref:
                    continue

                claim_id = claim.get("claim_id") or hashlib.md5(
                    f"{pdf_name}|claim|{page_no}|{idx}|{text}".encode()
                ).hexdigest()[:16]

                label = claim.get("label") or "other"
                if label not in claim_labels:
                    label = "other"

                claim_obj: Dict[str, Any] = {
                    "claim_id": claim_id,
                    "text": text.strip(),
                    "label": label,
                    "passage_id": passage_ref
                }

                sentiment = claim.get("sentiment")
                if sentiment in sentiment_labels:
                    claim_obj["sentiment"] = sentiment

                evidence = claim.get("evidence")
                if isinstance(evidence, dict):
                    cleaned_evidence: Dict[str, List[str]] = {}
                    if evidence.get("figure_ids"):
                        cleaned_evidence["figure_ids"] = [fid for fid in evidence["figure_ids"] if fid in figures_index]
                    if evidence.get("table_ids"):
                        cleaned_evidence["table_ids"] = [tid for tid in evidence["table_ids"] if tid in table_ids]
                    if evidence.get("num_ids"):
                        cleaned_evidence["num_ids"] = [nid for nid in evidence["num_ids"] if nid in num_id_set]
                    if cleaned_evidence:
                        claim_obj["evidence"] = cleaned_evidence

                final_result["data"]["claims"].append(claim_obj)

            # 关系（relations）
            for idx, relation in enumerate(payload.get("relations", [])):
                subject = relation.get("subject")
                predicate = relation.get("predicate")
                obj = relation.get("object")
                if not (subject and predicate and obj):
                    continue

                rel_id = relation.get("rel_id") or hashlib.md5(
                    f"{pdf_name}|rel|{page_no}|{idx}|{subject}|{predicate}|{obj}".encode()
                ).hexdigest()[:16]

                relation_obj: Dict[str, Any] = {
                    "rel_id": rel_id,
                    "subject": self._resolve_relation_end(subject, entity_map, alias_index),
                    "predicate": predicate,
                    "object": self._resolve_relation_end(obj, entity_map, alias_index)
                }

                provenance = relation.get("provenance") or {}
                passage_index = relation.get("passage_index")
                if isinstance(passage_index, int):
                    passage_ref = self._lookup_passage_id(passage_index_map, page_no, passage_index)
                    if passage_ref:
                        provenance = dict(provenance)
                        provenance["passage_id"] = passage_ref
                if provenance:
                    provenance.setdefault("page", page_no)
                    relation_obj["provenance"] = provenance
                else:
                    relation_obj["provenance"] = {"page": page_no}

                final_result["data"]["relations"].append(relation_obj)

        # 汇总信息
        final_result["data"]["extraction_summary"]["figures_count"] = len(final_result["data"]["figures"])
        final_result["data"]["extraction_summary"]["tables_count"] = len(final_result["data"]["tables"])
        final_result["data"]["extraction_summary"]["numerical_data_count"] = len(final_result["data"]["numerical_data"])

        # 更新处理元数据
        extraction_run = final_result["doc"].setdefault("extraction_run", {})
        extraction_run["pipeline_steps"] = ["ocr", "page_llm_extraction", "aggregation"]
        extraction_run["synthesis_model"] = config.MODELS.get("gemini", "google/gemini-2.5-flash")

        metadata = extraction_run.setdefault("processing_metadata", {})
        metadata.update({
            "pages_processed": page_count,
            "successful_pages": sum(1 for payload in page_payloads if payload.get("passages")),
            "page_tasks": len(page_payloads),
            "figure_count": len(final_result["data"]["figures"]),
            "page_level_strategy": "per_page_llm",
            "failed_pages": sum(1 for payload in page_payloads if not payload.get("passages")),
            "aggregation_timestamp": datetime.now().isoformat()
        })

        return final_result

    def _create_minimal_doc(self, pdf_name: str, page_count: int,
                           date_str: str, publication: str) -> Dict:
        """创建最小doc结构"""
        return {
            "doc_id": hashlib.md5(pdf_name.encode()).hexdigest(),
            "title": pdf_name,
            "source_uri": f"{publication}/{pdf_name}",
            "language": "en",
            "timestamps": {
                "ingested_at": datetime.now().isoformat(),
                "extracted_at": datetime.now().isoformat()
            },
            "extraction_run": {
                "vision_model": "deepseek-ai/DeepSeek-OCR",
                "synthesis_model": "google/gemini-2.5-flash",
                "pipeline_steps": ["ocr", "llm_extraction"],
                "processing_metadata": {
                    "pages_processed": page_count,
                    "successful_pages": page_count,
                    "date": date_str,
                    "publication": publication
                }
            }
        }

    def _create_minimal_data(self) -> Dict:
        """创建最小data结构"""
        return {
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

    def _merge_doc_metadata(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base) if isinstance(base, dict) else {}
        if not isinstance(update, dict):
            return merged
        for key, value in update.items():
            if value in (None, "", [], {}):
                continue
            if key in {"tickers", "symbols"}:
                merged.setdefault("tickers", [])
                if isinstance(value, list):
                    merged["tickers"].extend(value)
                else:
                    merged["tickers"].append(value)
            else:
                merged[key] = value
        return merged

    def _normalize_entity_name(self, name: Optional[str]) -> Optional[str]:
        if not name or not isinstance(name, str):
            return None
        return re.sub(r"\s+", " ", name).strip()

    def _ensure_unique_list(self, items: Any) -> List[str]:
        if not items:
            return []
        if not isinstance(items, list):
            items = [items]
        seen = set()
        unique: List[str] = []
        for item in items:
            if item is None:
                continue
            value = item.strip() if isinstance(item, str) else str(item)
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        return unique

    def _register_entity_aliases(self, alias_index: Dict[str, str], entity: Dict[str, Any]) -> None:
        entity_id = entity.get("entity_id")
        if not entity_id:
            return
        name = entity.get("name")
        if isinstance(name, str) and name.strip():
            alias_index[name.strip().lower()] = entity_id
        ticker = entity.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            alias_index[ticker.strip().lower()] = entity_id
        for alias in entity.get("aliases", []) or []:
            if isinstance(alias, str) and alias.strip():
                alias_index[alias.strip().lower()] = entity_id

    def _resolve_entity_reference(
        self,
        reference: Any,
        entity_map: Dict[str, Dict],
        alias_index: Dict[str, str]
    ) -> Optional[str]:
        if not reference:
            return None
        if isinstance(reference, dict):
            if reference.get("entity_id"):
                return reference["entity_id"]
            name = reference.get("name") or reference.get("text") or reference.get("subject") or reference.get("object")
        else:
            name = str(reference)

        if not name:
            return None
        key = name.strip().lower()
        if key in entity_map:
            return entity_map[key]["entity_id"]
        if key in alias_index:
            return alias_index[key]
        return None

    def _lookup_passage_id(
        self,
        passage_index_map: Dict[int, Dict[int, str]],
        page_no: int,
        passage_index: Optional[int]
    ) -> Optional[str]:
        if passage_index is None:
            return None
        page_map = passage_index_map.get(page_no)
        if not page_map:
            return None
        for candidate in [passage_index, passage_index - 1, passage_index + 1]:
            if isinstance(candidate, int) and candidate in page_map:
                return page_map[candidate]
        return None

    def _coerce_to_number(self, value: Any, value_text: Optional[str]) -> Tuple[Optional[float], Optional[str], bool]:
        if isinstance(value, (int, float)):
            return float(value), value_text if value_text is not None else str(value), False

        candidate_text = None
        if isinstance(value, str) and value.strip():
            candidate_text = value.strip()
        elif isinstance(value_text, str) and value_text.strip():
            candidate_text = value_text.strip()
        elif value is not None:
            candidate_text = str(value)

        if not candidate_text:
            return None, value_text, False

        text = candidate_text.strip()
        negative = False
        if text.startswith("(") and text.endswith(")"):
            negative = True
            text = text[1:-1]

        is_percentage = "%" in text or "％" in text
        cleaned = re.sub(r"[^0-9\.\-]", "", text)
        if cleaned.count('-') > 1:
            cleaned = cleaned.replace('-', '')
            cleaned = '-' + cleaned

        if not cleaned or cleaned in {"-", "."}:
            return None, candidate_text, is_percentage

        try:
            number = float(cleaned)
            if negative and number > 0:
                number = -number
            if is_percentage and abs(number) > 1.5:
                number = number / 100.0
            return number, candidate_text, is_percentage
        except ValueError:
            return None, candidate_text, is_percentage

    def _infer_unit_from_value_text(self, value_text: Optional[str], is_percentage: bool) -> str:
        if is_percentage:
            return "%"
        if not value_text:
            return "unitless"
        lower = value_text.lower()
        if "$" in value_text or "usd" in lower:
            return "USD"
        if "eur" in lower or "€" in value_text:
            return "EUR"
        if "gbp" in lower or "£" in value_text:
            return "GBP"
        if "cny" in lower or "rmb" in lower or "¥" in value_text or "元" in value_text:
            return "CNY"
        if "jpy" in lower:
            return "JPY"
        return "unitless"

    def _infer_metric_type(
        self,
        metric_type: Optional[str],
        unit: Optional[str],
        context: str,
        value_text: Optional[str],
        is_percentage: bool
    ) -> str:
        allowed = {"currency", "percentage", "basis_points", "multiple", "count", "ratio", "per_share", "duration", "other"}
        if metric_type in allowed:
            return metric_type

        text = f"{unit or ''} {context or ''} {value_text or ''}".lower()
        if is_percentage or "%" in (unit or "") or "percent" in text or "margin" in text or "growth" in text or "同比" in text or "环比" in text:
            return "percentage"
        if any(token in text for token in ["$", "usd", "eur", "¥", "cny", "rmb", "million", "billion", "千", "亿"]):
            return "currency"
        if "basis point" in text or "bp" in text:
            return "basis_points"
        if "per share" in text or "/share" in text or "每股" in text:
            return "per_share"
        if "ratio" in text or "multiple" in text or "倍" in text:
            return "ratio"
        if any(word in text for word in ["unit", "units", "shipments", "customers", "stores", "employees", "people", "台", "辆", "份"]):
            return "count"
        if any(word in text for word in ["year", "quarter", "month", "week", "day", "hour", "个月", "季度"]):
            return "duration"
        return "other"

    def _resolve_relation_end(self, value: Any, entity_map: Dict[str, Dict], alias_index: Dict[str, str]) -> str:
        entity_id = self._resolve_entity_reference(value, entity_map, alias_index)
        if entity_id:
            return entity_id
        if isinstance(value, dict):
            for key in ["name", "text", "subject", "object"]:
                if value.get(key):
                    return str(value[key])
        return str(value) if value is not None else ""

    def _detect_language(self, text: str) -> str:
        if not text:
            return "en"
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_chars = len(re.findall(r"[A-Za-z]", text))
        return "zh" if chinese_chars > latin_chars else "en"

    def _extract_doc_metadata(self, markdown_content: str, pdf_name: str, date_str: Optional[str]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        title_match = re.search(r"^#\s*(.+)", markdown_content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            first_line = next((line.strip() for line in markdown_content.splitlines() if line.strip()), pdf_name)
            metadata["title"] = first_line[:200]

        if date_str:
            metadata["report_date"] = date_str

        metadata["language"] = self._detect_language(markdown_content)
        metadata["word_count"] = len(re.findall(r"\w+", markdown_content))
        metadata["full_text"] = markdown_content

        ticker_candidates: set = set()
        ticker_patterns = [
            r"Ticker[s]?:\s*([A-Z0-9\-\s,;]+)",
            r"股票代码[:：]\s*([A-Z0-9\-\s,;]+)"
        ]
        for pattern in ticker_patterns:
            for match in re.finditer(pattern, markdown_content):
                raw = match.group(1)
                parts = re.split(r"[,;\s]+", raw)
                for part in parts:
                    ticker = part.strip().upper()
                    if ticker and 1 <= len(ticker) <= 6:
                        ticker_candidates.add(ticker)
        if ticker_candidates:
            metadata["tickers"] = sorted(ticker_candidates)

        return metadata

    async def _call_model_with_prompt(self, processor: OpenRouterProcessor,
                                      model_key: str, prompt: str, pdf_name: str) -> Dict:
        """调用单个模型（简化版，最大上下文）"""
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
        # Gemini 2.5 Flash 支持最大 1M tokens 上下文
        # 使用配置的max_tokens而非硬编码（提升响应速度）
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "financial_report_schema_v1_3_1",
                "schema": self.validator.schema
            }
        }
        response = await processor.call_model(
            model_key,
            messages,
            max_tokens=config.api.LLM_MAX_TOKENS,  # 使用配置的16000而非65536
            response_format=response_format
        )
        json_result = self._extract_json_from_response(response)
        message = (response.get('choices') or [{}])[0].get('message', {})
        raw_content = message.get('content')
        if isinstance(raw_content, list):
            raw_content = "\n".join(
                part.get('text', '')
                for part in raw_content
                if isinstance(part, dict) and part.get('type') == 'text'
            ).strip()
        if not raw_content and message.get('parsed') is not None:
            try:
                raw_content = json.dumps(message['parsed'], ensure_ascii=False)
            except TypeError:
                raw_content = str(message['parsed'])
        return {
            "model": model_key,
            "result": json_result,
            "raw_response": raw_content or "",
            "usage": response.get('usage', {})
        }

    def _build_simplified_extraction_prompt(self, markdown_content: str, pdf_name: str,
                                            page_count: int, date_str: str, publication: str,
                                            figures_data: List[Dict]) -> str:
        """构建简化的提取提示词（整合文本和图表数据）"""
        # 构建图表数据摘要
        figures_summary = "\n".join([
            f"- 页{fig.get('page', '?')}: {fig.get('type', 'unknown')}图 - {fig.get('title', '无标题')} ({len(fig.get('series', []))}个数据系列)"
            for fig in figures_data
        ]) if figures_data else "无图表"

        prompt = f"""
# 任务：金融报告数据结构化提取（JSON Schema v1.3.1）

## 文档信息
- 文件名：{pdf_name}
- 页数：{page_count}
- 日期：{date_str}
- 来源：{publication}

## OCR提取的完整文本
```markdown
{markdown_content}
```

## 已识别的图表数据（共{len(figures_data)}张，数据已提取）
{figures_summary}

**注意：图表数据（figures）已经通过视觉模型提取完成，你只需要提取文本中的其他数据。**

## 输出要求

**仅输出一个完整的JSON对象，不要包含任何解释文本、markdown标记或其他内容。**

### 关键字段要求
- 顶层字段必须包含：`schema_version`、`doc`、`passages`、`entities`、`data`
- `schema_version` 固定为 "1.3.1"
- `doc` 至少包含 `doc_id`、`title`、`timestamps`、`extraction_run`
- `data` 内需提供 `figures`、`tables`、`numerical_data`、`claims`、`relations`、`extraction_summary`
- `extraction_summary` 中补充 `figures_count`、`tables_count`、`numerical_data_count`

### 关键注意事项
- `fiscal_period` 使用完整格式：例如 `FY2024Q4`、`CY2025H1`
- 百分比转换为 0-1 的小数（如 18.2% → 0.182）
- 所有 `page` 字段填写 1 到 {page_count} 的整数
- ID 需稳定（hash 或 UUID），时间统一为 ISO 8601
- 优先提取 `numerical_data`、`tables` 和 `claims`，并补充必要的 `passages` 与 `entities`
"""
        return prompt

    def _build_extraction_prompt(self, markdown_content: str, figures_json: List[Dict]) -> str:
        """构建综合提取提示词：完整markdown + 预提取图表JSON（Schema v1.3.1）【已废弃】"""
        with open(config.SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_content = f.read()

        figures_block = json.dumps(figures_json, ensure_ascii=False, indent=2)
        prompt = f"""
# 任务：金融报告数据结构化提取（JSON Schema v1.3.1）

## 输入材料：
1. **Markdown全文**（OCR提取的完整文本）：
```markdown
{markdown_content}
```

2. **预提取图表数据**（视觉模型已识别的图表）：
```json
{figures_block}
```

## 输出要求：

### 1. 严格遵循Schema v1.3.1
输出必须是一个完整的JSON对象，严格符合以下Schema：
```json
{schema_content}
```

### 2. 必需的顶层字段
- `schema_version`: 必须为 "1.3.1"
- `doc`: 文档元数据（doc_id, title, timestamps, extraction_run等）
- `passages`: 可检索的文本片段数组（用于RAG）
- `entities`: 标准化实体数组（公司/指数/政府/产品等）
- `data`: 核心数据对象，包含：
  - `figures`: 图表数据（整合预提取结果）
  - `tables`: 表格数据
  - `numerical_data`: 原子数值事实
  - `claims`: 文本性主张/结论
  - `relations`: 三元组关系
  - `extraction_summary`: 提取统计摘要

### 3. 数据提取指南

**figures（图表）：**
- 整合预提取的图表JSON，规范化为标准格式
- 必需字段：figure_id, title, page, type, series, provenance
- type枚举：bar, line, area, pie, scatter, heatmap, waterfall, combo, other
- series必须包含：name, unit, values（数组，缺失用null）
- 坐标轴信息：axes.x（时间/类别/数值），axes.y_left/y_right（单位/范围）

**tables（表格）：**
- 提取所有表格，包含表头和行数据
- 必需字段：table_id, title, page, columns, rows, provenance
- rows为对象数组，每行是列名到值的映射

**numerical_data（数值数据）：**
- 提取所有关键数值（收入、利润、增长率等）
- 必需字段：num_id, context, metric_type, unit, value, provenance
- metric_type枚举：currency, percentage, basis_points, multiple, count, ratio, per_share, duration, other
- 百分比建议用0-1表示（如18.2% → 0.182）

**passages（文本片段）：**
- 将文档拆分为可检索的段落/要点
- 必需字段：passage_id, text, page
- 可选：section（章节标题）, labels（如executive_summary, risk, guidance）

**entities（实体）：**
- 识别并标准化所有实体
- 必需字段：entity_id, name
- type枚举：company, index, government, product, other
- 可选：ticker, isin, country, aliases

**claims（主张）：**
- 提取关键结论和预测
- 必需字段：claim_id, text, passage_id
- label枚举：guidance_up, guidance_down, risk, outlook, strategy, other
- 链接证据：evidence.figure_ids, evidence.table_ids, evidence.num_ids

**relations（关系）：**
- 提取实体与指标/事件的关系
- 必需字段：rel_id, subject, predicate, object
- 示例：("Apple", "reports", "Q4 revenue $89.5B")

### 4. 质量要求
- 所有page字段必须与原始页码一致
- 单位不做换算，保持原文单位
- 数值保留原文精度
- 时间格式统一为ISO 8601
- 所有ID字段使用稳定的hash或UUID

### 5. 输出格式
**仅输出最终JSON对象，不要包含任何解释文本、markdown标记或其他内容。**
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
            b64, text_override = self._encode_image_to_base64(image_path)
            page_idx = self._infer_page_from_image_path(image_path)

            # 使用正确的OpenAI消息格式
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的图表数据提取专家。请仅输出一个JSON对象，结构与 schema中 'data.figures' 的 items 完全一致。必须包含 figure_id/title/page/type/series/provenance 等字段。不要输出解释文本。"
                },
                {
                    "role": "user",
                    "content": self._build_figure_request_payload(
                        page_idx,
                        prompt_body=f"请识别此图的结构化数据，并将 page 设为 {page_idx}。",
                        b64=b64,
                        text_override=text_override
                    )
                }
            ]

            resp = await processor.call_model(
                "gemini", messages, max_tokens=getattr(config.api, "LLM_MAX_TOKENS_IMAGE", 1536)
            )
            return resp['choices'][0]['message']['content']

    def _extract_json_from_response(self, response_data: Any) -> Dict:
        """从LLM响应中提取JSON对象，优先处理结构化返回"""
        import re

        # 结构化响应处理
        if isinstance(response_data, dict):
            # 如果已经是符合schema的字典，直接返回
            required_top_level = {"schema_version", "doc", "passages", "entities", "data"}
            if required_top_level.issubset(set(response_data.keys())):
                return response_data

            choices = response_data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    parsed_payload = message.get("parsed")
                    if isinstance(parsed_payload, dict):
                        return parsed_payload
                    if isinstance(parsed_payload, list):
                        for item in parsed_payload:
                            if isinstance(item, dict):
                                return item

                    tool_calls = message.get("tool_calls") or []
                    for call in tool_calls:
                        function_data = call.get("function", {}) if isinstance(call, dict) else {}
                        arguments = function_data.get("arguments")
                        if isinstance(arguments, dict):
                            return arguments
                        if isinstance(arguments, str):
                            try:
                                return json.loads(arguments)
                            except json.JSONDecodeError:
                                continue

                    content = message.get("content")
                    if isinstance(content, list):
                        text_parts = [
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        content = "\n".join(text_parts).strip()
                    if isinstance(content, str) and content.strip():
                        return self._extract_json_from_response(content.strip())

            # 如果无法解析，尝试将字典序列化后走文本兜底逻辑
            try:
                return self._extract_json_from_response(json.dumps(response_data, ensure_ascii=False))
            except TypeError:
                logger.error("响应数据无法序列化为JSON字符串进行解析兜底。")
                return {}

        if isinstance(response_data, (list, tuple)):
            for item in response_data:
                result = self._extract_json_from_response(item)
                if result:
                    return result
            return {}

        if not isinstance(response_data, str):
            response_text = str(response_data)
        else:
            response_text = response_data

        # 文本解析兜底逻辑
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        code_block_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(\{.*?\})\s*\n```',
            r'```json\s*(.*?)```',
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        code_block_match = re.search(r'```(?:json)?\s*(\{)', response_text, re.DOTALL)
        if code_block_match:
            start_pos = code_block_match.start(1)
            brace_count = 0
            for i in range(start_pos, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            json_str = response_text[start_pos:i+1]
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break

        brace_count = 0
        start_idx = -1
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(response_text[start_idx:i+1])
                    except json.JSONDecodeError:
                        start_idx = -1
                        brace_count = 0

        logger.error(f"无法从响应中提取JSON: {response_text[:200]}...")
        return {}

    def _encode_image_to_base64(self, image_path: str,
                                *, allow_text_fallback: bool = True) -> Tuple[Optional[str], Optional[str]]:
        """将图像编码为base64，必要时返回文本兜底"""
        if allow_text_fallback:
            for suffix in ('.json', '.txt'):
                text_path = Path(image_path).with_suffix(suffix)
                if text_path.exists():
                    try:
                        text_payload = text_path.read_text(encoding='utf-8').strip()
                    except UnicodeDecodeError:
                        text_payload = text_path.read_text(encoding='utf-8', errors='ignore').strip()
                    if text_payload:
                        return None, text_payload

        compressed_path = None
        if hasattr(self, "ocr_processor") and hasattr(self.ocr_processor, "get_preprocessed_image_path"):
            compressed_path = self.ocr_processor.get_preprocessed_image_path(image_path)

        candidate_paths = []
        if compressed_path:
            candidate_paths.append(Path(compressed_path))
        candidate_paths.append(Path(image_path))

        for path in candidate_paths:
            if path and path.exists():
                with open(path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    return b64, None

        raise FileNotFoundError(f"无法找到图像或压缩文件: {image_path}")

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
        """批量处理PDF文件，采用阶段A/B流水线并发模式"""
        total = len(pdf_paths)
        logger.info(f"{Colors.BLUE}开始批量处理 {total} 个PDF文件{Colors.RESET}")
        if not pdf_paths:
            return []

        stage_a_limit = getattr(config, "MAX_CONCURRENT_PDFS", total or 1)
        processing_cfg = getattr(config, "processing", None)
        if processing_cfg is not None:
            stage_a_limit = getattr(processing_cfg, "MAX_CONCURRENT_PDFS", stage_a_limit)
        stage_a_limit = max(1, min(stage_a_limit, total))

        api_concurrency = self._get_max_api_concurrency()
        consumer_count = max(1, min(api_concurrency, total))

        logger.info(
            f"{Colors.YELLOW}阶段A最大并发: {stage_a_limit} | 阶段B消费者数量: {consumer_count}{Colors.RESET}"
        )

        job_queue: asyncio.Queue = asyncio.Queue()
        results: List[Dict] = []
        failed_files: Set[str] = set()

        stage_a_semaphore = asyncio.Semaphore(stage_a_limit)

        async def stage_a_worker(pdf_path: str, index: int) -> Tuple[str, Optional[Exception]]:
            async with stage_a_semaphore:
                start_time = time.time()
                pdf_name = Path(pdf_path).name
                logger.info(
                    f"{Colors.CYAN}[阶段A {index}/{total}] 准备: {pdf_name}{Colors.RESET}"
                )
                try:
                    job = await self._process_pdf_stage_a(pdf_path)
                    await job_queue.put(job)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"{Colors.GREEN}[阶段A {index}/{total}] 入队完成: {pdf_name} (耗时: {elapsed:.1f}秒){Colors.RESET}"
                    )
                    return pdf_path, None
                except Exception as exc:
                    failed_files.add(pdf_path)
                    logger.error(
                        f"{Colors.RED}[阶段A {index}/{total}] 失败: {pdf_name} - {exc}{Colors.RESET}"
                    )
                    return pdf_path, exc

        async def consumer_worker(worker_id: int) -> None:
            while True:
                job = await job_queue.get()
                if job is None:
                    job_queue.task_done()
                    logger.info(
                        f"{Colors.YELLOW}阶段B消费者#{worker_id} 已结束{Colors.RESET}"
                    )
                    break
                start_time = time.time()
                pdf_name = Path(job.pdf_path).name
                logger.info(
                    f"{Colors.CYAN}[阶段B#{worker_id}] 开始处理: {pdf_name}{Colors.RESET}"
                )
                try:
                    result = await self._process_pdf_stage_b(job)
                    results.append(result)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"{Colors.GREEN}[阶段B#{worker_id}] 完成: {pdf_name} (耗时: {elapsed:.1f}秒){Colors.RESET}"
                    )
                except Exception as exc:
                    failed_files.add(job.pdf_path)
                    logger.error(
                        f"{Colors.RED}[阶段B#{worker_id}] 失败: {pdf_name} - {exc}{Colors.RESET}"
                    )
                finally:
                    job_queue.task_done()

        stage_a_tasks = [
            asyncio.create_task(stage_a_worker(pdf_path, idx + 1))
            for idx, pdf_path in enumerate(pdf_paths)
        ]

        consumer_tasks = [
            asyncio.create_task(consumer_worker(worker_id + 1))
            for worker_id in range(consumer_count)
        ]

        stage_a_results = await asyncio.gather(*stage_a_tasks)
        stage_a_success = sum(1 for _, err in stage_a_results if err is None)
        logger.info(
            f"{Colors.BLUE}阶段A完成: {stage_a_success}/{total} 个任务已入队{Colors.RESET}"
        )

        for _ in range(consumer_count):
            await job_queue.put(None)

        await job_queue.join()
        await asyncio.gather(*consumer_tasks)

        logger.info(f"{Colors.GREEN}批量处理完成！{Colors.RESET}")
        logger.info(f"成功处理: {len(results)} 个文件")

        failed_list = sorted(failed_files)
        logger.info(f"失败文件: {len(failed_list)} 个")
        if failed_list:
            logger.warning(f"失败文件列表: {[Path(f).name for f in failed_list]}")

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

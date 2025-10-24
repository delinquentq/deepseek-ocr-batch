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

# 新增：优化引擎导入
from md_to_json_engine import MarkdownToJsonEngine
from batch_figure_processor import BatchFigureProcessor
from json_merger import JsonMerger
from md_cleaner import MarkdownCleaner
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
        """调用OpenRouter模型 (简化版，无流式，无JSON强格式)"""
        response = await self.client.chat.completions.create(
            model=config.MODELS[model_name],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,  # 确保输出稳定
            top_p=0.9
        )
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

        # 新增：优化引擎
        self.md_cleaner = MarkdownCleaner()
        self.md_engine = MarkdownToJsonEngine()
        self.batch_figure_processor = BatchFigureProcessor(batch_size=15)  # 每批15张图
        self.json_merger = JsonMerger()

        self._setup_directories()

    def _setup_directories(self):
        """设置目录结构 - 包含分离的输出目录"""
        for dir_path in [config.INPUT_DIR, config.OUTPUT_DIR, config.OUTPUT_REPORT_DIR, config.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"确保目录存在: {dir_path}")

    async def process_single_pdf(self, pdf_path: str) -> Dict:
        """处理单个PDF并生成JSON（单模型 + 图像并发）- 分离输出目录"""
        pdf_path_obj = Path(pdf_path).resolve()

        # 修复路径识别：更健壮的相对路径计算
        input_dir_obj = Path(config.INPUT_DIR).resolve()
        try:
            # 尝试直接计算相对路径
            rel_path = pdf_path_obj.relative_to(input_dir_obj)
            rel_parent = rel_path.parent
        except ValueError:
            # 如果失败，使用原有的父目录匹配方法
            input_root_name = input_dir_obj.name
            rel_parent = Path()
            for parent in pdf_path_obj.parents:
                if parent.name == input_root_name:
                    rel = pdf_path_obj.relative_to(parent)
                    rel_parent = rel.parent
                    break

        pdf_name = pdf_path_obj.stem

        # 从文件名中提取日期（格式：xxx_YYYY-MM-DD.pdf）
        date_str = None
        publication = str(rel_parent) if rel_parent != Path('.') else "unknown"

        import re
        date_match = re.search(r'_(\d{4}-\d{2}-\d{2})$', pdf_name)
        if date_match:
            date_str = date_match.group(1)
            # 移除日期后缀，保留原始文件名
            pdf_name_clean = pdf_name[:date_match.start()]
        else:
            pdf_name_clean = pdf_name

        # 构建输出目录结构
        # OCR: 日期/刊物/文件名/ (需要文件夹存放MD和images)
        # JSON: 日期/刊物/ (直接放JSON文件)
        if date_str:
            ocr_output_dir = str(Path(config.OUTPUT_DIR) / date_str / publication / pdf_name_clean)
            json_output_dir = str(Path(config.OUTPUT_REPORT_DIR) / date_str / publication)
        else:
            # 如果没有日期，使用原有结构
            ocr_output_dir = str(Path(config.OUTPUT_DIR) / rel_parent / pdf_name_clean)
            json_output_dir = str(Path(config.OUTPUT_REPORT_DIR) / rel_parent)

        os.makedirs(ocr_output_dir, exist_ok=True)
        os.makedirs(json_output_dir, exist_ok=True)

        logger.info(f"OCR输出目录: {ocr_output_dir}")
        logger.info(f"JSON输出目录: {json_output_dir}")

        try:
            # 1. DeepSeek OCR处理 - 输出到 OCR 目录
            logger.info(f"{Colors.BLUE}步骤1: DeepSeek OCR处理{Colors.RESET}")
            markdown_path, figure_paths = self.ocr_processor.process_pdf(pdf_path, ocr_output_dir)

            # 2. 读取Markdown内容（完整，无截断）
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            page_count = self._count_pages_from_markdown(markdown_content)

            # ========== 新优化流程：MD清洗 + 规则引擎 + 批量图表处理 ==========

            # 2.5. MD文档清洗（删除无效内容）
            logger.info(f"{Colors.BLUE}步骤2: MD文档清洗{Colors.RESET}")
            cleaned_markdown, clean_stats = self.md_cleaner.clean(markdown_content)
            logger.info(f"{Colors.GREEN}✓ 清洗完成: 减少{clean_stats['reduction_ratio']*100:.1f}%内容, 删除{len(clean_stats['removed_sections'])}个章节{Colors.RESET}")

            # 保存清洗后的MD（可选）
            cleaned_md_path = os.path.join(ocr_output_dir, f"{pdf_name}_cleaned.md")
            with open(cleaned_md_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_markdown)

            # 3. 使用规则引擎直接转换MD到JSON（使用清洗后的MD）
            logger.info(f"{Colors.BLUE}步骤3: 规则引擎转换MD到JSON{Colors.RESET}")
            base_json = self.md_engine.convert(
                cleaned_markdown, pdf_name, date_str, publication
            )
            logger.info(f"{Colors.GREEN}✓ 规则引擎转换完成（无API调用）{Colors.RESET}")

            # 4. 批量处理图表（一次处理10-20张，大幅提速）
            logger.info(f"{Colors.BLUE}步骤4: 批量识别图表数据（{len(figure_paths)}张）{Colors.RESET}")
            semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
            async with OpenRouterProcessor() as processor:
                figures_data = await self.batch_figure_processor.process_figures_batch(
                    processor, figure_paths, semaphore, cleaned_markdown
                )
            logger.info(f"{Colors.GREEN}✓ 批量图表识别完成: {len(figures_data)}/{len(figure_paths)} 张{Colors.RESET}")

            # 5. 合并JSON（规则引擎结果 + 图表数据）
            logger.info(f"{Colors.BLUE}步骤5: 合并JSON数据{Colors.RESET}")
            best_result = self.json_merger.merge(base_json, figures_data)

            # 补充页数到 doc.extraction_run.processing_metadata
            if "doc" in best_result and "extraction_run" in best_result["doc"]:
                md = best_result["doc"].get("extraction_run", {}).get("processing_metadata", {})
                md.setdefault("pages_processed", page_count)
                md.setdefault("successful_pages", page_count)
                md["input_relative_path"] = str(rel_parent)
                best_result["doc"]["extraction_run"]["processing_metadata"] = md

            # 输出统计（包含figure数量验证）
            stats = self.json_merger.get_merge_statistics(best_result)
            logger.info(f"{Colors.CYAN}数据统计: {stats}{Colors.RESET}")

            # 关键验证：确保figures已合并
            figures_count = len(best_result.get("data", {}).get("figures", []))
            if figures_count == 0 and len(figure_paths) > 0:
                logger.error(f"{Colors.RED}⚠️  警告：{len(figure_paths)}张图表未能合并到JSON！{Colors.RESET}")
            else:
                logger.info(f"{Colors.GREEN}✓ 图表数据已合并: {figures_count}/{len(figure_paths)} 张{Colors.RESET}")

            # 保存最终结果
            output_path = os.path.join(json_output_dir, f"{pdf_name_clean}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(best_result, f, indent=2, ensure_ascii=False)
            logger.info(f"{Colors.GREEN}✓ 保存JSON: {output_path}{Colors.RESET}")

            logger.info(f"{Colors.GREEN}✓ PDF处理完成: {pdf_path}{Colors.RESET}")
            logger.info(f"  - OCR结果: {ocr_output_dir}")
            logger.info(f"  - JSON报告: {json_output_dir}")
            return best_result

        except Exception as e:
            logger.error(f"{Colors.RED}✗ PDF处理失败 {pdf_path}: {e}{Colors.RESET}")
            traceback.print_exc()

    async def _extract_figures_data_parallel(self, figure_paths: List[str]) -> List[Dict]:
        """并行提取所有图表的数据（使用视觉模型识别图表内容）"""
        if not figure_paths:
            return []

        semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
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

    async def _extract_single_figure_data(self, processor: OpenRouterProcessor,
                                         image_path: str, semaphore: asyncio.Semaphore) -> Dict:
        """提取单个图表的数据（视觉识别）"""
        async with semaphore:
            try:
                b64 = self._encode_image_to_base64(image_path)
                page_idx = self._infer_page_from_image_path(image_path)

                messages = [
                    {
                        "role": "system",
                        "content": "你是专业的图表数据提取专家。请识别图表类型（柱状图/折线图/饼图/表格等），并提取其中的所有数据。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""请分析这张图表（第{page_idx}页），提取以下信息并以JSON格式输出：

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

仅输出JSON，不要其他文字。"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            }
                        ]
                    }
                ]

                resp = await processor.call_model("gemini", messages, max_tokens=2048)
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
        response = await processor.call_model(
            model_key,
            messages,
            max_tokens=config.api.LLM_MAX_TOKENS  # 使用配置的16000而非65536
        )
        content = response['choices'][0]['message']['content']
        json_result = self._extract_json_from_response(content)
        return {
            "model": model_key,
            "result": json_result,
            "raw_response": content,
            "usage": response.get('usage', {})
        }

    def _build_simplified_extraction_prompt(self, markdown_content: str, pdf_name: str,
                                            page_count: int, date_str: str, publication: str,
                                            figures_data: List[Dict]) -> str:
        """构建简化的提取提示词（整合文本和图表数据）"""
        with open(config.SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_content = f.read()

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

### 1. 严格遵循Schema v1.3.1
输出必须是一个完整的JSON对象，严格符合以下Schema：
```json
{schema_content}
```

### 2. 关键提取指南

**重要提示：**
- fiscal_period格式必须为：`FY2024Q4` 或 `CY2025H1`（不能是`1H`、`Q4`等简写）
- 百分比用0-1表示（如18.2% → 0.182）
- 所有page字段必须是整数（1到{page_count}）
- 所有ID字段使用稳定的hash或UUID
- 时间格式统一为ISO 8601

**必需的顶层字段：**
- `schema_version`: "1.3.1"
- `doc`: 文档元数据
- `passages`: 文本片段数组（至少提取主要段落）
- `entities`: 实体数组（公司、指数等）
- `data`: 包含figures/tables/numerical_data/claims/relations/extraction_summary

**数据提取优先级：**
1. **numerical_data**：提取所有关键数值（收入、利润、增长率等）
2. **tables**：识别并提取所有表格
3. **figures**：识别图表并提取数据（如果文本中有图表描述）
4. **passages**：将文档拆分为可检索的段落
5. **entities**：识别所有公司、指数等实体
6. **claims**：提取关键结论和预测
7. **relations**：提取实体关系

### 3. 输出格式
**仅输出最终JSON对象，不要包含任何解释文本、markdown标记或其他内容。**
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
            b64 = self._encode_image_to_base64(image_path)
            page_idx = self._infer_page_from_image_path(image_path)

            # 使用正确的OpenAI消息格式
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的图表数据提取专家。请仅输出一个JSON对象，结构与 schema中 'data.figures' 的 items 完全一致。必须包含 figure_id/title/page/type/series/provenance 等字段。不要输出解释文本。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"请识别此图的结构化数据，并将 page 设为 {page_idx}。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ]

            resp = await processor.call_model("gemini", messages, max_tokens=2048)
            return resp['choices'][0]['message']['content']

    def _extract_json_from_response(self, response_text: str) -> Dict:
        """从LLM响应中提取JSON对象（增强版，支持markdown代码块和嵌套JSON）"""
        import re

        # 策略1: 尝试直接解析
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # 策略2: 提取markdown代码块中的JSON（```json...```）
        # 使用正则匹配代码块，支持 ```json 或 ``` 开头
        code_block_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json\n{...}\n```
            r'```\s*\n(\{.*?\})\s*\n```',  # ```\n{...}\n```
            r'```json\s*(.*?)```',  # ```json{...}```（无换行）
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        # 策略3: 手动查找markdown代码块中的JSON（括号匹配）
        code_block_match = re.search(r'```(?:json)?\s*(\{)', response_text, re.DOTALL)
        if code_block_match:
            start_pos = code_block_match.start(1)
            # 从{开始，使用括号计数找到完整的JSON
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

        # 策略4: 查找第一个完整的JSON对象（从头开始）
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
                        # 继续查找下一个可能的JSON对象
                        start_idx = -1
                        brace_count = 0

        # 如果都失败，返回空结构
        logger.error(f"无法从响应中提取JSON: {response_text[:200]}...")
        return {}

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
        """批量处理PDF文件（极速并行优化 - RTX 4090 48G）"""
        logger.info(f"{Colors.BLUE}开始批量处理 {len(pdf_paths)} 个PDF文件{Colors.RESET}")
        logger.info(f"{Colors.CYAN}使用极速并行模式：多PDF同时处理{Colors.RESET}")

        # RTX 4090 48G优化：增加并发数
        # OCR很快（27秒），API很慢（120秒），所以可以同时处理更多PDF
        max_parallel = min(6, len(pdf_paths))  # 最多6个PDF同时处理（从3增加到6）

        logger.info(f"{Colors.YELLOW}并发配置: {max_parallel}个PDF同时处理{Colors.RESET}")
        logger.info(f"{Colors.YELLOW}预计总时间: {120 + (len(pdf_paths) - max_parallel) * 20}秒{Colors.RESET}")

        results = []
        failed_files = []

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(max_parallel)

        async def process_with_semaphore(pdf_path: str, index: int):
            async with semaphore:
                try:
                    start_time = time.time()
                    logger.info(f"{Colors.CYAN}[{index}/{len(pdf_paths)}] 开始处理: {Path(pdf_path).name}{Colors.RESET}")
                    result = await self.process_single_pdf(pdf_path)
                    elapsed = time.time() - start_time
                    logger.info(f"{Colors.GREEN}[{index}/{len(pdf_paths)}] 完成: {Path(pdf_path).name} (耗时: {elapsed:.1f}秒){Colors.RESET}")
                    return (pdf_path, result, None)
                except Exception as e:
                    logger.error(f"{Colors.RED}[{index}/{len(pdf_paths)}] 失败: {Path(pdf_path).name} - {e}{Colors.RESET}")
                    return (pdf_path, None, str(e))

        # 并行处理所有PDF
        tasks = [
            process_with_semaphore(pdf_path, i+1)
            for i, pdf_path in enumerate(pdf_paths)
        ]

        # 等待所有任务完成
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"{Colors.RED}处理异常: {item}{Colors.RESET}")
                continue

            pdf_path, result, error = item
            if error:
                failed_files.append(pdf_path)
            elif result:
                results.append(result)

        # 输出处理摘要
        logger.info(f"{Colors.GREEN}批量处理完成！{Colors.RESET}")
        logger.info(f"成功处理: {len(results)} 个文件")
        logger.info(f"失败文件: {len(failed_files)} 个")

        if failed_files:
            logger.warning(f"失败文件列表: {[Path(f).name for f in failed_files]}")

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
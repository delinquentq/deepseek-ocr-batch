"""
批量处理系统配置文件
针对RTX 4090 48G显存优化的参数配置 - 极速版
"""

import os
from pathlib import Path

# ==================== 基础路径配置 ====================
BASE_DIR = Path(__file__).parent
MODEL_PATH = os.getenv("DEEPSEEK_OCR_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")

# ==================== 硬件优化配置 (RTX 4090 48G) ====================
class HardwareConfig:
    """硬件配置 - 针对RTX 4090 48G优化 - 极速模式（6个PDF并发 + 高API并发）"""

    # GPU配置 - 充分利用显存（OCR阶段不占用太多显存）
    GPU_MEMORY_UTILIZATION = 0.85  # 提升到85%（40.8GB），OCR本身显存占用不高
    MAX_CONCURRENCY = 15  # 优化并发数（与API调用并发保持一致）
    TENSOR_PARALLEL_SIZE = 1  # 单卡配置

    # 批处理配置 - 优化OCR速度
    BATCH_SIZE = 10  # 提升到10（平衡速度和显存）
    NUM_WORKERS = 32  # 提升到32线程（CPU预处理加速）

    # 内存配置
    SWAP_SPACE = 0  # 禁用交换空间以提高性能
    BLOCK_SIZE = 512  # 保持512
    MAX_MODEL_LEN = 8192  # 最大模型长度

# ==================== DeepSeek OCR 配置 ====================
class OCRConfig:
    """OCR处理配置"""

    # 模型配置
    MODEL_PATH = MODEL_PATH
    PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
    CROP_MODE = True

    # 图像处理配置
    PDF_DPI = 144  # PDF转图像DPI
    IMAGE_SIZE = 640
    BASE_SIZE = 1024
    MIN_CROPS = 2
    MAX_CROPS = 6

    # 采样参数
    TEMPERATURE = 0.0
    MAX_TOKENS = 8192
    SKIP_SPECIAL_TOKENS = False

    # N-gram重复检测
    NGRAM_SIZE = 20
    WINDOW_SIZE = 50
    WHITELIST_TOKEN_IDS = {128821, 128822}  # <td>, </td>

# ==================== OpenRouter API 配置 ====================
class APIConfig:
    """OpenRouter API配置 - 极速优化"""

    # API密钥和端点
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # 测试模型配置
    MODELS = {
        "gemini": "google/gemini-2.5-flash"
    }

    # 请求配置 - 优化超时和重试
    MAX_RETRIES = 5  # 增加重试次数确保成功
    REQUEST_TIMEOUT = 600  # 10分钟（从5分钟提升）
    RETRY_DELAY_BASE = 1  # 减少退避时间加快重试

    # LLM参数 - 优化生成质量和速度
    LLM_TEMPERATURE = 0.0  # 降低温度确保输出稳定性
    LLM_TOP_P = 0.95  # 提高采样范围
    LLM_MAX_TOKENS = 8000  # 降低到8000加快响应（从16000降低，提升2倍速度）
    LLM_MAX_TOKENS_IMAGE = 1536  # 单独的图像请求上限（从2048进一步下调）

# ==================== 文件路径配置 ====================
class PathConfig:
    """文件路径配置 - 分离OCR和JSON输出"""

    # 基础目录
    BASE_DIR = BASE_DIR
    INPUT_DIR = BASE_DIR / "input_pdfs"
    OUTPUT_DIR = BASE_DIR / "output_results"  # 仅存放OCR结果（MD和图像）
    OUTPUT_REPORT_DIR = BASE_DIR / "output_report"  # 新增：专门存放JSON文件
    TEMP_DIR = BASE_DIR / "temp_processing"
    LOG_DIR = BASE_DIR / "logs"

    # Schema文件
    SCHEMA_PATH = BASE_DIR / "json schema.json"

    # 日志文件
    LOG_FILE = LOG_DIR / "batch_processor.log"
    ERROR_LOG_FILE = LOG_DIR / "errors.log"

    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保所有必要目录存在"""
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, Path) and attr_name.endswith('_DIR'):
                attr_value.mkdir(parents=True, exist_ok=True)

# ==================== 处理配置 ====================
class ProcessingConfig:
    """处理流程配置 - 极速优化"""

    # 质量控制
    ENABLE_QUALITY_CHECK = True
    MIN_FIGURE_COUNT = 0  # 降低最小图表要求（从1到0）
    MIN_CONTENT_LENGTH = 50  # 降低最小内容长度（从100到50）

    # 图像预处理与压缩
    FIGURE_MAX_DIMENSION = 1024  # 缩放统一尺寸
    FIGURE_ENABLE_DENOISE = True  # 启用中值滤波降噪
    FIGURE_JPEG_QUALITY = 70  # 最终输出图像质量
    FIGURE_WEBP_QUALITY = 60  # 临时压缩质量
    FIGURE_TEXT_FALLBACK_MAX_CHARS = 2000  # 文本兜底内容长度限制

    # 错误处理
    CONTINUE_ON_ERROR = True  # 单个文件失败时继续处理其他文件
    SAVE_INTERMEDIATE_RESULTS = True  # 保存中间结果

    # 输出控制
    SAVE_RAW_RESPONSES = False  # 关闭原始响应保存以节省空间和时间
    GENERATE_REPORTS = True  # 生成处理报告

    # 并发控制 - 极保守配置（解决CancelledError）
    MAX_CONCURRENT_PDFS = 2  # 最大并发PDF处理数（从6降到2，避免连接竞争）
    MAX_CONCURRENT_API_CALLS = 10  # 最大并发API调用数（从15降到5，减少连接池压力）

# ==================== 日志配置 ====================
class LogConfig:
    """日志配置"""

    # 日志级别
    LOG_LEVEL = "INFO"

    # 日志格式
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # 文件配置
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

    # 控制台输出
    ENABLE_CONSOLE_OUTPUT = True
    ENABLE_COLOR_OUTPUT = True

# ==================== 验证配置 ====================
class ValidationConfig:
    """数据验证配置 - 严格模式"""

    # Schema验证 - 严格遵循JSON Schema
    STRICT_SCHEMA_VALIDATION = True  # 保持严格验证
    AUTO_FIX_SCHEMA_ERRORS = True  # 自动修复可修复的错误

    # 数据质量检查 - 全面验证
    VALIDATE_FIGURE_DATA = True
    VALIDATE_NUMERICAL_DATA = True
    CHECK_DATA_CONSISTENCY = True

    # 容错设置 - 严格要求
    ALLOW_MISSING_FIGURES = True  # 允许缺少图表（某些文档可能没有）
    ALLOW_EMPTY_DATA_FIELDS = False  # 不允许空数据字段
    MAX_VALIDATION_ERRORS = 10  # 增加最大错误数（从5到10）

    # 新增：Schema强制要求
    ENFORCE_REQUIRED_FIELDS = True  # 强制要求所有必需字段
    VALIDATE_DATA_TYPES = True  # 验证数据类型
    VALIDATE_FORMATS = True  # 验证格式（日期、URI等）

# ==================== 性能监控配置 ====================
class MonitoringConfig:
    """性能监控配置"""

    # 性能指标
    TRACK_PROCESSING_TIME = True
    TRACK_MEMORY_USAGE = True
    TRACK_GPU_UTILIZATION = True

    # 统计报告
    GENERATE_PERFORMANCE_REPORT = True
    SAVE_METRICS_TO_FILE = True

    # 警告阈值
    PROCESSING_TIME_WARNING_THRESHOLD = 300  # 5分钟
    MEMORY_USAGE_WARNING_THRESHOLD = 0.9  # 90%
    ERROR_RATE_WARNING_THRESHOLD = 0.1  # 10%

# ==================== 导出配置类 ====================
class Config:
    """主配置类 - 整合所有配置"""

    hardware = HardwareConfig()
    ocr = OCRConfig()
    api = APIConfig()
    paths = PathConfig()
    processing = ProcessingConfig()
    logging = LogConfig()
    validation = ValidationConfig()
    monitoring = MonitoringConfig()

    @classmethod
    def validate_config(cls):
        """验证配置的有效性"""
        errors = []

        # 检查API密钥
        if not cls.api.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY 环境变量未设置")

        # 检查模型路径
        if not cls.ocr.MODEL_PATH:
            errors.append("DEEPSEEK_OCR_MODEL_PATH 未设置")

        # 检查Schema文件
        if not cls.paths.SCHEMA_PATH.exists():
            errors.append(f"JSON Schema文件不存在: {cls.paths.SCHEMA_PATH}")

        # 检查CUDA可用性
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA不可用，请检查GPU环境")

        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")

        return True

    @classmethod
    # 便捷的 .env 加载器
    def _load_env_file(cls, base_dir: Path):
        env_path = base_dir / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            # 覆盖空值或未设置的环境变量，避免 setdefault 保留空值
                            if (k not in os.environ) or (os.environ.get(k, "") == ""):
                                os.environ[k] = v
            except Exception:
                pass

    @classmethod
    def setup_environment(cls):
        """设置环境"""
        import os
        import torch

        # 加载 .env
        try:
            cls._load_env_file(cls.paths.BASE_DIR if hasattr(cls.paths, "BASE_DIR") else BASE_DIR)
        except Exception:
            pass

        # 重新同步关键环境变量
        cls.api.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", cls.api.OPENROUTER_API_KEY)
        cls.ocr.MODEL_PATH = os.getenv("DEEPSEEK_OCR_MODEL_PATH", cls.ocr.MODEL_PATH)

        # 设置CUDA环境
        if torch.version.cuda == '11.8':
            os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

        os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '0')
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", '0')

        # 确保目录存在
        cls.paths.ensure_directories()

        # 不在此处强制验证，留给测试或运行时处理
        return True

# ==================== 便捷函数 ====================
def get_config():
    """获取配置实例"""
    return Config()

def setup_environment():
    """设置环境的便捷函数"""
    return Config.setup_environment()

# ==================== 模块导出 ====================
__all__ = [
    'Config',
    'HardwareConfig',
    'OCRConfig',
    'APIConfig',
    'PathConfig',
    'ProcessingConfig',
    'LogConfig',
    'ValidationConfig',
    'MonitoringConfig',
    'get_config',
    'setup_environment'
]


class BatchConfig:
    def __init__(self):
        # 先尝试加载 .env
        try:
            base_dir = Path(__file__).resolve().parent
            self._load_env_file(base_dir)
        except Exception:
            pass
        
        # 从环境变量读取配置
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
        self.DEEPSEEK_OCR_MODEL_PATH = os.getenv('DEEPSEEK_OCR_MODEL_PATH', 'deepseek-ai/DeepSeek-OCR')
        self.CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        self.VLLM_USE_V1 = int(os.getenv('VLLM_USE_V1', '0'))
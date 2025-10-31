# Repository Guidelines

## Project Structure & Module Organization
主流程脚本位于仓库根目录：`batch_pdf_processor.py` 负责OCR→Markdown→JSON流水线，`run_batch_processor.py` 提供CLI入口，`config_batch.py` 暴露分层配置对象。输入、输出与缓存分别在 `input_pdfs/`、`output_results/`、`temp_processing/`，测试数据写入 `logs/` 与 `output_report/` 以便审计。图像与语言后处理在 `process/`、`md_to_json_engine.py`、`json_merger.py` 中，GPU推理组件保存在 `deepencoder/` 与 `deepseek_ocr.py`。测试脚本 `test_*.py` 与样例JSON (`report json.json`) 共置于根目录，便于直接运行；备份实现以 `_v1_backup.py` 后缀保留，请仅在必要时同步更新两份代码，确保接口一致。生成物会落在 `output_results/<document>/`，下含 `images/`、`*.json`（兼容输出 `*_final.json`）与 `processing_log.txt`，请勿手动篡改，调试可另建 `sandbox/` 目录。

## Build, Test, and Development Commands
使用 `conda create -n deepseek-ocr python=3.10` 与 `pip install -r requirements_batch.txt` 初始化依赖，随后执行 `bash setup_batch_environment.sh` 验证CUDA与模型缓存。批量处理采用 `python run_batch_processor.py --setup` 进行依赖预检，再用 `python run_batch_processor.py -f sample.pdf -y` 排程批次；需要快速抽样可追加 `--limit 5 --dry-run`。图文调试可通过 `python run_dpsk_ocr_image.py --image 1_0.jpg`，PDF 单测运行 `python run_dpsk_ocr_pdf.py --file input_pdfs/demo.pdf`。部署服务器端使用 `bash server_deploy.sh OPENROUTER_API_KEY=...`，持续监控可执行 `tail -f logs/batch_processor.log`、`watch -n5 nvidia-smi`，或 `python test_api_connection.py --model gemini` 检查OpenRouter响应；本地调试异常时可用 `python -m pdb run_batch_processor.py --setup-only`。

## Coding Style & Naming Conventions
全仓Python遵循4空格缩进、`black`/`ruff`兼容格式（推荐 `python -m ruff check .` 和 `python -m black .` 在提交前运行）。公共类型通过 `dataclass` 和 `typing` 注解暴露，文件级日志统一使用 `logging.getLogger(__name__)`，异常分层使用标准异常 (`ValueError`, `RuntimeError`) 并附带可定位的信息。命名采用蛇形：`process_ngram_limit`、`figure_paths`，常量与配置使用全大写，例如 `MAX_CONCURRENT_PDFS`；异步函数需以 `async_` 前缀或动词说明用途，保持KISS。重复的I/O、JSON序列化或GPU设置逻辑请提炼到工具函数或 `Config`，避免偏离DRY/YAGNI，并为复杂流程写模块级docstring说明输入输出契约。

## Testing Guidelines
所有新增功能至少覆盖一个脚本级测试（例如 `python test_batch_system.py`）以及针对性模块测试（`python test_figure_extraction.py`、`python test_parallel_processing.py`）。测试命名为 `test_<scenario>.py` 并放在仓库根目录，临时文件写入 `temp_processing/fixtures` 后在 `tearDown` 中清理，保持CI可重复。提交前执行 `python test_batch_system.py` 和受影响的 `test_*.py`，对API或GPU相关改动补充 `python test_api_connection.py`，并在PR描述中粘贴关键日志、吞吐量或显存占用，目标保持与基线相同或更优；快速冒烟可运行 `python test_simple.py` 或 `python test_det.py`。当测试生成新的 `test_report.json` 时，请将差异附在PR评论，方便审阅者复现结果。

## Commit & Pull Request Guidelines
Git历史遵循Conventional Commits（如 `feat: ...`, `refactor: ...`），摘要控制在40-60字符，正文列出动机、实现、回滚提示。分支命名推荐 `feature/<scope>` 或 `fix/<issue>`，PR需包含：变更概要、关联Issue、性能或准确率指标、截图/样例JSON路径、测试命令输出，以及配置/部署变更清单。推送前执行 `git status -sb` 确认无调试文件；若修改GPU或部署脚本，请请求至少一名GPU维护者review，并在描述中列出所需环境变量与回退指引。

## Security & Configuration Tips
敏感密钥通过 `.env` 或CI密文注入，永远不要提交 `OPENROUTER_API_KEY`、`DEEPSEEK_OCR_MODEL_PATH` 等，可用 `git update-index --assume-unchanged .env` 防泄露。所有脚本通过 `Config.validate_config()` 校验Schema与路径，新增Schema时同步更新 `json schema.json` 与 `json结构范式说明.md`。服务器部署前运行 `bash setup_batch_environment.sh` 以检测驱动与CUDA版本，必要时在PR中记录 `CUDA_VISIBLE_DEVICES`、`GPU_MEMORY_UTILIZATION`、`MAX_CONCURRENT_PDFS` 等参数以便复现；本地配置样例维护在 `.env.example`，新增键时同步注释用途并声明作用域。

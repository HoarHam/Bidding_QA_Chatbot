# 开发环境说明

## 快速开始

```bash
uv sync --dev                 # 后端依赖 + 测试/格式化工具
cd frontend && npm install    # 前端依赖

uv run bidding-qa-chatbot check                 # 检查配置与依赖连通性
uv run bidding-qa-chatbot ingest                # 语料入库（首次必做）
uv run bidding-qa-chatbot serve                 # 后端 :8001
cd frontend && npm run dev                      # 前端 :3000
```

## 本机环境的已知坑（都实测踩过）

### 1. 系统代理会让 httpx 连不上 Qdrant

本机开启了 Windows 系统代理（注册表 `ProxyEnable=1`，指向 `127.0.0.1:10808`）。
httpx 默认 `trust_env=True` 会走它，而该代理连不通 Qdrant Cloud，表现为 `ConnectError`
（裸 TLS 连接和 `curl` 都正常，只有 httpx 失败——很容易误判成网络不通）。

因此 Qdrant 客户端与健康检查客户端默认 `trust_env=False`（见 `HTTP_TRUST_ENV`）。
如果你的网络确实必须经代理，把它设为 `true`。

### 2. HuggingFace 直连不通

`huggingface.co` 与 `cdn-lfs.huggingface.co` 均超时，`hf-mirror.com` 可用。
`.env` 里已配好 `HF_ENDPOINT=https://hf-mirror.com`，**不要删**。
`config.py` 会把该值写回 `os.environ`，因为 `huggingface_hub` 只认环境变量——
这也是 `embeddings/*` 里所有模型导入都必须延迟到函数内部的原因。

### 3. 模型加载会做无谓的联网校验

`huggingface_hub` 默认在每次加载时向远端 HEAD 一批**可选**配置文件，
国内网络下会反复超时重试，把启动拖慢到分钟级。模型加载因此采用
「先 `local_files_only=True` 读缓存，失败再联网下载」的策略：
缓存命中时瞬时完成，首次运行仍能正常下载。

### 4. Windows 默认编码是 GBK

本机 locale 为中文，Python 打开文件默认用 GBK，读中文语料或打印中文会直接崩
（`UnicodeDecodeError` 或 stdout 编码错误）。**所有文件 IO 显式写 `encoding="utf-8"`**，
命令行建议设 `PYTHONUTF8=1`。

### 5. `curl` 的证书吊销检查

Windows Schannel 在吊销服务器不可达时报 `CRYPT_E_REVOCATION_OFFLINE`。
调试时加 `--ssl-no-revoke` 即可。**这只影响 curl**，Python 的 OpenSSL 栈不受影响。

### 6. 包索引

清华源对本站返回 **403**，会让依赖解析直接失败。现已改用阿里云镜像，两处保持一致：

- 环境变量 `UV_DEFAULT_INDEX`（Windows 用户环境变量，改后需重开终端）
- `pyproject.toml` 的 `[tool.uv].index-url`

**环境变量优先级高于项目配置**，所以两处都要写对；另外注意**结尾斜杠**也要一致——
只差一个斜杠就会让 `uv.lock` 在下一次解析时把每条依赖的 `source` 行都改写一遍
（不影响功能，但会产生无意义的 diff）。

## 测试

```bash
uv run pytest                                    # 全部（离线）
uv run pytest tests/test_bm25.py::test_vocab_is_sorted_and_unique
RUN_MODEL_TESTS=1 uv run pytest -m model         # 需要真实模型权重的用例
uv run ruff check . && uv run ruff format --check .
```

测试**不需要任何密钥或云服务**：LLM 用 `respx` 或假 provider，
向量库用 `QdrantClient(":memory:")`。

`tests/conftest.py` 里有一个 autouse 夹具会清掉 `.env` 注入的密钥类环境变量——
`config.py` 在导入时执行 `load_dotenv()`，不清掉的话测试会读到**真实的付费密钥**。

## 代码约定

- 所有文件 IO 显式 `encoding="utf-8"`。
- 模型/重依赖在函数内部延迟导入（`embeddings/*`、`llm/*`）。
- 密钥一律 `SecretStr`；异常消息、日志、健康检查输出都过 `log.redact()`。
- 错误分期：可重试（超时/连接/5xx）vs 致命（401/400）。
  重试策略只允许存在于 `llm/chain.py` 一处，SDK 的 `max_retries` 全部设 0。
- 改动 `schemas.py` 后同步 `frontend/lib/types.ts`，
  `tests/test_contract_sync.py` 会在字段名不一致时失败。

## CUDA（可选，本机不适用）

默认从 PyPI 安装的是 **CPU 版 torch**（Windows 轮子约 122MB，实测不含 CUDA）。
本机没有 NVIDIA GPU，因此这条路径**未经实测**。若换到有卡的机器：

```bash
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu124
```

注意：本机 `download.pytorch.org` 不可达（HTTP 000），需要换网络环境。
代码会自动检测：`torch.cuda.is_available()` 为真时用 GPU，否则用 CPU。

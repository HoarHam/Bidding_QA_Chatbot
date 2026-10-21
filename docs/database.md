# 账号与对话历史（PostgreSQL）

用户账号管理 + 对话历史持久化：跨会话恢复、搜索历史对话。

## 本机搭建 PostgreSQL

```bash
# 装（本机已有 chocolatey）
choco install postgresql17 -y
# 或： winget install PostgreSQL.PostgreSQL.17

# 建库（会提示输入安装时设的 postgres 密码）
psql -U postgres -c "CREATE DATABASE chatbot;"
```

然后更新 `.env`：

```ini
POSTGRES_PASSWORD=<你设的真实密码>    # 目前仍是占位符 <your_postgres_password>
```

**注意**：`.env` 里的 `POSTGRES_*` 走的是 `pydantic-settings` 的字段名映射
（`POSTGRES_HOST` → `postgres_host`），而**整串连接串**用 `DATABASE_URL`：

```ini
# 想跳过 POSTGRES_* 直接给整串时用这个（优先级更高，也会覆盖 alembic 的目标库）
# DATABASE_URL=postgresql+psycopg://user:pass@host:5432/chatbot
```

> 密码里的特殊字符必须 URL 编码。代码里已用 `quote_plus` 处理从 `POSTGRES_*`
> 拼出来的连接串；但如果你手写 `DATABASE_URL`，`@`、`/`、`:` 等要自己编码成
> `%40`、`%2F`、`%3A`——不编码会把连接串解析成完全不同的主机。

### 建表

```bash
uv run alembic upgrade head
```

`create_all` 只在测试里用；正式环境的表结构变更一律走迁移，
因为它不会修改已存在的表。

```bash
uv run alembic revision --autogenerate -m "描述"   # 改完模型后生成迁移
uv run alembic upgrade head                        # 应用
uv run alembic downgrade -1                        # 回滚一步
uv run alembic upgrade head --sql                  # 只输出 SQL，供 DBA 审阅
```

`DATABASE_URL` 会同时作用于应用与 alembic，两者不会指向不同的库。

## Windows 上的一个硬约束

**psycopg 的异步模式无法在 `ProactorEventLoop` 上运行**，而那是 Windows 的默认事件循环。
会抛：

```
psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode
```

实测 uvicorn 在 `reload=False` 时会用 Proactor，而且**设事件循环策略不足以覆盖它**
（uvicorn 创建循环的时机不受策略约束）。因此：

- `cli.py` 在 Windows 上**显式创建 `SelectorEventLoop`** 交给 uvicorn，不依赖策略；
- `--reload` 走另一条分支（uvicorn 要自己起监督进程），改用模块顶层设置策略，
  子进程重新导入时才生效；
- `alembic/env.py` 同样在 `asyncio.run()` 之前设置策略；
- `app.py` 启动时会检测循环类型，不对就**明确报错**并给出解决办法，
  而不是让它在第一次查询时抛一句难懂的 `InterfaceError`。

## 表结构

```mermaid
erDiagram
    users ||--o{ conversations : 拥有
    conversations ||--o{ messages : 包含

    users {
        uuid id PK
        string username UK
        string email UK
        string password_hash
        bool is_active
    }
    conversations {
        uuid id PK
        uuid user_id FK
        string title
        int message_count
    }
    messages {
        uuid id PK
        uuid conversation_id FK
        string role
        text content
        text reasoning
        json sources
        json attachments
        string feedback
        datetime feedback_at
        string provider
        string model
    }
```

设计要点：

- **消息顺序只能按 `seq`，不能按 `created_at`。** 同一事务里连续插入的消息会拿到
  **完全相同**的 `created_at`（Windows 上 `datetime.now()` 只有约 15ms 分辨率），
  而兜底比较的 `id` 是随机 UUID——顺序因此是随机的，表现为**用户回看历史时消息错位**。
  这个问题实测约每 8 次运行出现 1 次，是靠跑多次测试才暴露出来的。
  `seq` 是会话内序号（从 1 开始），与 `message_count` 同源，
  并有 `(conversation_id, seq)` 唯一约束：并发写入撞车时报错，而不是产生两条顺序相同的消息。
- **`message_count` 是冗余字段。** 会话列表要显示条数，不冗余就得为每条会话
  做一次子查询。由 `add_message` 一并维护，其值同时就是下一条消息的 `seq`。
- **`sources` 存成 JSON 而不是关联表。** 引用来源只随消息整体读写，
  从不单独查询，单独建表只增加连接开销。
- **`feedback` / `feedback_at`**（迁移 `0006`）：用户对助手消息的 👍/👎，
  仅 `"up"` / `"down"` / NULL；`null` 表示未评价/已清除。只允许写本人会话中的助手消息，
  用于后续分析检索与生成策略。
- **`attachments` 同样存成 JSON**（迁移 `0005`）。内容是
  `[{id, filename, content_type, size, kind, ocr_text}]`：对外接口只下发元数据，
  `ocr_text` 仅服务端用于同会话追问。该列使用 `none_as_null=True`，
  让 Python `None` 写成 SQL NULL——否则 `attachments IS NOT NULL` 会把每条无附件
  消息也当成有值，追问注入的 limit 会被空消息挤占。
- **类型刻意可移植**（`sa.Uuid`、`sa.JSON().with_variant(JSONB, "postgresql")`）：
  Postgres 上得到原生 `uuid` 与 JSONB，SQLite 上退化为可用的等价物。
  这样 CI 与本地无 Postgres 时也能跑完整测试。
- **约束名显式约定**（见 `db/base.py`）：数据库自动生成的名字会随环境变化，
  导致 autogenerate 出的迁移在不同环境产生不同 SQL。

## 认证

- **JWT**（HS256），默认 7 天有效。密钥来自 `JWT_SECRET`，**必须显式配置**——
  不给默认值：默认密钥等于把所有人的账号交出去，随机生成的临时密钥则会在重启后
  让所有 token 失效、多 worker 下互相冲突。
- **密码用 bcrypt 直接哈希**，不引 passlib（多年未更新，且与 bcrypt 4.x 有兼容告警）。
- **bcrypt 有 72 字节上限**，超出部分被静默丢弃（中文一个字 3 字节，24 个汉字就触顶）。
  代码显式拒绝超长密码，而不是让它悄悄降级成「前 72 字节相同即视为同一密码」。
- **登录不区分「用户不存在」与「密码错误」**，两者返回完全相同的响应——
  否则登录接口会变成用户名枚举接口。

### 越权防护

**每个查询都带 `user_id` 条件**（见 `db/repository.py` 顶部说明）。
只按主键取会话（`WHERE id = ?`）会形成越权漏洞：任何人拿到一个 UUID
就能读到别人的对话。授权写在查询里，而不是依赖调用方记得检查。

`tests/test_conversations.py` 有专门的用例端到端验证：Bob 建的会话，
Alice 拿 id 去读/删都返回 404。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/auth/register` | 注册并直接返回 token |
| POST | `/api/auth/login` | 用户名**或邮箱**登录 |
| GET | `/api/auth/me` | 当前用户 |
| GET | `/api/conversations` | 会话列表（按最后更新倒序） |
| GET | `/api/conversations/{id}` | 会话详情 + 全部消息（跨会话恢复） |
| DELETE | `/api/conversations/{id}` | 删除会话（级联删消息） |
| GET | `/api/conversations/search?q=` | 搜索历史对话 |

对话接口全部要求 `Authorization: Bearer <token>`，前端收到 401 会跳登录页。

### 搜索为什么用 ILIKE 而不是全文检索

中文**没有默认分词器**，Postgres 的 `to_tsvector` 会把整句当成一个词，
基本不工作。数据量再大时应接 `pg_trgm` 或外部分词，而不是改成 tsvector。

用户输入里的 `%` `_` `\` 会被转义，否则搜 `100%` 时 `%` 会当通配符用
（有测试覆盖）。

## 消息落库的时机

**用户消息在开流之前落库**（`_resolve_conversation`），**助手回答在流结束后落库**
（`_persist_assistant`）。

两者都用**独立事务**，没有复用请求作用域的 `DbSession`——它的提交发生在响应
**完全发送之后**（StreamingResponse 的依赖清理在流结束才执行），会导致两个问题：
流中途断开时用户消息一并丢失；助手消息落库用的是另一个会话，读不到尚未提交的
会话行而外键失败。

助手回答落库**失败只记日志**，不影响已经发给用户的回答——用户拿到了答案，
不该因为存储问题看到一个报错。

## 前端

- 登录态存在 `localStorage`，服务端渲染时访问不到，所有读写都判空。
- **安全提示**：localStorage 里的 token 可被同源 JS 读取，对 XSS 无防护。
  更稳妥的是 httpOnly cookie + CSRF 防护。当前阶段选前者是为了简化前后端分离部署，
  若要对外开放应重新评估。
- 会话 id 来自后端 `done` 事件：首轮对话的会话是服务端建的（用户消息要先落库
  才能跑 Agent），前端接住 id 后，后续提问才接得上同一会话。
- **`/api/sources/*` 仍不要求登录**：`<img src>` 无法携带 Authorization 头，
  加了会让参考来源里的图片全都加载不出来。这些接口只暴露知识库语料，
  不含任何用户数据，因此这个边界是合理的。

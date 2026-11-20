# Retrieval

Retrieval 是一个面向论文语料、项目工作记忆和代码知识库的检索平台。生产系统由 React、FastAPI、SQLite、Qdrant、BGE-M3、BGE reranker、BWH gateway 和可安装的 MCP 客户端组成。

## 当前能力

- 论文 hybrid/dense/keyword 搜索、流式搜索、rerank、论文内搜索和文档匹配。
- 101k+ 论文的 Markdown/PDF 阅读、HTTP Range 下载和最多 10 篇批量归档。
- ACE Journal 文本记忆 CRUD、全文 chunk 检索、目录 snapshot 索引（异步 bulk job）和清理。
- Code KB 本地 AST 切块、gitignore、增量 manifest、异步索引、watcher 和路径过滤。
- 重复设计/代码发现：按文件对报告语义重复，找得到不共享任何标识符的重新实现（`/api/code/duplicates` 与 MCP 工具 `find_duplicate_code` 均于 2026-09-02 上线）。
- Public Code Core 受管公共源码语料：source/revision 控制面、原子 promotion 和带 rerank 的搜索。
- 医学 OA 语料采集线（EuropePMC，internal-only，强制代理与限速）。
- Index Catalog 与 ACE Playbook。
- 公网 OIDC/API key、角色权限、额度计费、幂等退款和访问审计。
- PyPI 当前稳定版为 `retrieval-mcp==0.5.14`（2026-11-15），**35 个** stdio 工具。0.5.14 让自动索引（文件监视器和自动刷新）每个 scope 至少间隔 5 分钟提交一次，间隔内的改动合并成一批，手动 `index_code` 不受限；后端把策略 `min_check_interval_s` 的下限定为 300 秒。0.5.13（2026-11-13）让 `index_code` 在同一次遍历里同时索引文档（进入 Journal，用 `journal_search` 查），自动索引一并覆盖文档；`journal_index_dir` 已弃用。0.5.12 显示服务端在索引后自动做的重复代码检测摘要（每个结果只出现一次，由服务端判定）。0.5.11 把初始化指令改成硬规则：任务里第一次搜索必须用检索 MCP，之后才用 grep 等命令，例外三条。0.5.10 减少整 scope 读取：基线用精简 inventory，未变化的 `journal_index_dir` 提前返回，增量 `index_code` 直接发送删除列表。0.5.9 可选把 Claude Code / Codex 对话同步进 Journal（`RETRIEVAL_JOURNAL_CONVERSATIONS=1`），deja 与 zstd 解码由插件自带。0.5.7 撤掉 0.5.6 写死在文件 hash 里的切块代数，增量身份恢复为文件字节 sha256；修复旧客户端切出的块请对项目跑一次全量 `index_code`（默认即全量），后端对 id 与内容都未变的块直接复用已存向量，只有真正变了的块重新 embedding。0.5.6 修复代码切块：chonkie 返回 UTF-8 字节偏移，按字符切片使非 ASCII 字符之后的块整体错位（内容和行号都错），超过 384K 字符的源文件只有一个块；大文件改为按空行分段后正常切块。0.5.5 默认逐行显示代码源行号，search_code(show_line_numbers=False) 可返回纯代码；0.5.4 的 `index_inventory` 每个项目一行，默认最多 40 个（需关注的快照优先，其次最近索引），超出时注明"Listing k of n"，并新增 `project` 路径过滤；305 个 scope 曾输出 56,159 字符。0.5.3 的 `import_public_code_repo` 自动检查创建者在仓库路径下的私有索引，精确匹配的 chunk 复用 dense/sparse 向量，只有缺失或变化的部分重新 embedding；job results 返回 `reused_chunk_count`。私有索引保留，公开 payload 不带私有 owner/project/host，工具面与 0.5.2 完全相同。0.5.2 引入本地 clone → Git bundle → 服务端 mirror/index 导入；0.5.1 改善后端错误原因；0.5.0 将公共与联合检索并入 `search_code(sources="private"|"public"|"all")`。普通无参客户端通过轻量 shim 复用每用户唯一 daemon。
- `0.4.15` 新增 MCP 工具 `search_all_code`：一次调用同时检索私有 Code KB
  与受管 Public Code 语料，用 RRF（`rerank=false`，默认）或共享 cross-encoder
  （`rerank=true`）融合成一个排序，每条结果标注 `[private]` 或
  `[public:<source_id>@<rev>]`。`include_public=true` 时鉴权 fail-closed：非 admin
  直接报错，绝不悄悄降级为纯私有结果。后端新增路由
  `POST /api/code/federated-search`（挂在 `/api/code/search` 旁，契约不变）。
  工具面 34 → **35 个**，其余 34 个工具的 digest 不变
  （`scripts/maintenance/regen_digests.py --expect search_all_code`）。
- 未发布的 `0.4.14` 候选让上传文档在检索之后可达，工具名与签名不变，仍为 **34 个**。
  `papers` 的 upload/webpage payload profile 里根本没有 `paper_id` 键，所以 `search_papers`
  搜到一本书时那行 `paper_id:` 是空的，`search_within_paper` / `get_paper_markdown` 无从
  接手——书找得到却读不了。现在这类结果输出 `doc:<documents.id>`，路由到后端新增的
  `doc_ids` 过滤器和 `/api/documents/{id}/fulltext|markdown`。纯数字 id 等价接受：实测
  生产 106,156 个 paper_id 里 0 个是纯数字，两个命名空间撞不了。整本书动辄数百万字符，
  所以不带 `save_path` 的内联形式在 400,000 字节以上拒绝并提示改用 `save_path`，论文维持
  原有不设限行为。tools/list digest `f74cad71…`。
- 已发布的 `0.4.13` 为 **34 个**工具：`match_paper` 改为快速提交现有 Tasks
  系统中的 exact durable job，新增 `match_paper_status` 查询进度、结果或取消；在真实
  scratch Qdrant、真实 BGE-M3/reranker、HTTP 和官方 MCP stdio 门完成前不替换稳定版。
  该候选的 durable job 部分曾于 2026-09-16 被三名审查阻断（取消缺原子 CAS、跨进程无
  claim/lease、retry 错路由、无 end-to-end deadline、rerank 不可取消、公网无 owner
  隔离）。2026-09-18 已逐项处理并在 `.88` 实测：提交 35ms 返回；rerank 阶段下发取消
  47ms 生效且不发布部分结果；非 owner 读/取消均 404。"retry 错走 crawl"经查不成立——
  `_retry_urls_for_task` 的 task_type guard 自初始提交就在。
  `search_papers` 另新增 `source_type`（`paper`/`upload`/`webpage`/`review`）：10.6 万篇
  论文会把单本上传书挤出 top_k，`source_type="upload"` 才能只检索书。
- `0.4.9` 相对 0.4.8 的两处变化：
  - `index_code` / `journal_index_dir` 的描述改了（嵌套 scope guard 现在放行「子目录关掉 gitignore 过滤」这种确实增加覆盖的情况）。
  - 大于 64 KiB 的上传用 zstd-3 压缩（实测 335 KB → 4.9 KB），新增 `zstandard` 依赖；响应压缩由 httpx 自动协商。**逐请求协商，旧后端不需要同步升级**：不支持的后端在路由执行前返回 400/415/422，客户端明文重发一次后本会话不再压缩。
- `0.4.10` 修的是跨项目 scope 串台：daemon 是每用户单例，相对 scope（`scope_dir="."`、
  `repo_root="."`、`save_dir="."`）却在 daemon 进程里解析，于是**谁先拉起单例，谁的目录就成了
  所有项目的 `"."`**——journal 读到别的仓库，`journal_record` 把条目写进别的仓库的 namespace，
  下载也落到错的项目。现在 shim 在认证过的 raw 握手里带上自己的目录，daemon 用 contextvar
  按会话绑定（并发的不同项目互不干扰）；`DAEMON_PROTOCOL_SCHEMA_VERSION` 1 → 2，修好的 shim
  不会复用没修的 daemon。工具面不变（33 个，digest 未变）。
- `0.4.11` 已于 2026-09-15 上传 PyPI，但**禁止使用**：它加入 Retrieval-first instructions
  和五个 search 工具的有序 query 数组，却在 Python 3.12 的真实 `uvx --no-cache` 中因
  description 缩进产生 daemon surface digest drift，启动期 fail-closed。Python 3.13 wheel
  smoke 虽通过，不能覆盖该跨版本失败。`0.4.12` 在追加 suffix 前先 canonicalize 原描述；
  Python 3.10--3.13 surface digest、Python 3.12 wheel/官方 PyPI `uvx --no-cache` 和真实后端
  数组调用均已通过，现为稳定版本。

## 快速导航

| 目标 | 文档 |
|---|---|
| 理解当前系统 | [overview.md](overview.md) |
| Agent/开发约束 | [AGENTS.md](AGENTS.md) |
| 从功能定位到代码 | [INDEX.md](INDEX.md) |
| 本地与生产环境 | [env.md](env.md) |
| 当前未完成工作 | [plan.md](plan.md) |
| 已完成工作与验证 | [journal.md](journal.md) |
| API 合同 | [docs/API.md](docs/API.md) |
| 部署手册 | [docs/DEPLOYMENT_CN.md](docs/DEPLOYMENT_CN.md) |
| MCP 使用 | [mcp_plugin/README.md](mcp_plugin/README.md) |

## 本地开发

```powershell
cd C:\Users\Arc\Prd\retrival
.\.venv\Scripts\python.exe -m pytest tests
npm --prefix frontend run build
.\.venv\Scripts\python.exe main.py
```

本地需要下载外部资源时使用 `http://127.0.0.1:10808`。后端修改后执行完整重启，不依赖 `uvicorn --reload` 更新嵌套模块。

## MCP

After an approved package release is installed, keep the ordinary command-style MCP configuration.
The default `retrieval-mcp` process is a lightweight stdio proxy: it auto-starts or
reuses one per-user daemon and forwards this client's JSONL session to it. Do not
add a client URL or transport setting; normal clients need no separate preparation
command.

```json
{
  "mcpServers": {
    "retrieval": {
      "command": "retrieval-mcp",
      "args": []
    }
  }
}
```

Existing `uvx` command configurations remain compatible. Version-pinned runners
should use `retrieval-mcp==0.4.12`; a once-installed `retrieval-mcp` avoids a
package-runner wrapper for each agent. `status` and `stop` are diagnostics/control
commands; normal MCP clients should invoke no-argument `retrieval-mcp` only.

公网连接设置 `RETRIEVAL_API_URL=https://retrieval.rnarket.com` 和 `RETRIEVAL_API_KEY=<sk-key>`；可信内网默认连接 `http://10.100.100.111:8000`。

Windows 盘符路径和 UNC scope 会统一分隔符并按大小写不敏感的目录身份处理；旧索引中的混合大小写 scope 仍可检索和删除。`path_glob` 必须相对 `repo_root`/`scope_dir`，支持 `\` 输入、`**`、目录前缀和单文件路径；绝对路径、盘符相对路径（如 `E:docs`）及 `..` 会返回具体校验错误。`\\?\` / `\\.\` device spelling 不作为独立 scope 接受，请传等价的普通 drive/UNC 路径，避免同一目录形成两个身份。

## 当前生产快照

2026-09-11 `/api/stats`：102,286 documents、101,835 papers、13,555,041 SQL chunks、
13,556,378 paper Qdrant points；Public Code 99 configured / 88 indexed、
`public_code_chunks` 3,735,500 points，另有 `code_chunks` 2,328,461 points、
`journal_chunks` 115,005 points。动态规模只以运行接口与 Qdrant 只读查询为准。

2026-09-11 `git fetch --prune origin` 后，本地 `HEAD=21a4660`、
`origin/main=27fb486`，ahead 2 / behind 0。两条领先提交已分别部署或发布，但仍需推送
Gitea；不能把交付状态与 Git 灾备同步状态混为一谈。

`.88` 后端已包含 2026-09-02 的 90 文件 manifest、2026-09-04 的 OOM 修复和
2026-09-06 的 match-doc event-loop offload。2026-09-11 复核：PID `1658270`、
`ActiveEnterTimestamp=2026-09-06 19:01:32 CST`、`NRestarts=0`，door `:18080` /
internal `:18181`。运行时 OpenAPI **101 paths / 112 operations**；关键部署文件与本地
SHA-256 逐项一致。

**代码已提交不等于已上线。** 判断某项是否上线请以 [journal.md](journal.md) 的生产
实测记录为准，不要以提交历史或工作树推断。2026-09-11 Code KB 增量刷新扫描到
`656` 个受管源码文件、`0` changed chunks、`already up to date`，watcher active；当前
`search_code` 可召回 match-doc offload 与 MCP relative-scope 修复。

## 安全提醒

- GPU0 默认禁止使用；`10.100.100.88` 是经操作者授权的主机级例外，`10.100.100.111` 仍禁止。详见 [AGENTS.md](AGENTS.md)。
- 禁止提交 token、cookie、OAuth/payment secret、私钥或真实用户凭据。
- `papers` collection schema、BWH 路由和大规模索引变更必须先读 [AGENTS.md](AGENTS.md) 的验证门。


当前发布 `retrieval-mcp==0.5.5`（2026-10-11）：`search_code` 新增 `show_line_numbers=True`，私有/公共/联合及批量查询默认逐行显示源文件行号；传 `False` 返回纯代码。前端默认独立行号栏，复制代码保留原始内容。该选项仅影响展示，服务 API、索引与 embedding 不变。

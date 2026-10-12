# Lumi 项目架构图

> **Lumi** 是一个开源、本地优先的桌面照片管理器（macOS / Windows / Linux），
> 基于 [Lap](https://github.com/julyx10/lap) 二次开发。原作者 julyx10，本 fork 维护者 liuyihua2015。
> 你的照片永远留在自己的硬盘上，AI 搜索、人脸识别都在本机运行，无需上传云端。

## 一图看懂（Mermaid）

```mermaid
flowchart TD
    U["👤 你（用户）<br/>打开 Lap，浏览、搜索、整理照片"]

    subgraph F["① 界面层（src-vite · Vue 3）—— 你看到和点击的一切"]
        direction TB
        V["📄 页面<br/>首页 · 图片查看器 · 图片编辑器 · 设置"]
        COM["🧩 功能组件<br/>相册 · 日历 · 地图 · 搜索框 · 缩略图 · 标签…"]
        STORE["📦 状态管理（Pinia）<br/>记住你正在看哪个相册、选中了哪些照片"]
        API["🔌 API 桥接（api.js）<br/>把每一次点击打包成请求"]
        V --> COM --> STORE --> API
    end

    subgraph M["② 通信桥（Tauri IPC）<br/>界面和引擎之间的\"电话线\""]
        IPC["📨 命令通道<br/>请求下去 · 结果回来"]
    end

    subgraph B["③ 引擎层（src-tauri · Rust）—— 真正干活的部门"]
        direction TB
        CMD["🎛️ 命令中心（t_cmds）<br/>接到请求，分派给下面 6 个部门"]
        SQL["🗄️ 数据库引擎（t_sqlite）<br/>查照片索引、相册、标签、收藏"]
        IMG["🖼️ 图像引擎（t_image）<br/>解码 RAW / HEIC / JPEG<br/>生成缩略图 · 编辑图片"]
        VID["🎬 视频引擎（t_video）<br/>FFmpeg 转码与播放"]
        AI["🤖 AI 引擎（t_ai）<br/>文字搜照片 · 找相似图（CLIP）"]
        FACE["👤 人脸引擎（t_face）<br/>认出照片里的人并归类（InsightFace）"]
        FOP["📁 文件引擎（t_utils / t_storage）<br/>移动 · 复制 · 删除 · 导入 · 备份"]
        CMD --> SQL & IMG & VID & AI & FACE & FOP
    end

    subgraph D["④ 数据层（全部保存在你电脑本地）"]
        direction TB
        DB["🗃️ SQLite 数据库<br/>Lap 自己的索引：相册 · 标签 · 收藏 · AI 数据"]
        P["📷 你的照片和视频<br/>原文件永远留在原文件夹，不会被搬走"]
        MD["🧠 AI 模型 + 缩略图缓存<br/>搜索、人脸识别、快速预览用"]
    end

    U --> F
    API --> IPC
    IPC --> CMD
    SQL --> DB
    IMG --> P
    VID --> P
    FOP --> P
    AI --> MD
    FACE --> MD
    DB -.->|查询结果| SQL
    MD -.->|识别结果| AI

    classDef user fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef mid fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef engine fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef data fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    class U user
    class V,COM,STORE,API ui
    class IPC mid
    class CMD,SQL,IMG,VID,AI,FACE,FOP engine
    class DB,P,MD data
```

## 逐层解释（完全不懂代码也能懂）

| 层 | 是什么 | 打个比方 |
| :-- | :-- | :-- |
| ① 界面层 | 用 Vue 3 写的桌面窗口，包含首页、图片查看器、编辑器、设置等页面 | **餐厅的菜单和前台**：你点菜、看到菜品 |
| ② 通信桥 | Tauri 提供的"命令通道"，界面和引擎之间传话 | **服务员**：把你的需求端进厨房，再把菜端出来 |
| ③ 引擎层 | 用 Rust 写的 6 个功能部门：数据库、图像、视频、AI、人脸、文件管理 | **厨房里的各个厨师**：各管一摊，配合出菜 |
| ④ 数据层 | 三类本地数据：SQLite 索引库、你的照片原文件、AI 模型与缩略图缓存 | **仓库和冰箱**：食材和调料都放在自己家，不送外卖 |

## 一次点击发生了什么？（数据流示例）

> 你在搜索框输入"海边日落"并按回车：

1. **界面层**：搜索框组件把文字交给状态管理 → 打包成请求（`api.js`）
2. **通信桥**：通过 Tauri IPC 把请求传给引擎层
3. **引擎层**：命令中心把请求分给 **AI 引擎**，AI 引擎用本机的 CLIP 模型把"海边日落"变成特征向量，去 SQLite 里比对照片索引
4. **数据层**：比对结果返回 → 引擎层把命中的照片缩略图读出来 → 通过通信桥送回界面层
5. **界面层**：缩略图网格刷新，你看到结果

整个过程都在本机完成，**照片不会上传到任何云端**。

## 关键第三方库（引擎层请来的"外援"）

| 外援 | 负责什么 |
| :-- | :-- |
| LibRaw | 解码各种 RAW 相机原始格式（CR2、NEF、ARW…） |
| libheif / libde265 | 解码苹果 HEIC/HEIF 格式 |
| libjpeg-turbo | 快速解码 JPEG、生成缩略图 |
| FFmpeg | 视频处理、视频缩略图、播放兼容 |
| ONNX Runtime + CLIP | 本地 AI 推理：文字搜照片、相似照片 |
| InsightFace | 人脸检测与识别 |
| Leaflet | 照片地图与打卡热力图 |
| Video.js / daisyUI | 视频播放界面 / UI 组件库 |

## 目录速查

```
lap/ (Lumi)
├── src-tauri/          # ③ 引擎层（Rust 后端）
│   └── src/
│       ├── main.rs     # 程序入口：启动引擎、注册命令
│       ├── t_cmds.rs   # 命令中心：接收界面请求并分派
│       ├── t_sqlite.rs # 数据库引擎（最大的模块，近 10000 行）
│       ├── t_image.rs  # 图像引擎
│       ├── t_video.rs  # 视频引擎
│       ├── t_ai.rs     # AI 引擎
│       ├── t_face.rs   # 人脸引擎
│       ├── t_utils.rs  # 文件引擎（工具）
│       ├── t_storage.rs# 文件引擎（存储/备份）
│       └── ...         # 其他辅助模块（去重、相似、迁移等）
├── src-vite/           # ① 界面层（Vue 3 前端）
│   └── src/
│       ├── views/      # 4 个主页面：首页/查看器/编辑器/设置
│       ├── components/ # 55 个功能组件（相册、日历、地图、搜索…）
│       ├── stores/     # 状态管理（Pinia）
│       ├── common/     # API 桥接、路由、快捷键、工具函数
│       └── composables/# 可复用逻辑
├── scripts/            # 构建辅助：下载 AI 模型、FFmpeg
├── docs/               # 官方文档网站
├── i18n/               # 多语言 README
└── .github/            # 自动构建发布（CI/CD）
```

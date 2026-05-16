# subtask_pipeline

通用视觉任务数据集生成 pipeline。程序分两步工作：

1. `prior`：从 annotation 和图像中自动生成 autolabel 提示信息。
2. `generate_dataset.py` / `generate`：读取上一步 autolabel 提示信息，批量生成结构化 `model_response` 标注结果。

旧脚本 `api_gemini_without_wrist.py` 只作为兼容参考保留；新增主入口是 `api_subtask_auto_label.py` 和 `generate_dataset.py`。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中配置：

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_TEMPERATURE=0.2
```

命令行 `--model` 会覆盖 `.env` 中的 `GEMINI_MODEL`。

## 输入格式

单条数据由一个 annotation JSON 和一个图像根目录组成：

```text
image_root/
  stage_00/
    frame_000001.jpg
    frame_000002.jpg
  stage_01/
    frame_000120.jpg
```

annotation 示例：

```json
{
  "task_name": "complete the visual task",
  "skills": [
    {
      "skill_idx": 0,
      "skill_description": "move to the target object",
      "object_id": ["target object"],
      "manuipation_object_id": ["target object"],
      "frame_duration": [1, 120]
    }
  ],
  "valid_duration": [1, 300]
}
```

兼容字段：

```text
skills 或 skill_annotation
manuipation_object_id 或 manipulating_object_id
```

## 1. 生成 Autolabel 提示信息

每个 skill 会在 `frame_duration` 内均匀采样 `k` 帧，默认 `k=10`，不包含起始帧和终止帧。

```powershell
python api_subtask_auto_label.py prior `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --output-dir outputs\episode_0001\prior `
  --sample-k 10
```

输出：

```text
outputs/episode_0001/prior/
  subtasks/
    subtask_00_prior.json
  task_prior.json
  autolabel_prompt_info.json
```

每个 skill 的提示信息结构：

```json
{
  "skill_idx": 0,
  "skill_description": "...",
  "subtask_name": "...",
  "completion_conditions": ["..."],
  "required_visual_evidence": ["..."],
  "negative_conditions": ["..."],
  "common_false_positives": ["..."],
  "ambiguous_cases": ["..."],
  "memory_update_guidance": {
    "when_in_progress": "...",
    "when_completed": "...",
    "completed_progress_phrase_style": "short natural verb-object phrase"
  },
  "prompt_rules": ["..."]
}
```

这些字段是后续数据集生成唯一的数据特定规则来源。程序本身不再内置某个 task 的专用规则。

## 2. 通用数据集生成

推荐使用独立入口 `generate_dataset.py`：

```powershell
python generate_dataset.py `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --autolabel-json outputs\episode_0001\prior\autolabel_prompt_info.json `
  --output outputs\episode_0001\generation `
  --frame-stride 80
```

也可以用主 CLI：

```powershell
python api_subtask_auto_label.py generate `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --prompt-info-json outputs\episode_0001\prior\autolabel_prompt_info.json `
  --output outputs\episode_0001\generation `
  --frame-stride 80
```

输出仍采用当前 `model_response` 结构：

```json
{
  "model_response": {
    "reasoning": "...",
    "new_memory": {
      "Progress": "...",
      "World state": "..."
    },
    "subtask": "...",
    "current_skill_status": "in_progress",
    "visible_transition": "",
    "is_subtask_completed": false
  }
}
```

## 批量目录模式

目录结构：

```text
data/annotations/episode_0001.json
data/images/episode_0001/stage_00/...
outputs/priors/episode_0001/autolabel_prompt_info.json
```

调用：

```powershell
python generate_dataset.py `
  --annotation-json data\annotations `
  --image-root data\images `
  --autolabel-root outputs\priors `
  --output outputs\generation `
  --frame-stride 80 `
  --episode-offset 0 `
  --episode-limit 100 `
  --resume
```

`--resume` 会跳过已经完整生成的单 episode 输出。

## 一步运行 prior + generation

单条数据可直接运行：

```powershell
python api_subtask_auto_label.py run-all `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --output-dir outputs\episode_0001 `
  --sample-k 10 `
  --frame-stride 80
```

## Prompt 配置

默认 prompt 在：

```text
prompts/default_prompts.json
```

可通过 `--prompt-config` 替换。主要 key：

```text
subtask_prior_system
subtask_prior_user
parent_prior_system
parent_prior_user
generation_system
generation_user
```

## 常用参数

```text
--env-file                 指定 .env 文件
--model                    覆盖 GEMINI_MODEL
--temperature              覆盖 GEMINI_TEMPERATURE
--max-output-tokens         最大输出 token
--max-retries              API 错误重试次数
--max-response-retries      无效 JSON 响应重试次数
--request-delay             每次请求后的等待秒数
--include-previous-image    generation 时同时传入上一个采样帧
--episode-offset            批量模式跳过前 N 个 episode
--episode-limit             批量模式最多处理 N 个 episode
--resume                    跳过已有完整输出
```

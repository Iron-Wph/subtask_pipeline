# 子任务描述自动标注程序

本项目新增了一套模块化 pipeline，用于：

1. 读取一个任务的图像目录和 annotation JSON。
2. 运行两级父子 prior agent：每个子 agent 只负责一个子任务，父 agent 汇总整条任务。
3. 将 prior 结果作为提示词上下文，生成与旧脚本 `model_response` 一致的结构化标注结果。

原始 `api_gemini_without_wrist.py` 保留不动；新入口是 `api_subtask_auto_label.py`。

## 目录结构

```text
subtask_auto_labeler/
  config.py          # dotenv 和 Gemini 参数配置
  dataset.py         # annotation 读取、图像目录解析、帧采样
  gemini_client.py   # Gemini API 调用、重试、JSON 解析
  prompts.py         # prompt JSON 加载与渲染
  prior.py           # 子 agent / 父 agent prior 流程
  generation.py      # 大规模结构化标注生成流程
  cli.py             # 命令行入口
prompts/default_prompts.json
api_subtask_auto_label.py
```

## 环境安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

创建 `.env`：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
GEMINI_API_KEY=你的 Gemini API Key
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_TEMPERATURE=0.2
```

命令行里的 `--model` 会覆盖 `.env` 里的 `GEMINI_MODEL`。

## 输入数据格式

单条数据由一个 annotation JSON 和一个图像根目录组成。

图像目录保持旧脚本格式：

```text
image_root/
  stage_00/
    frame_000001.jpg
    frame_000002.jpg
  stage_01/
    frame_000120.jpg
```

annotation JSON 需要包含：

```json
{
  "task_name": "put the cup on the table",
  "skills": [
    {
      "skill_idx": 0,
      "skill_description": "move to the cup",
      "object_id": ["cup"],
      "manuipation_object_id": ["cup"],
      "frame_duration": [1, 120]
    }
  ],
  "valid_duration": [1, 300]
}
```

兼容 `skill_annotation` 字段；`manuipation_object_id` 会优先读取，也兼容旧拼法 `manipulating_object_id`。

## 1. 生成先验 prior

每个子任务会从 `frame_duration` 内部均匀采样 `k` 帧，不包含起始帧和终止帧。默认 `k=10`。

```powershell
python api_subtask_auto_label.py prior `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --output-dir outputs\episode_0001\prior `
  --sample-k 10
```

输出示例：

```text
outputs/episode_0001/prior/
  subtasks/
    subtask_00_prior.json
    subtask_01_prior.json
  task_prior.json
```

`subtask_00_prior.json` 中会包含：

```json
{
  "agent_type": "subtask_prior_agent",
  "stage_idx": 0,
  "completion_conditions": ["the target object is clearly under robot control"],
  "visual_changes": ["the gripper moves from approach pose to contact pose"],
  "false_positive_risks": ["occlusion may hide whether the object is still supported"],
  "sampled_frame_analysis": []
}
```

`task_prior.json` 是父 agent 对所有子任务先验的全局调整结果。

## 2. 基于 prior 生成结构化标注

单 episode：

```powershell
python api_subtask_auto_label.py generate `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --task-prior-json outputs\episode_0001\prior\task_prior.json `
  --output outputs\episode_0001\generation `
  --frame-stride 80
```

annotation 和 image root 均为目录时：

```powershell
python api_subtask_auto_label.py generate `
  --annotation-json data\annotations `
  --image-root data\images `
  --prior-root outputs\priors `
  --output outputs\generation `
  --frame-stride 80
```

目录模式下，程序会寻找：

```text
data/annotations/episode_0001.json
data/images/episode_0001/stage_00/...
outputs/priors/episode_0001/task_prior.json
```

生成结果采用旧脚本的 `model_response` 结构：

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

## 3. 一步运行 prior + generation

用于单条数据：

```powershell
python api_subtask_auto_label.py run-all `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --output-dir outputs\episode_0001 `
  --sample-k 10 `
  --frame-stride 80
```

输出：

```text
outputs/episode_0001/
  prior/
    subtasks/subtask_00_prior.json
    task_prior.json
  generation/
    episode_0001_generation.json
    generation_results.json
```

## Prompt 配置

所有 prompt 都在 `prompts/default_prompts.json` 中，可以复制后修改：

```powershell
python api_subtask_auto_label.py prior `
  --prompt-config prompts\my_prompts.json `
  --annotation-json data\annotations\episode_0001.json `
  --image-root data\images\episode_0001 `
  --output-dir outputs\episode_0001\prior
```

主要 prompt key：

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
```

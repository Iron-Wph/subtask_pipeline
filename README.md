# subtask_pipeline

通用视觉任务数据集生成 pipeline。程序分两步工作：

1. `prior`：从 annotation 和图像中自动生成 autolabel 提示信息。
2. `generate_dataset.py` / `generate`：读取上一步 autolabel 提示信息，批量生成结构化 `model_response` 标注结果。

旧脚本 `api_gemini_without_wrist.py` 只作为兼容参考保留；推荐使用 `api_subtask_auto_label.py` 和 `generate_dataset.py`。

## 环境安装

本项目推荐使用 conda 管理 Python 环境。

```bash
conda create -n subtask_pipeline python=3.11 -y
conda activate subtask_pipeline
pip install -r requirements.txt
python -m pip install "httpx[socks]"
```

创建并编辑环境变量文件：

```bash
cp .env.example .env
nano .env
```

`.env` 示例：

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_TEMPERATURE=0.2
```

命令行参数 `--model` 会覆盖 `.env` 中的 `GEMINI_MODEL`。

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

每个 skill 会在 `frame_duration` 内均匀采样 `k` 帧，默认 `k=10`，不包含起始帧和终止帧。从第二个采样帧开始，prior 请求会同时带上上一采样帧图像和上一轮响应，用于提取更明确的视觉状态转移。

```bash
python api_subtask_auto_label.py prior \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --output-dir outputs/episode_0001/prior \
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
  "state_transition_evidence": ["..."],
  "negative_conditions": ["..."],
  "common_false_positives": ["..."],
  "ambiguous_cases": ["..."]
}
```

这些字段是后续数据集生成唯一的数据特定规则来源。`memory` 更新格式、输出 JSON 格式、主体称谓等通用规则由 `generate_dataset.py` 的通用 prompt 固定提供，不写入每个 skill 的 prior JSON。

## 子任务描述应该包含什么

为了把 `api_gemini_without_wrist.py` 这类脚本改成通用于各个任务的数据生成程序，子任务提示信息应该只描述数据本身的可见判据，不包含通用写作规则、memory 更新模板或 prompt 规则。

建议每个 skill 至少包含：

```text
skill_idx                    子任务序号
skill_description            annotation 原始动作描述
subtask_name                 短的可执行子任务名，动词 + 目标对象
completion_conditions        完成该子任务必须满足的语义后置条件
required_visual_evidence     标 completed 前必须看到的视觉证据
state_transition_evidence    需要前后对比的视觉变化，例如红灯变绿、门从关到开
negative_conditions          明确说明未完成的视觉条件
common_false_positives       容易误判为完成但证据不足的画面
ambiguous_cases              应保守标为 no_for_sure 的情况
```

描述原则：

```text
1. 主体统一写 robot，不写 agent。
2. 图像视角是 robot 的 main/head camera view，不把 camera 写成执行动作的主体。
3. 完成条件要写结果状态，不只写动作过程。
4. 对 press / toggle / turn on / open / close 等状态变化动作，必须写清楚可见状态转移。
5. 对 pick / place 等操作动作，必须写清楚对象支撑关系、释放关系或目标位置关系。
6. 负例和误判条件要具体到可见证据，例如遮挡、反光、颜色不确定、只接触但未移动。
```

例如 `press the radio button` 这类 skill，好的描述应该包含：

```json
{
  "subtask_name": "press the radio button",
  "completion_conditions": [
    "the target radio button has been pressed and the radio's target indicator has changed to its completed state"
  ],
  "required_visual_evidence": [
    "the target button or indicator on the radio is clearly visible after the press",
    "the completed indicator state is visually distinguishable from the previous state"
  ],
  "state_transition_evidence": [
    "the target indicator changes from red to green between observations"
  ],
  "negative_conditions": [
    "the target indicator is still red",
    "the robot gripper is near the button but the indicator state has not changed"
  ],
  "common_false_positives": [
    "the button is occluded by the robot gripper",
    "a reflection or unrelated colored object looks like the target indicator"
  ],
  "ambiguous_cases": [
    "the target indicator is partly hidden, overexposed, or color-ambiguous"
  ]
}
```

## 2. 通用数据集生成

推荐使用独立入口 `generate_dataset.py`：

```bash
python generate_dataset.py \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --autolabel-json outputs/episode_0001/prior/autolabel_prompt_info.json \
  --output outputs/episode_0001/generation \
  --frame-stride 80
```

也可以用主 CLI：

```bash
python api_subtask_auto_label.py generate \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --prompt-info-json outputs/episode_0001/prior/autolabel_prompt_info.json \
  --output outputs/episode_0001/generation \
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

```bash
python generate_dataset.py \
  --annotation-json data/annotations \
  --image-root data/images \
  --autolabel-root outputs/priors \
  --output outputs/generation \
  --frame-stride 80 \
  --episode-offset 0 \
  --episode-limit 100 \
  --resume
```

`--resume` 会跳过已经完整生成的单 episode 输出。

## 一步运行 prior + generation

单条数据可直接运行：

```bash
python api_subtask_auto_label.py run-all \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --output-dir outputs/episode_0001 \
  --sample-k 10 \
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

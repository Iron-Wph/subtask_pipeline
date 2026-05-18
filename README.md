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

最终每个 skill 的列表字段默认至少生成 4 条非重复描述，可通过 `--prior-min-items` 调整。列表字段包括 `completion_conditions`、`required_visual_evidence`、`state_transition_evidence`、`negative_conditions`、`common_false_positives` 和 `ambiguous_cases`。同一字段内会要求覆盖不同角度，例如目标颜色/形状/位置/大小/数量、机器人夹爪接触或撤离、前后状态变化、遮挡、反光、运动模糊和视角歧义。

prior 阶段不是直接生成最终逐帧标签，但每个采样帧会保存轻量状态记忆：`frame_reasoning` 和 `frame_state_memory`。其中 `frame_state_memory` 记录目标部件当前状态、机器人与目标部件的空间/接触关系、相对上一采样帧的变化和不确定性。后续汇总时会用这些帧级状态信息生成更稳定的完成条件。

```bash
python api_subtask_auto_label.py prior \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --output-dir outputs/episode_0001/prior \
  --sample-k 10 \
  --prior-min-items 4
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
  "target_visual_description": {
    "target_object": "...",
    "target_part": "...",
    "color": "...",
    "shape": "...",
    "position": "...",
    "size": "...",
    "count": "..."
  },
  "completion_conditions": ["..."],
  "required_visual_evidence": ["..."],
  "state_transition_evidence": ["..."],
  "negative_conditions": ["..."],
  "common_false_positives": ["..."],
  "ambiguous_cases": ["..."],
  "generation_prompt_guidance": "natural-language Rule 16 guidance for this skill"
}
```

这些结构化字段用于保存可见判据，`generation_prompt_guidance` 是面向大规模生成阶段的自然语言第 16 条判据。它不是把 key/value 原样贴给模型，而是由子 agent 汇总采样帧后生成、再由父 agent 结合全局任务重新改写的自然规则块。`memory` 更新格式、输出 JSON 格式、主体称谓等通用规则仍由 `generate_dataset.py` 的通用 prompt 固定提供。

## 子任务描述应该包含什么

为了把 `api_gemini_without_wrist.py` 这类脚本改成通用于各个任务的数据生成程序，子任务提示信息应该只描述数据本身的可见判据，不包含通用写作规则或 memory 更新模板；其中 `generation_prompt_guidance` 是例外，它是由这些可见判据改写出来的任务特定自然语言规则，用来替换旧脚本第 16 条里的硬编码任务规则。

建议每个 skill 至少包含：

```text
skill_idx                    子任务序号
skill_description            annotation 原始动作描述
subtask_name                 短的可执行子任务名，动词 + 目标对象
target_visual_description   目标对象/部件的颜色、形状、位置、大小、数量等可见属性
completion_conditions        完成该子任务必须满足的语义后置条件
required_visual_evidence     标 completed 前必须看到的视觉证据
state_transition_evidence    需要前后对比的视觉变化，例如红灯变绿、门从关到开
negative_conditions          明确说明未完成的视觉条件
common_false_positives       容易误判为完成但证据不足的画面
ambiguous_cases              应保守标为 no_for_sure 的情况
generation_prompt_guidance   直接填入 generation 第 16 条的自然语言判据，由子 agent 生成并由父 agent 改写强化
```

描述原则：

```text
1. 主体统一写 robot，不写 agent。
2. 图像视角是 robot 的 main/head camera view，不把 camera 写成执行动作的主体。
3. 完成条件要写结果状态，不只写动作过程。
4. 对 press / toggle / turn on / open / close 等状态变化动作，必须写清楚可见状态转移。
5. 对 pick / place 等操作动作，必须写清楚对象支撑关系、释放关系或目标位置关系。
6. 负例和误判条件要具体到可见证据，例如遮挡、反光、颜色不确定、只接触但未移动。
7. 不要只写 `visible state change`、`indicator turns on`、`button is pressed` 这类泛化短语；应写清楚同一个目标部件的颜色、形状、位置和前后状态，例如 “the same small circular power button changes from red to green”。
8. 不要单独使用 `indicator`、`button`、`it` 这类指代不明的词；每条条件都应写完整目标对象和目标部件，例如 “the radio small circular power button on the top control area”。
9. `target_visual_description` 是后续字段的一致性来源。若其中写了颜色前后状态，例如 `red before completion, green after completion`，则完成条件必须写同一目标部件的最终颜色，状态转移必须写同一目标部件的前后颜色变化；不能再引入未确认的第二个灯、孔、按钮或开关。
```

例如 `press the radio button` 这类 skill，好的描述应该包含：

```json
{
  "subtask_name": "press the radio button",
  "target_visual_description": {
    "target_object": "radio",
    "target_part": "small circular power button on the top control area",
    "color": "red before completion, green after completion",
    "shape": "small round power button",
    "position": "on the top control area of the radio",
    "size": "small dot-sized button relative to the radio body",
    "count": "one target button"
  },
  "completion_conditions": [
    "the robot has pressed the radio's small circular power button and the same small circular power button is visibly green",
    "the small round power button on the radio's top control area is in the completed green state after robot contact",
    "the radio's one target power button is no longer red and is clearly green in the robot's head-camera view",
    "the robot's left gripper can retract while the same small circular button remains green"
  ],
  "required_visual_evidence": [
    "the small round power button on the radio's top control area is clearly visible",
    "the radio small circular power button on the top control area is green after the press",
    "the visible green mark belongs to the same dot-sized circular button rather than another part of the radio",
    "only one target power button is being evaluated on the radio's top control area"
  ],
  "state_transition_evidence": [
    "the same small circular power button changes from red to green between observations",
    "the radio small circular power button on the top control area is red before robot pressure and green after the press",
    "the robot gripper moves away from the radio while the radio small circular power button on the top control area remains green",
    "the color change occurs on the top control area's one target button, not on a background reflection"
  ],
  "negative_conditions": [
    "the small circular power button remains red",
    "the robot gripper is near or touching the radio small circular power button on the top control area but that same target button is not green",
    "the green color is visible on another radio part but the target circular power button is still red or hidden",
    "the target button is fully occluded, so the completed green state cannot be verified"
  ],
  "common_false_positives": [
    "the radio small circular power button on the top control area is occluded by the robot gripper",
    "a reflection or unrelated colored object looks like the target power button turning green",
    "the robot gripper presses the radio body near the control area but misses the small circular power button",
    "the target button appears darker from shadow but has not visibly changed from red to green"
  ],
  "ambiguous_cases": [
    "the radio small circular power button on the top control area is partly hidden, overexposed, or color-ambiguous",
    "motion blur around the gripper or radio makes the small circular button boundary unclear",
    "the robot's head-camera view shows the radio at an angle where the top control area is not fully visible",
    "the gripper covers the button at the moment when the red-to-green transition would need to be checked"
  ],
  "generation_prompt_guidance": "For this candidate skill, judge whether the robot has pressed the radio's one small circular power button on the top control area. Treat the skill as completed only when the same target power button is visible and green in the current image; gripper contact alone is not sufficient. Use the previous image only to compare the same button changing from red to green. Keep the skill in progress if the target button remains red, and use no_for_sure if the gripper, glare, blur, or viewpoint makes the target button color unreliable."
}
```

## 2. 通用数据集生成

generation 阶段才是逐帧结构化标注，和旧脚本 `api_gemini_without_wrist.py` 一样，每一帧都会输出 `reasoning`、`new_memory`、`current_skill_status`、`visible_transition` 和 `is_subtask_completed`。区别是现在不再把特定任务规则写死在脚本里，而是读取 prior 阶段生成的 `autolabel_prompt_info.json` 作为任务特定判据。

generation 不会把完整 `autolabel_prompt_info.json` 直接塞进 Gemini 上下文，也不会把结构化 JSON 的 key/value 原样贴进 prompt。新的流程是：子 agent 汇总每个 skill 时生成 `generation_prompt_guidance`；父 agent 收集所有子任务后，根据全局顺序、共享对象、状态承接和跨子任务误判风险，把这段内容改写成更详细、更自然的第 16 条判据。

generation 阶段优先读取父 agent 的 `generation_prompt_guidance` 并直接填入 `Use these task-specific visible postconditions as guidance`；只有旧 prior 文件缺少该字段时，程序才会用结构化字段生成自然语言回退。

内部抽取字段包括：

```text
global prior:
  task_name
  prior_min_items
  task_summary
  global_completion_order
  global_visual_adjustments
  cross_subtask_false_positive_risks

current skill prior:
  child_prior
    stage_idx / skill_idx / skill_description / subtask_name
    target_visual_description
    completion_conditions
    required_visual_evidence
    state_transition_evidence
    negative_conditions
    common_false_positives
    ambiguous_cases
    generation_prompt_guidance
  parent_adjusted_prior
    同上，但来自父 agent 的全局调整结果，其中 generation_prompt_guidance 是父 agent 改写后的自然语言判据
```

`raw_frame_requests`、采样帧分析日志、Gemini metadata 和完整子任务列表不会进入每帧请求。旧脚本里写死的任务规则被拆成了两部分：

```text
通用判断逻辑:
  保留在 generation_user prompt 的第 1-15 点和第 17-19 点。

任务特定视觉判据:
  优先使用父 agent 改写后的 `generation_prompt_guidance`，直接填入第 16 点；只有旧 prior 文件缺少该字段时，generation 才根据结构化字段生成自然语言回退。
```

第 16 点示例：

```text
16. Use these task-specific visible postconditions as guidance.

For this candidate skill, judge whether the robot has pressed the red radio's single small circular power button on the top control area. Treat the press as completed only when the same small circular power button is clearly visible in the current robot head-camera image and its visible color has changed into the completed green state. Robot gripper contact, a finger hovering over the radio, or a pressing motion is not enough by itself; the decisive evidence is the target button's final green state on the radio itself. If a previous image is available, use it only to compare the same target button changing from red before the press to green after the press. Do not use green reflections, wall lights, colored room objects, gripper colors, or other radio marks as substitutes for the radio's one target power button. Keep the skill in progress when the target button remains red or when the robot is touching the radio but the target button has not visibly turned green. Use no_for_sure when the gripper, motion blur, glare, low resolution, or viewpoint hides the target button or makes the red-vs-green color state unreliable.
```

推荐使用独立入口 `generate_dataset.py`：

```bash
python generate_dataset.py \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --autolabel-json outputs/episode_0001/prior/autolabel_prompt_info.json \
  --output outputs/episode_0001/generation \
  --frame-stride 80 \
  --save-rendered-prompts
```

如果打开 `--save-rendered-prompts`，程序会为每一次大规模生成请求额外保存一份真实渲染后的 prompt：

```text
outputs/episode_0001/generation/
  episode_0001_generation.json
  episode_0001_generation_prompts/
    request_000001_stage_00_frame_000080.json
```

每个 prompt JSON 包含：

```json
{
  "system_instruction": "...",
  "user_prompt": "...",
  "completion_guidance": "...",
  "image_path": "...",
  "previous_image_path": "...",
  "stage_idx": 0,
  "frame_number": 80
}
```

也可以用主 CLI：

```bash
python api_subtask_auto_label.py generate \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --prompt-info-json outputs/episode_0001/prior/autolabel_prompt_info.json \
  --output outputs/episode_0001/generation \
  --frame-stride 80 \
  --save-rendered-prompts
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
  --prior-min-items 4 \
  --frame-stride 80 \
  --save-rendered-prompts
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
--sample-k                  prior 阶段每个 skill 的均匀采样帧数
--prior-min-items           prior 阶段每个 skill 列表字段的最少条数，默认 4
--request-delay             每次请求后的等待秒数
--include-previous-image    generation 时同时传入上一个采样帧
--episode-offset            批量模式跳过前 N 个 episode
--episode-limit             批量模式最多处理 N 个 episode
--resume                    跳过已有完整输出
--save-rendered-prompts     保存每次 generation 请求真实发送给模型的 system/user prompt
```

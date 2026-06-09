# 子任务自动标注流水线

本项目用于通用视觉任务数据集的自动标注。流程分为两个阶段：

1. `prior`：从 `annotation` 和图像中生成每个子任务的可视化判据与提示信息。
2. `generate`：读取上一步生成的提示信息，逐帧生成结构化 `model_response` 标注结果。

旧脚本 `api_gemini_without_wrist.py` 仅作为兼容参考保留。推荐使用 `api_subtask_auto_label.py` 和 `generate_dataset.py`。

## 环境安装

推荐使用 `conda` 管理 Python 环境：

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
GEMINI_MODEL=gemini-3.5-flash
GEMINI_TEMPERATURE=0.2
```

命令行参数 `--model` 会覆盖 `.env` 中的 `GEMINI_MODEL`。

## 输入格式

单条数据由一个 `annotation` JSON 文件和一个图像根目录组成：

```text
image_root/
  stage_00/
    frame_000001.jpg
    frame_000002.jpg
  stage_01/
    frame_000120.jpg
```

`annotation` 示例：

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

## 生成先验提示信息

`prior` 阶段会在每个 `skill` 自己的 `frame_duration` 内采样。默认 `--sample-k 10` 表示在该子任务范围内均匀采样 10 帧。如果设置 `--prior-frame-stride N`，则会覆盖 `--sample-k`，改为在同一子任务范围内每隔 `N` 帧采样一次。

当前消融分支使用“结构化优先”的子 agent 设计：

- 逐帧子 agent 只输出观察字段，例如 `frame_reasoning`、`objects`、`robot_state`、`scene_context`、`skill_relevant_observations`、`temporal_change_from_previous` 和 `uncertainty`。
- 逐帧子 agent 不直接输出完成条件、完成门槛、误判规则、状态标签或 `generation_prompt_guidance`。
- 子任务汇总 agent 只接收第一帧采样图像/响应和最后一帧采样图像/响应。
- 中间帧响应仍保存在 `raw_frame_requests` 中，便于离线检查，但当前分支不会把中间帧注入汇总 prompt。
- 汇总 agent 通过首尾观察对比生成 `completion_gates`、`completion_conditions`、`required_visual_evidence`、`not_sufficient_for_completion`、`common_false_positives`、`ambiguous_cases` 和 `generation_prompt_guidance`。

运行命令：

```bash
python api_subtask_auto_label.py prior \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --output-dir outputs/episode_0001/prior \
  --sample-k 10 \
  --prior-frame-stride 30 \
  --prior-min-items 4 \
  --parent-prior-attempts 3
```

输出目录：

```text
outputs/episode_0001/prior/
  subtasks/
    subtask_00_prior.json
  task_prior.json
  autolabel_prompt_info.json
  prior_checkpoint.json
```

`autolabel_prompt_info.json` 是 `generate_dataset.py` 的首选输入。`task_prior.json` 是同一份 prior 信息的主文件；正常情况下两者内容一致。

父 agent 审查阶段会按 `skill` 拆分。每个 `child prior` 会单独请求一个 `parent review agent`。`--parent-prior-attempts` 表示单个 `skill` 的父 agent 审查最大重试轮数，默认 3 轮；每一轮内部仍会按 `--max-response-retries` 处理无效 JSON 修复。

如果某个 `skill` 的父 agent 多轮后仍返回无效 JSON 或 API 报错，程序只会给该 `skill` 写入 `review_failed_child_fallback`，并继续保存可用于 generation 的 `task_prior.json` 和 `autolabel_prompt_info.json`。此时总状态为 `partial_skill_review_fallback`，`usable_for_generation` 为 `true`。该 fallback 仍可用于 generation，因为逐帧 generation 默认只读取 `child prior` 的 `generation_prompt_guidance` 和结构化 guardrails，父 agent 审查字段不会作为 generation 规则注入。

## 子任务先验字段

每个 `skill` 的提示信息结构如下：

```json
{
  "skill_idx": 0,
  "skill_description": "...",
  "skill_type_hypothesis": "move_to | object_acquisition | object_placement | state_change_press_toggle | open_close | push_pull_slide | pour_transfer_insert_remove | other",
  "target_binding": {
    "target_object": "...",
    "target_part": "...",
    "manipulated_object": "...",
    "support_surface": "...",
    "target_location": "...",
    "robot_effector": "...",
    "visible_attributes": ["..."],
    "count": "..."
  },
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
  "pre_completion_state": ["..."],
  "in_progress_state": ["..."],
  "completion_gates": ["..."],
  "completion_conditions": ["..."],
  "required_visual_evidence": ["..."],
  "state_transition_evidence": ["..."],
  "negative_conditions": ["..."],
  "not_sufficient_for_completion": ["..."],
  "common_false_positives": ["..."],
  "ambiguous_cases": ["..."],
  "generation_prompt_guidance": "..."
}
```

字段含义：

```text
skill_idx                       子任务编号。
skill_description               annotation 中的原始动作描述。
skill_type_hypothesis           子 agent 推断的动作类型。
target_binding                  当前子任务绑定的目标物体、目标部件、被操作物体、支撑面、目标位置、机械臂和可见属性。
subtask_name                    简短可执行的子任务名称。
target_visual_description       目标的可见属性，例如颜色、形状、位置、大小和数量。
pre_completion_state            完成前可见状态，会作为子任务结构化 guardrail 加入 Rule 16。
in_progress_state               进行中但不能判定完成的状态。
completion_gates                判定 completed 前必须同时满足的决定性可见门槛。
completion_conditions           子任务语义完成条件。
required_visual_evidence        判定完成所需的直接视觉证据。
state_transition_evidence       需要时间对比时的前后状态变化。
negative_conditions             明确表示未完成的可见条件。
not_sufficient_for_completion   不足以判定完成的中间模式。
common_false_positives          容易被误判为完成的视觉模式。
ambiguous_cases                 应保守输出 no_for_sure 的情况。
generation_prompt_guidance      子 agent 生成的主要自然语言 Rule 16 规则。
```

这些字段保存的是子 agent 的视觉判据。`generation_prompt_guidance` 是首尾采样帧汇总后的主要自然语言规则，不是原始键值对转储。父 agent 只作为审查和质检 annotator；父 agent 输出字段会保存用于检查，但不会注入逐帧 generation prompt。

为了避免检查 `model_response.skills` 时丢失信息，每个父 agent 审查结果中也会包含 `child_prior_snapshot`，其中复制了子 prior 的关键完成条件、负例、`no_for_sure`、误判和 guidance 字段。

通用记忆格式、输出 JSON 格式、actor 命名规则和状态规则仍来自共享的 generation prompt。

## 先验约束层级

`prior` 生成使用三层约束：

1. `universal_visual_rubric`：定义直接可见证据、目标一致性和保守判断原则。
2. `action_primitive_rubric`：定义通用机器人动作类型，例如移动、拾取、放置、按压、打开和关闭。
3. `prior_review_rubric`：要求父 agent 审查子 agent 的弱判据和误判风险。

大规模 generation 不会直接加载这些通用 rubric。它只读取当前 `skill` 的子 `generation_prompt_guidance` 和子结构化 guardrails。

对于状态变化类动作，例如 `press`、`toggle`、`switch`、`turn on` 和 `turn off`，prior 阶段必须分离“稳定物理目标身份”和“目标最终状态”。例如，应表达为“同一个小圆形电源按钮从红色/关闭状态变为绿色/开启状态”，而不是把目标直接命名为“绿色按钮”。如果当前图像中同一个按钮仍是红色，应判定为同一目标部件未完成，而不是说“绿色按钮不可见”。

## 当前消融分支

相关分支如下：

```text
codex/schema-summary-endpoints
  基线实验分支。子 agent 输出逐帧观察 schema。汇总 agent 只接收首尾采样帧响应和图像。generation 使用子 generation_prompt_guidance、子结构化 guardrails 和额外通用动作 guardrails。

codex/ablate-open-close-constraints
  只移除显式 open/close 专用约束。pick/lift 和 state-change 通用 guardrails 仍在 generation 中生效。

codex/clean-generation-ablation
  当前干净 generation 消融分支。generation 不注入硬编码动作类型 guardrail 块。Rule 16 使用任务上下文、子 generation_prompt_guidance、子结构化 guardrails 和共享通用 generation prompt 规则。
```

## 子任务描述原则

为了让程序通用于不同任务，子任务提示信息应只描述数据本身的可见判据，不应包含通用写作规则或 memory 更新模板。`generation_prompt_guidance` 是例外，它是由可见判据改写出的任务特定自然语言规则，用来替换旧脚本第 16 条中的硬编码任务规则。

描述原则：

```text
1. 主体统一写 robot，不写 agent。
2. 图像视角是 robot 的 main/head camera view，不把 camera 写成动作主体。
3. 完成条件要写结果状态，不只写动作过程。
4. 对 press、toggle、turn on、open、close 等状态变化动作，必须写清楚可见状态转移。
5. 对 pick、place 等操作动作，必须写清楚对象支撑关系、释放关系或目标位置关系。
6. 负例和误判条件要具体到可见证据，例如遮挡、反光、颜色不确定、只接触但未移动。
7. 不要只写 visible state change、indicator turns on、button is pressed 这类泛化短语；应写清楚同一个目标部件的颜色、形状、位置和前后状态。
8. 不要单独使用 indicator、button、it 这类指代不明的词；每条条件都应写完整目标对象和目标部件。
9. target_visual_description 是后续字段的一致性来源。如果其中写了颜色前后状态，则完成条件和状态转移必须写同一目标部件的前后状态，不能再引入未确认的第二个灯、孔、按钮或开关。
10. skill_description、object_id、manuipation_object_id 定义当前子任务的原子动作和原子操作对象。不要因为画面里出现按钮、开关、把手、指示灯或控制面板，就把 open/close 改写成 press/switch，或把这些非目标机制写成当前子任务目标。
```

## 生成阶段运行

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

也可以使用统一 CLI：

```bash
python api_subtask_auto_label.py generate \
  --annotation-json data/annotations/episode_0001.json \
  --image-root data/images/episode_0001 \
  --prompt-info-json outputs/episode_0001/prior/autolabel_prompt_info.json \
  --output outputs/episode_0001/generation \
  --frame-stride 80 \
  --save-rendered-prompts
```

`--frame-stride` 会复用旧脚本的采样方式：先读取 `annotation` JSON 中的 `valid_duration`，从第一个有效帧开始按 `range(valid_start, valid_end, frame_stride)` 取帧；每个采样帧再根据各 `skill` 的 `frame_duration` 判断属于哪个 `skill`，并从对应 `stage_xx/frame_*.jpg` 或 `skill_xx/frame_*.jpg` 目录读取同名图像。

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

## 第 16 条规则注入内容

Generation 不会把完整 `autolabel_prompt_info.json` 粘贴进 Gemini，也不会把原始结构化 JSON 键值对直接粘贴进 prompt。

当前流程是：

```text
1. 子 agent 生成主要 generation_prompt_guidance。
2. 父 agent 返回 QA 审查字段，只保存到 JSON，不作为 generation-time 规则使用。
3. generation 读取子 generation_prompt_guidance。
4. generation 追加 pre_completion_state、in_progress_state、completion_gates 和 not_sufficient_for_completion 作为子结构化 guardrails。
5. 如果旧 prior 文件缺少子 generation_prompt_guidance，代码会从子结构化字段渲染自然语言 guidance。
```

当开启 `--save-rendered-prompts` 时，保存的 prompt 中会看到这些渲染后的标题：

```text
Primary child-agent skill guidance for the current candidate skill
Child-agent structured visual state guardrails
```

这些标题由 `generation.py` 生成，不是 `autolabel_prompt_info.json` 中的字面 JSON key。

Generation 还会在通用 prompt 中加入 atomic skill scope 约束：`subtask` 和 `new_memory.Progress` 必须沿用当前候选 `skill` 的原子动作和目标对象，不能把可见但未标注的机制动作写成当前任务。例如当前 `skill` 是 `open the microwave door` 时，即使机械爪靠近红色释放按钮，Progress 也应写“robot is trying to open the microwave door, but the door remains closed/unclear”，而不是“robot is pressing the red release button to open the door”。按钮可以出现在 `World state` 中作为客观场景信息，但不能成为当前 `subtask` 或完成证据。

## 通用硬性完成规则

`generation.py` 目前只对 move-to / navigation 做确定性硬规则。这条规则来自采样边界本身，不依赖某个具体任务：

```text
move-to / navigation skill:
  同一个 move-to skill 的非最后一次采样请求不允许输出 completed 或 completed_and_transitioning。
  即使当前图像看起来已经接近目标，也只能输出 in_progress 或 no_for_sure。
  该 skill 的最后一次采样请求会被固定为 completed。
  is_subtask_completed 会被固定为 true。
```

这条规则的目的，是把 move-to skill 的完成边界固定在该 skill 的最后一次 generation 请求上，避免中间帧过早写入完成状态，也避免最后一帧被模型保守地写成 `in_progress`。

## 时间一致性后处理

第一轮生成完成后，程序会按每个 `skill` 片段运行时间一致性校验。

如果某个 `skill` 的所有响应都没有 `is_subtask_completed = true`，程序会保留原始模型输出，不会重复请求最后一帧，也不会强制补 completed。这些片段只记录在：

```text
temporal_validation.missing_completion_backfill.preserved_stages
```

如果某个 `skill` 中间出现过 completed，但该 completed 不属于末尾连续完成后缀，则该 completed 会被重复请求，并强制改为未完成。例如：

```text
T F F ... F  ->  F F F ... F
F T F T T    ->  F F F T T
F T T        ->  F T T
```

重复请求记录会保留原始响应：

```text
temporal_retry.original_model_response
temporal_retry.repeated = true
```

`result_used` 表示最终标签是否保留在生成结果中。`no_for_sure` 记录会保留，`result_used = true`，但 `model_response.is_subtask_completed` 仍为 `false`。`context_source_used` 保持为 `true`，因为该记录仍属于原始在线 memory/context 链；temporal retry 不会重跑后续帧。

## 保存渲染后的提示词

如果打开 `--save-rendered-prompts`，程序会为每次大规模生成请求额外保存两份真实渲染后的 prompt：

```text
outputs/episode_0001/generation/
  episode_0001_generation.json
  episode_0001_generation_prompts/
    request_000001_stage_00_frame_000080.json
    request_000001_stage_00_frame_000080.md
```

`.md` 文件会分块显示：

```text
Request Metadata
Prompt Values Injected Into Template
Completion Guidance Injected As Rule 16
System Instruction
Full User Prompt
```

这样可以直接检查 `task_name`、`skill_description`、`object_id`、`frame_duration`、`completion_guidance` 等字段如何进入最终 prompt。

## 批量目录模式

目录结构示例：

```text
data/annotations/episode_0001.json
data/images/episode_0001/stage_00/...
outputs/priors/episode_0001/autolabel_prompt_info.json
```

运行命令：

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

`--resume` 会跳过已经完整生成的单个 episode 输出。

## 中断和报错保存

长时间运行时，程序会在键盘中断、模型请求超时、无效 JSON、API 报错或其他异常退出前尽量保存已完成结果。

### 先验阶段

如果中断发生在子任务逐帧 prior 请求中，当前子任务文件会保存为 partial checkpoint：

```text
outputs/episode_0001/prior/subtasks/subtask_XX_prior.json
```

其中包含：

```text
status                    interrupted 或 failed
expected_sample_count      当前 skill 原计划采样请求数
processed_sample_count     已成功完成的采样请求数
current_request            中断时正在处理的 stage_idx、skill_idx、frame_number 和 image_path
error                      异常类型、错误消息、是否键盘中断和 traceback
raw_frame_requests         已完成的逐帧 Gemini 观察 schema 响应
sampled_frame_analysis     已完成帧的物体、机械臂和场景观察摘要
frame_observation_schemas  已完成帧的结构化观察列表
```

如果中断发生在子任务汇总请求中，同一个 `subtask_XX_prior.json` 会保留 frame-level preliminary prior 和错误信息。如果中断发生在父 agent 审查阶段，`task_prior.json` 会保存已完成的全部子任务结果和父阶段错误信息。

如果不是键盘中断，而是父 agent 返回无效 JSON、请求超时或 API 报错，程序会对当前 `skill` 按 `--parent-prior-attempts` 重试父 agent 审查请求。默认每个 `skill` 3 轮，每轮内部仍使用 `--max-response-retries`。如果某个 `skill` 多轮后仍失败，程序不会丢弃已完成的子 agent 结果，也不会阻断其他 `skill`；它会保存一个可用于 generation 的 per-skill review fallback。

整个 prior pipeline 还会额外保存：

```text
outputs/episode_0001/prior/prior_checkpoint.json
```

该文件用于快速查看当前停在第几个 `skill`、已经完成哪些子任务，以及错误来源。

### 生成阶段

如果中断发生在逐帧标注 generation 中，当前 episode 输出文件会保存为 partial checkpoint：

```text
outputs/episode_0001/generation/episode_0001_generation.json
```

其中包含：

```text
status            interrupted 或 failed
expected_count    当前 episode 原计划 generation 请求数
processed_count   已成功完成的请求数
current_request   中断时正在处理的 request_index、stage_idx、frame_number 和 image_path
error             异常类型、错误消息、是否键盘中断和 traceback
results           已完成的 model_response 标注结果
```

批量运行时，聚合文件也会保存当前进度：

```text
outputs/generation/generation_results.json
```

`--resume` 只会跳过 `status = complete` 且 `processed_count == expected_count` 的完整 episode。`status = interrupted` 或 `status = failed` 的 partial 文件不会被当作完成结果跳过；再次运行时会重新生成该 episode。

## 一步运行先验和生成

单条数据可以直接运行：

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

## 提示词配置

默认提示词配置文件：

```text
prompts/default_prompts.json
```

可以通过 `--prompt-config` 替换。主要 key：

```text
subtask_prior_system
subtask_prior_user
subtask_prior_summary_system
subtask_prior_summary_user
parent_prior_system
parent_prior_user
generation_system
generation_user
```

## 常用参数

```text
--env-file                  指定 .env 文件
--model                     覆盖 GEMINI_MODEL
--temperature               覆盖 GEMINI_TEMPERATURE
--max-output-tokens          最大输出 token 数
--max-retries               API 错误重试次数
--max-response-retries       无效 JSON 响应重试次数
--sample-k                  prior 阶段每个 skill 的均匀采样帧数
--prior-frame-stride         prior 阶段每个 skill 内的采样步长，会覆盖 --sample-k
--prior-min-items            prior 阶段每个 skill 列表字段的最少条数，默认 4
--parent-prior-attempts      单个 skill 的 parent review 请求重试轮数，默认 3
--request-delay              每次请求后的等待秒数
--include-previous-image     generation 时同时传入上一个采样帧
--episode-offset             批量模式跳过前 N 个 episode
--episode-limit              批量模式最多处理 N 个 episode
--resume                     跳过已有完整输出
--save-rendered-prompts      保存每次 generation 请求真实发送给模型的 system/user prompt
```

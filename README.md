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
GEMINI_MODEL=gemini-3.5-flash
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

Prior generation samples each skill inside its own `frame_duration`. By default, `--sample-k 10` uniformly selects internal frames for that skill. If `--prior-frame-stride N` is set, it overrides `--sample-k` and samples every N frames inside the same skill range.

This experimental branch uses a schema-first child agent. Frame-level requests output observation schema only: `frame_reasoning`, `objects`, `robot_state`, `scene_context`, `skill_relevant_observations`, `temporal_change_from_previous`, and `uncertainty`. They do not directly output `completion_conditions`, `completion_gates`, false-positive rules, status labels, or `generation_prompt_guidance`.

The child-summary agent receives only the first sampled-frame image/response and the last sampled-frame image/response. Intermediate frame responses remain saved in `raw_frame_requests` for offline inspection, but they are not injected into the summary prompt in this branch. The summary agent compares the start and end observations to produce `completion_gates`, `completion_conditions`, `required_visual_evidence`, `not_sufficient_for_completion`, `common_false_positives`, `ambiguous_cases`, and `generation_prompt_guidance`.

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

输出：

```text
outputs/episode_0001/prior/
  subtasks/
    subtask_00_prior.json
  task_prior.json
  autolabel_prompt_info.json
```

`autolabel_prompt_info.json` 是 `generate_dataset.py` 的首选输入。`task_prior.json` 是同一份 prior 信息的主文件；正常情况下两者内容一致。父 agent 审查阶段会按 `--parent-prior-attempts` 做多轮完整请求，默认 3 轮；每一轮完整请求内部仍然会按 `--max-response-retries` 处理无效 JSON 修复。如果父 agent 多轮后仍返回无效 JSON 或 API 报错，但所有子 agent prior 已经完成，程序会自动写出 child-only fallback 的 `task_prior.json` 和 `autolabel_prompt_info.json`，其中 `status` 为 `parent_failed_child_fallback`，`usable_for_generation` 为 `true`。这种 fallback 仍可用于 generation，因为逐帧 generation 默认只读取 child prior 的 `generation_prompt_guidance` 和结构化 guardrails，父 agent 审查字段不作为 generation 规则注入。

Prompt information structure for each skill:

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
  "generation_prompt_guidance": "natural-language Rule 16 guidance for this skill"
}
```

These structured fields store the child agent's visual criteria. `generation_prompt_guidance` is the child agent's main natural-language Rule 16 guidance after sampled-frame consolidation; it is not a raw key/value dump. The parent agent acts as a reviewer and QA annotator only; parent review fields are saved for inspection and are not injected into frame-level generation prompts. To avoid information loss when inspecting `model_response.skills`, each parent review skill also includes a `child_prior_snapshot` copied from the child prior with the key completion, negative, no_for_sure, false-positive, and guidance fields. Generic memory format, output JSON format, actor naming, and status rules still come from the shared `generate_dataset.py` prompt.

Prior generation uses three levels of constraints: `universal_visual_rubric` defines direct visible evidence and target consistency; `action_primitive_rubric` defines generic robot action primitives such as move, pick, place, press, and open/close; `prior_review_rubric` asks the parent agent to audit weak child-agent criteria and false-positive risks for offline review. Large-scale generation does not load these generic rubrics directly; it loads only the child `generation_prompt_guidance` and child structured guardrails for the current skill.

## Ablation Branches

- `codex/schema-summary-endpoints`: baseline experimental branch. Child agents output frame-level observation schemas. The summary agent receives only the first and last sampled frame responses/images. Generation uses child `generation_prompt_guidance`, child structured guardrails, and extra generic action-primitive guardrails.
- `codex/ablate-open-close-constraints`: removes only explicit open/close-specific constraints. Pick/lift and state-change primitive guardrails are still active during generation.
- `codex/clean-generation-ablation`: clean generation ablation. Generation does not inject any hard-coded action-primitive guardrail block. Rule 16 uses task context, child `generation_prompt_guidance`, child structured guardrails, and shared generic generation prompt rules.

## 子任务描述应该包含什么

为了把 `api_gemini_without_wrist.py` 这类脚本改成通用于各个任务的数据生成程序，子任务提示信息应该只描述数据本身的可见判据，不包含通用写作规则或 memory 更新模板；其中 `generation_prompt_guidance` 是例外，它是由这些可见判据改写出来的任务特定自然语言规则，用来替换旧脚本第 16 条里的硬编码任务规则。

建议每个 skill 至少包含：

```text
skill_idx                       Subtask index.
skill_description               Original action description from annotation.
skill_type_hypothesis           Child-agent action primitive, such as move_to / object_acquisition / object_placement / state_change_press_toggle.
target_binding                  Binding for object, part, manipulated object, support surface or target location, robot effector, visible attributes, and count.
subtask_name                    Short executable subtask name: verb + target object.
target_visual_description       Visible target attributes such as color, shape, position, size, and count.
pre_completion_state            Visible state before completion; added to generation Rule 16 as child structured guardrails.
in_progress_state               Progress states that must not be marked completed; added to generation Rule 16 as child structured guardrails.
completion_gates                Decisive visible gates that must all hold before completed is allowed.
completion_conditions           Semantic postconditions for the skill.
required_visual_evidence        Direct visual evidence required before marking completed.
state_transition_evidence       Before/after visual changes when temporal comparison is relevant.
negative_conditions             Visible conditions that mean the skill is not completed.
not_sufficient_for_completion   Intermediate patterns that are not enough, such as reaching, hovering, touching, occlusion, or edge-only contact.
common_false_positives          Visual patterns that may look completed but are insufficient.
ambiguous_cases                 Cases that should be handled conservatively as no_for_sure.
generation_prompt_guidance      Main natural-language Rule 16 guidance generated by the child agent.
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
10. `skill_description`、`object_id`、`manuipation_object_id` 定义当前子任务的原子动作和原子操作对象。不要因为画面里出现按钮、开关、把手、指示灯或控制面板，就把 open/close 改写成 press/switch，或把这些非目标机制写成当前子任务目标；它们只能作为误判项、非充分条件或场景状态出现。
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

Generation does not paste the full `autolabel_prompt_info.json` into Gemini, and it does not paste raw structured key/value JSON into the prompt. The flow is now: the child agent generates the main `generation_prompt_guidance`; the parent agent returns QA review fields that are saved in JSON but not used as generation-time rules.

Parent review output should not be used directly as Rule 16 guidance. It is intentionally an audit record. The detailed generation criteria remain in `subtask_priors` and are mirrored under each parent review item's `child_prior_snapshot`.

Generation reads the child `generation_prompt_guidance`, then appends `pre_completion_state`, `in_progress_state`, `completion_gates`, and `not_sufficient_for_completion` as child structured guardrails. If an old prior file lacks child `generation_prompt_guidance`, the code falls back to rendering natural guidance from the child structured fields.

When `--save-rendered-prompts` is enabled, the saved prompt shows these rendered section titles. They are generated by `generation.py`; they are not literal JSON keys in `autolabel_prompt_info.json`.

Generation 还会在通用 prompt 里加入 atomic skill scope 约束：`subtask` 和 `new_memory.Progress` 必须沿用当前 candidate skill 的原子动作和目标对象，不能把可见但未标注的机制动作写成当前任务。例如当前 skill 是 `open the microwave door` 时，即使机器人夹爪靠近红色释放按钮，Progress 也应写“robot is trying to open the microwave door, but the door remains closed/unclear”，而不是“robot is pressing the red release button to open the door”。按钮可以出现在 `World state` 中作为客观场景信息，但不能成为当前 subtask 或完成证据。

```text
Primary child-agent skill guidance for the current candidate skill:
  Comes from child_prior.generation_prompt_guidance.

Child-agent structured visual state guardrails:
  Comes from child_prior.pre_completion_state,
  child_prior.in_progress_state,
  child_prior.completion_gates,
  and child_prior.not_sufficient_for_completion.
```

### 通用硬性完成规则

`generation.py` 目前只对 move-to / navigation 做确定性硬规则。这条规则来自采样边界本身，不依赖某个具体 task：

```text
move-to / navigation skill:
  同一个 move-to skill 的非最后一次采样请求不允许输出 completed 或 completed_and_transitioning。
  即使当前图像看起来已经接近目标，也只能输出 in_progress 或 no_for_sure。
  该 skill 的最后一次采样请求会被固定为 completed，
  并且 is_subtask_completed 会被固定为 true。
```

这条规则的目的，是把 move-to skill 的完成边界固定在该 skill 的最后一次 generation 请求上，避免中间帧过早写入完成状态，也避免最后一帧被模型保守地写成 `in_progress`。它对所有任务通用，不依赖具体物体类别或 task 名称。

### Rule 16 通用视觉判据

pick-up、grasp、lift、press、toggle、switch、turn-on、turn-off、open、close 这类动作不做输出后的文本硬检查，也不通过 `Completion gate context` 强行改写结果。它们的通用要求会作为自然语言视觉判据追加到 generation prompt 的第 16 条 Rule 16 中，由模型结合当前图像和 memory 判断。

当前 Rule 16 会追加的通用视觉判据包括：

```text
pick-up / grasp / lift / object-acquisition skill:
  只有当前图像清楚显示机器人控制目标物体本体，
  且目标物体本体明确离开原始支撑面/容器/地面/固定位置，才应判断为 completed。
  伸手、触碰、夹住顶部/边缘/侧面/把手/开口、刚开始闭合夹爪、物体被挤压变形、局部遮挡、倾斜、旋转、滑动，都不是拿起完成的充分证据。
  对所有物体都应看到下半部分、底部或原始支撑接触边界已经离开支撑面，或者整个目标物体本体已经被清楚地带离支撑位置。
  在玻璃、镜面、透明、金属、深色或强反光台面上，阴影、反射、高光、镜像、暗线、透视错觉或很小的疑似缝隙不能作为离开台面的正证据。
  如果底部、下半部分、支撑接触边界或离台区域被夹爪、物体、台面边缘、容器壁、反光、眩光、模糊或视角遮住，应输出 no_for_sure，而不是 completed。

press / toggle / switch / turn-on / turn-off / state-change skill:
  只有当前图像中同一个目标按钮/开关/指示灯/显示部件清楚可见，并且明确显示最终颜色、亮灭、凹陷、姿态或开闭状态，才应判断为 completed。
  机械臂接触、正在按下、夹爪仍压在按钮上、任务顺序或 memory 都不能替代目标部件最终状态的可见证据。
  如果目标部件被夹爪遮挡、只露出边缘或一小块、颜色正在红绿过渡、颜色混合/过曝/阴影/模糊/反光，或可能来自附近灯光、墙上信号、炉灶按钮、反射或非目标标记，应输出 no_for_sure 或 in_progress。
  不能声称 red-to-green 或 off-to-on transition，除非当前图像清楚看到同一个目标部件已经处于最终状态。

open / close skill:
  只根据当前 skill 绑定的同一个可活动部件判断，例如门、盖子、抽屉、面板、翻盖、门边、铰链侧、把手侧或开口区域。
  open 需要看到该部件相对框架分离，并且门/盖/面板角度清楚显示打开状态。
  如果 skill 或 child prior 语义是 fully open / open completely / open wide enough for access，
  必须看到较大的稳定开口并且内部或目标区域可访问；只开一条缝、释放门锁、门体半开都不是完成。
  透过透明门窗、反射或前面板看到内部物体，不等于门/盖/面板已经打开。
  close 需要看到该部件与框架齐平或对齐，并且开口不再可见。
  抓住把手、触碰门面、靠近边缘、按下附近按钮/开关/指示灯、看到灯光或颜色变化，都不能作为 open/close 的完成证据，除非 annotation 明确把这些控件作为当前 skill 的目标部件。
  如果门边、铰链、开口、把手侧缝隙、实际开口区域或闭合缝被遮挡、裁切、反光、模糊或几何关系不清楚，应输出 no_for_sure，而不是 completed。
```

实现位置：

```text
subtask_auto_labeler/generation.py
  build_completion_gate_context()
    只负责 move-to / navigation 的采样边界硬规则。

  apply_hard_completion_gates()
    非最后一次 move-to 请求会被确定性改成 in_progress / false；
    最后一次 move-to 请求会被确定性改成 completed / true。

  is_move_to_skill()
    判断当前 skill 是否属于 move-to / navigation 类动作。



```


`--frame-stride` 会复用旧 `api_gemini_without_wrist.py` 的采样方式：先读取 annotation JSON 中的 `valid_duration`，从第一个有效帧开始按 `range(valid_start, valid_end, frame_stride)` 取帧；每个采样帧再根据各 skill 的 `frame_duration` 判断属于哪个 skill，并从对应 `stage_xx/frame_*.jpg` 或 `skill_xx/frame_*.jpg` 目录读取同名图像。因此请确认 `--image-root` 指向包含这些逐帧图像的目录，例如 `.../new_frame_files/task-0000/episode_00000010`。

Internal fields used by generation:

```text
global prior:
  task_name
  prior_min_items
  task_summary

current skill prior:
  child_prior
    stage_idx / skill_idx / skill_description / skill_type_hypothesis / subtask_name
    target_binding / target_visual_description
    pre_completion_state / in_progress_state / completion_gates
    completion_conditions / required_visual_evidence / state_transition_evidence
    negative_conditions / not_sufficient_for_completion
    common_false_positives / ambiguous_cases
    generation_prompt_guidance
```

`raw_frame_requests`, sampled-frame analysis logs, Gemini metadata, the full child prior list, and parent review fields are not sent to each frame request.

```text
Generic judgment logic:
  Stays in generation_user rules 1-15 and 17-19.

Task-specific visual criteria:
  Uses child `generation_prompt_guidance` as the main Rule 16 criteria and appends child structured guardrails only.
```

Rule 16 example:

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

如果打开 `--save-rendered-prompts`，程序会为每一次大规模生成请求额外保存两份真实渲染后的 prompt：一份 `.json` 给程序读，一份同名 `.md` 给人检查。

```text
outputs/episode_0001/generation/
  episode_0001_generation.json
  episode_0001_generation_prompts/
    request_000001_stage_00_frame_000080.json
    request_000001_stage_00_frame_000080.md
```

`.md` 文件会把内容分块显示：

```text
Request Metadata
Prompt Values Injected Into Template
Completion Guidance Injected As Rule 16
System Instruction
Full User Prompt
```

这样可以直接看出 `task_name`、`skill_description`、`object_id`、`frame_duration`、`completion_guidance` 等字段分别如何进入最终 prompt，而不是只看到 JSON 字符串转义后的长文本。


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

## 中断和报错保存

长时间运行时，程序会在用户键盘中断、模型请求超时、无效 JSON、API 报错或其它异常退出前尽量保存已完成结果。

### prior 阶段

如果中断发生在子任务 frame-level prior 请求中，当前子任务文件会被保存为 partial checkpoint：

```text
outputs/episode_0001/prior/subtasks/subtask_XX_prior.json
```

其中包含：

```text
status                  interrupted 或 failed
expected_sample_count    当前 skill 原计划采样请求数
processed_sample_count   已成功完成的采样请求数
current_request          中断时正在处理的 stage_idx / skill_idx / frame_number / image_path
error                    异常类型、错误消息、是否键盘中断、traceback
raw_frame_requests       已完成的 frame-level Gemini 观测 schema 响应
sampled_frame_analysis   已完成帧的物体/机械臂/场景观测摘要
frame_observation_schemas 已完成帧的结构化观测列表，用于 child-summary agent 全局生成完成条件
```

如果中断发生在子任务汇总请求中，同一个 `subtask_XX_prior.json` 会保留 frame-level preliminary prior 和错误信息。如果中断发生在父 agent 审查阶段，`task_prior.json` 会保存已完成的全部子任务结果和父阶段错误信息。

如果不是键盘中断，而是父 agent 自身返回无效 JSON、请求超时或 API 报错，程序会先按 `--parent-prior-attempts` 重试父 agent 完整请求。默认 3 轮父 agent 请求，每轮内部仍会使用 `--max-response-retries`。如果多轮后仍失败，程序不会丢弃已完成的子 agent 结果；它会保存一个可用于 generation 的 child-only fallback：

```text
outputs/episode_0001/prior/task_prior.json
outputs/episode_0001/prior/autolabel_prompt_info.json
```

该文件会包含：

```text
status                  parent_failed_child_fallback
usable_for_generation   true
parent_prior_attempts   父 agent 最大完整请求轮数
parent_attempt_count    实际父 agent 请求轮数
parent_errors           每轮失败的异常类型、错误消息和 traceback
subtask_priors          已完成的 child prior 结果
```

这种情况下后续仍然优先把 `autolabel_prompt_info.json` 传给 `generate_dataset.py`。`prior_checkpoint.json` 只用于定位中断进度和错误来源，不作为常规 generation 输入。

整个 prior pipeline 还会额外保存：

```text
outputs/episode_0001/prior/prior_checkpoint.json
```

这个文件用于快速查看当前停在第几个 skill、已经完成了哪些子任务，以及错误来源。

### generation 阶段

如果中断发生在逐帧标注 generation 中，当前 episode 输出文件会保存为 partial checkpoint：

```text
outputs/episode_0001/generation/episode_0001_generation.json
```

其中包含：

```text
status            interrupted 或 failed
expected_count    当前 episode 原计划 generation 请求数
processed_count   已成功完成的请求数
current_request   中断时正在处理的 request_index / stage_idx / frame_number / image_path
error             异常类型、错误消息、是否键盘中断、traceback
results           已完成的 model_response 标注结果
```

批量运行时，聚合文件也会保存当前进度：

```text
outputs/generation/generation_results.json
```

`--resume` 只会跳过 `status=complete` 且 `processed_count == expected_count` 的完整 episode。`status=interrupted` 或 `status=failed` 的 partial 文件不会被当作完成结果跳过；再次运行时会重新生成该 episode。

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
--parent-prior-attempts     父 agent 完整请求重试轮数，默认 3；全部失败后写 child-only fallback
--request-delay             每次请求后的等待秒数
--include-previous-image    generation 时同时传入上一个采样帧
--episode-offset            批量模式跳过前 N 个 episode
--episode-limit             批量模式最多处理 N 个 episode
--resume                    跳过已有完整输出
--save-rendered-prompts     保存每次 generation 请求真实发送给模型的 system/user prompt
```

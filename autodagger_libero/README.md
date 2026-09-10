# LIBERO AutoDAgger

使用 SmolVLA 执行 LIBERO 任务，由 Robometer 判断是否切换到 openpi teacher。当前实现覆盖采集、筛选、LeRobot v3 导出和 HTML 回放，不包含 student 训练或前缀动作重标注。

模型加载、仿真、测试和浏览器验证均在 **hw** 上执行；本地浏览器通过 SSH 转发访问。本文按 2026-09-10 同步后的实验配置整理。

## 环境与模型

以下命令均从 hw 的工作目录执行：

```bash
cd /home/ma-user/work/users/luyuxiang/code/DAgger_Robometer
```

| 组件 | Python 环境 | 模型／资源 |
|---|---|---|
| Student、数据导出 | `lerobot-0.5.1/.venv/bin/python` | `/home/ma-user/work/model/smolvla_libero` |
| Robometer | `robometer/.venv/bin/python` | `/home/ma-user/work/model/robometer-4b-fft-libero` |
| 采集器、只读回看 | `/home/ma-user/work/users/luyuxiang/envs/libero/bin/python` | `/home/ma-user/work/users/luyuxiang/code/LIBERO` |
| Teacher | 由用户手动启动和管理 | 根目录 `better_openpi`，当前连接端口 8000 |

`run_hw.sh` 选择各自环境，不自动安装依赖或启动后台任务。环境路径可通过 `STUDENT_PYTHON`、`COLLECTOR_PYTHON`、`ROBOMETER_PYTHON`、`CODE_ROOT`、`LIBERO_ROOT` 调整；Robometer 路径由 `ROBOMETER_MODEL` 指定。脚本生成私有 `.libero/config.yaml`，不修改全局 LIBERO 配置。

Teacher 使用的 `pi05_libero_bl/29999` 原训练归一化文件为：

```text
/home/ma-user/work/dataset/lerobot/physical-intelligence/libero/norm_stats.json
```

Teacher 服务负责加载正确权重和统计。采集器记录服务提供的模型元数据，不独立验证 teacher 权重文件。

## 启动服务和采集

先选择可用 GPU，在不同终端启动服务。当前 JSON 中 `student_replan=5`，因此 student 服务每次至少需要返回 5 个动作：

```bash
bash autodagger_libero/run_hw.sh student --actions-per-chunk 5
bash autodagger_libero/run_hw.sh robometer
```

可在各命令前设置 `CUDA_VISIBLE_DEVICES` 选择 GPU。Student 出现 `[Student READY]` 表示权重加载完成且 WebSocket 已开始监听；没有请求时保持等待是正常状态，Ctrl+C 停止服务。

**配置边界：**采集参数全部来自 JSON；student 服务仍有独立启动参数，`--actions-per-chunk` 默认是 1，不会自动读取 `student_replan`。两者必须满足“服务返回动作数 ≥ 采集器执行动作数”。上述命令已与当前 JSON 对齐。

手动启动 teacher，确认其地址为 JSON 中的 `teacher_url`。Robometer 可以先检查：

```bash
curl --fail http://127.0.0.1:8102/health
```

健康检查不代替真实评分验证。服务就绪后启动采集：

```bash
bash autodagger_libero/run_hw.sh collect
```

默认读取 [config.json](config.json)。也可以先复制并编辑完整 JSON，再选择文件：

```bash
bash autodagger_libero/run_hw.sh collect --config autodagger_libero/config_custom.json
```

采集入口不支持 `--episodes`、`--output` 等命令行覆盖；缺少字段、未知字段都会报错。模型服务、导出和回看入口仍使用各自的参数。

## 当前采集配置

下面是仓库中 JSON 的当前值，不是 `core.py` 的初始默认值。

| 字段 | 当前值 | 含义 |
|---|---|---|
| `student_url` | `ws://127.0.0.1:8100` | Student WebSocket |
| `teacher_url` | `ws://127.0.0.1:8000` | 手动启动的 teacher WebSocket |
| `robometer_url` | `http://127.0.0.1:8102` | 评分 HTTP 服务 |
| `suite` | `libero_10` | 当前套件 |
| `task_ids` | `null` | 全部任务；如 `[0]` 则仅任务 0 |
| `episodes` | `50` | 每任务的初始状态数量 |
| `initial_state_start` | `0` | 初始状态起始编号 |
| `seed` | `7` | 随机种子 |
| `output` | `runs/autodagger_10` | 原始数据目录，相对于启动目录 |
| `student_replan` / `teacher_replan` | `5` / `5` | 每次推理后执行的动作数量 |
| `monitor_every` | `5` | 每隔多少环境步评分 |
| `max_frames` | `8` | 从轨迹前缀均匀抽样的最大帧数 |
| `stall_steps` | `50` | 无有效改善的步数阈值 |
| `progress_epsilon` | `0.05` | 有效改善必须超过的增量 |
| `regression_enabled` | `false` | 当前关闭持续退步接管 |
| `regression_threshold` / `regression_checks` | `0.1` / `3` | 开启退步规则时的下降幅度与连续检查次数 |
| `success_threshold` | `0.5` | 成功概率阈值 |
| `timeout` / `retries` | `60` / `1` | 请求超时秒数／额外重试次数 |
| `dashboard_host` / `dashboard_port` | `127.0.0.1` / `8088` | 只读仪表盘地址 |
| `force_teacher_step` | `null` | 无强制接管；设为步号则标记测试运行 |

少量烟测可在单独 JSON 中设置 `task_ids=[0]`、`episodes=1` 和新输出目录。正式配置为 10 个任务 × 每任务 50 个初始状态，共 500 个 episode。更换设置时使用新输出目录。

## 接管和 episode 结束

控制流程为 `STUDENT → TEACHER → END`，也允许 student 直接结束。开头执行 10 个 settling 动作，不计入采集步数。

- 在 step 0、之后每 5 步和正常结束时评分；等待评分期间不推进仿真。
- 每次从完整轨迹前缀均匀抽取最多 8 帧，包含首帧和当前末帧，不是只取最近 8 帧。
- 成功概率 `>0.5` 时重置停滞计时并清零退步计数。
- 成功概率 `≤0.5` 且连续 50 步无有效改善时接管；比上次改善基准增加 `>0.05` 才算有效改善，小幅增加可以累计。
- 当前退步规则关闭。启用后，相比历史最高进度下降至少 0.1 且持续 3 次检查会接管；高成功概率仍抑制该规则。
- 接管立即清空 student 队列，从当前观测请求 teacher，不再自动交还 student。
- 强制接管仍需要评分服务正常工作；自动规则可在强制步号之前触发。

每次动作后，`done` 或 `env.check_success()` 为真则停止；否则达到下面的**当前实验步数上限**时停止。这些上限在 [collect.py](collect.py) 中，已是原先预算的两倍，不由 JSON 配置：

| 套件 | 最大动作数 |
|---|---:|
| `libero_spatial` | 440 |
| `libero_object` | 560 |
| `libero_goal` | 600 |
| `libero_10` | 1040 |
| `libero_90` | 800 |

上限由 student 和 teacher 共用，接管不重置计数。Robometer 高成功概率、teacher 停滞都不会直接结束 episode。结束后用包含终止观测的前缀做最终评分。

请求重试耗尽、无效动作、环境异常或 Ctrl+C 会尝试保存已采数据，标记 `interrupted`，并停止整个采集进程。首次策略连接、重置连接和健康检查不使用推理请求的重试循环。正常 episode 保存后继续下一个。

## 原始数据与恢复

运行根目录的 `run.json` 保存完整配置和模型元数据。每个 episode 独立保存：

| 文件／字段 | 内容 |
|---|---|
| `trajectory.npz` | 动作前 `image`、`image2`、8D `state`、实际 `action`、未裁剪 `policy_action`、`step`、`collect` |
| `metadata.json` | 任务、seed、初始状态编号、接管位置和原因、带步号的评分、结束原因、筛选结果与环境成功 |
| `collect` | `rollout` 为 student 动作；`teacher` 为 teacher 动作 |
| `accepted_for_distillation` | 曾接管且最终 Robometer 成功概率严格超过阈值；异常中断时为 false |
| `test_only` | 强制接管运行标记；导出和默认筛选读取排除这些数据 |

所有原始 episode 都保留，包括未接管、失败和中断的数据。环境真实成功只用于审计，不改变 Robometer 筛选结果。图像旋转 180° 一次；state 顺序为位置 3D、轴角 3D、夹爪关节 2D。实际动作裁剪到 LIBERO 的 `[-1,1]`，原始预测单独保留。

每步写入临时日志；`metadata.json` 是 episode 的提交标记。恢复时只读取完整提交的帧，忽略未完成的临时文件，并将未完成 episode 标记中断。相同配置和模型元数据的重复运行跳过已保存 episode，**已保存的中断 episode 也会跳过**；重采需使用新目录。不会在恢复时修改旧运行的实验配置。

## 导出 LeRobot v3：10 fps

```bash
bash autodagger_libero/run_hw.sh export --run runs/autodagger_10 \
  --output datasets/autodagger_10 --repo-id local/autodagger_10
```

Python 函数和命令行入口都默认 **10 fps**，相邻帧时间戳相差 0.1 秒。已有导出数据集不会自动更新，需要重新导出到新目录。导出帧率设置不修改仿真环境本身的控制频率。

仅导出同时满足以下条件的非空 episode：

- `took_over=true`、`robometer_success=true`、`accepted_for_distillation=true`。
- 非测试运行，且正常由环境终止或达到步数上限结束。

入选 episode 保留完整 rollout 前缀和 teacher 后缀，字符串 `collect` 保留。未接管、失败、中断、测试和零帧 episode 不进入导出数据集或其 episode 映射。无符合条件的数据时报错，不发布空数据集。

导出先写临时目录，完成 `finalize()` 和审计文件后再发布；不覆盖已有目录，也不上传到 Hub。`meta/autodagger_episodes.json` 保存入选 episode 与数据集索引的对应关系，`meta/autodagger_run.json` 保存来源配置。

若只想读取通过筛选的 teacher 帧：

```python
from autodagger_libero.export_dataset import selected_frames

for episode_id, frame_index, frame in selected_frames(
    "runs/autodagger_10", accepted_only=True, include_rollout=False
):
    action = frame["action"]
```

`include_rollout=True` 可读取 student 段，但其动作仍是 student 预测，不是 teacher 监督。需要 teacher 标签时单独生成；训练时需明确处理动作 chunk 跨执行者边界的问题。

## HTML 仪表盘与轨迹播放器

采集期间默认监听 hw 的 `127.0.0.1:8088`。本地建立端口转发：

```bash
ssh -N -L 8088:127.0.0.1:8088 hw
```

- 仪表盘：<http://127.0.0.1:8088/>，显示实时双相机、执行者、请求耗时、评分、接管原因及统计。
- 播放器：<http://127.0.0.1:8088/player.html>，或点击历史列表的“播放”。

采集结束后可独立打开保存的运行，不加载模型：

```bash
bash autodagger_libero/run_hw.sh view --run runs/autodagger_10
```

如果 8088 被占用，可使用 `view --run runs/autodagger_10 --port 8089`，并在本地相应转发 8089。

播放器支持全部已保存 episode、双相机同步播放、暂停、拖动、单步、跳到接管位置，以及执行动作和 state 显示。曲线使用环境步号；当前帧分数只取该帧之前最近一次已记录评分，不生成插值分数。测试数据和模拟评分有明确标记。

画面是**动作前观测**；终止画面参与评分，但没有被保存为额外的动作帧。播放速度 2/5/10/20 帧每秒仅是观看选项，不代表真实执行耗时。

整个界面只读，不控制机器人或采集进程。服务限制并发请求，实时缓存有界；播放器使用独立锁和单 episode 解码缓存，超过 512 MiB 解压大小的轨迹会报错。浏览器断开不影响采集，播放器失败后可重新拖动或点击播放重试。

GET 接口：`/api/state`、`/api/episodes`（最近 100 条）、`/api/episodes/{id}`、`/api/replay-episodes`（全部）、`/api/replay/{id}/{index}`、`/frames/image.jpg`、`/frames/image2.jpg`。更新后端需要重启仪表盘／回看进程，模型服务无需随之重启。

## 文件导航

| 文件 | 职责 |
|---|---|
| `run_hw.sh` | 各环境启动入口 |
| `config.json` | 完整采集配置 |
| `collect.py` | LIBERO 环境、动作循环、接管与结束 |
| `core.py` | 参数校验、Gate、原始存储与恢复 |
| `clients.py` | 策略 WebSocket、Robometer HTTP 客户端 |
| `serve_student.py` | SmolVLA LIBERO adapter、就绪和退出提示 |
| `dashboard.py` / `dashboard.html` | 只读 API、实时仪表盘 |
| `replay.py` / `player.html` / `player.js` | 轨迹解码缓存与浏览器播放器 |
| `view_run.py` | 保存运行的独立回看入口 |
| `export_dataset.py` | 成功接管筛选、10 fps 导出、帧读取示例 |
| `prepare_libero.py` | 私有 LIBERO 路径配置 |
| `check_robometer_files.py` | 模型配置及分片存在性检查 |
| `tests/` | 控制、网络、导出、浏览器和模型烟测 |
| `pyproject.toml` | Black 行宽 88、Python 目标 3.8 |

## 验证与维护

最近已在 hw 验证：17 项单元测试；导出筛选、空结果、10 fps 元数据和时间戳、动作 chunk 回读；仪表盘与播放器 Chromium 检查。回放测试验证了原始文件没有变化。截图保存在 hw 的 `autodagger_libero/validation/`。

```bash
bash autodagger_libero/run_hw.sh test
bash autodagger_libero/run_hw.sh check-export
bash autodagger_libero/run_hw.sh check-browser
PYTHONPATH="$PWD" lerobot-0.5.1/.venv/bin/python \
  -m autodagger_libero.tests.check_player
```

GPU 模型检查和真实环境烟测单独运行：

```bash
bash autodagger_libero/run_hw.sh check-student
TEACHER_URL=ws://127.0.0.1:8000 bash autodagger_libero/run_hw.sh check-services
```

`check-student` 使用合成观测检查真实 student 加载和动作输出。`check-services` 临时启动 GPU 1 上的测试 student，连接已有 teacher，在真实 LIBERO 中运行 20 步，但 **Robometer 是模拟评分器**；不能据此声称三个真实模型的完整链路已验证。测试只停止自己启动的进程，拒绝占用中的测试端口。

Robometer 服务中的混合精度修复已用独立小型 CUDA LayerNorm 测试验证；它也不替代完整模型评分验证。同步版本的环境配置锁定 `torchao==0.13.0`。基础模型／处理器缓存是否完整、当前检查点是否成功加载，应以服务启动与真实请求结果判断。

Black 在 hw 的独立 uv 工具环境中执行：

```bash
/home/ma-user/.local/bin/uvx \
  --python lerobot-0.5.1/.venv/bin/python \
  --from black==24.8.0 black autodagger_libero
```

添加 `--check` 仅检查、不改写。运行这些命令不需要恢复已删除的 `.runtime` 目录。

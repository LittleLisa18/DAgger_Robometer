# 在线蒸馏验证记录

验证日期：2026-09-10 至 2026-09-11（Asia/Hong_Kong）。

## 环境与输入

- 在 `hw` 独立目录 `/tmp/codex-smolvla-distill-01a08b28` 验证，未覆盖服务器现有仓库。
- 使用服务器已有 `lerobot-0.5.1/.venv`：PyTorch `2.10.0+cu128`。
- Student：公开通用 `lerobot/smolvla_base`，revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`，模型文件 906,712,520 字节；不是采集用的 LIBERO 微调 checkpoint。
- Teacher：已有 `ws://127.0.0.1:8000`，实测响应形状为 `(10, 7)`。服务握手 metadata 为空，因此本次没有从服务端独立验证其 checkpoint 身份。
- 数据：从已接受的真实采集数据中选取两个 episode，保留完整 rollout/teacher 段，通过现有 exporter 导出为 462 帧、10 FPS 的 LeRobot v3 数据。
- 来源 episode：`libero_10_t002_i014_s7`（239 帧）、`libero_10_t002_i022_s7`（223 帧）。仅建立临时选择目录和新导出，不改写原始轨迹。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 13 项单元/回归测试 | 通过：collect 掩码、episode 子集与边界、数据指纹、teacher 长度、网络重试/超时、像素转换、归一化、loss reduction |
| 两 GPU 的 DDP 数值测试 | 通过：与单进程等价 batch 的 loss/梯度一致，覆盖 rank 无 BC、全局无 BC、单 rank 请求失败 |
| 真实 teacher + 数据 + tokenizer + processors | 通过；KD/BC 合并动作形状 `(4,10,7)`，数值有限 |
| 单 GPU 训练 | 从通用 base 训练 10 步并保存；batch_size=1，float32 |
| 双 GPU 训练 | 从通用 base 训练 10 步并保存；每 rank batch_size=1，bfloat16 autocast |
| 单 GPU 恢复 | 从第 10 步恢复到第 11 步并保存 |
| 双 GPU 恢复 | 从第 10 步恢复到第 11 步并保存 |
| 现有 student adapter | `SmolVLALiberoAdapter` 加载单卡第 11 步 checkpoint，真实观测推理返回有限值 `(10,7)` |
| 本地与 hw 验证源码 | 8 个新增/修改源码及测试文件归一化换行后的 SHA-256 一致 |
| 静态检查 | 新增代码 Ruff 检查/格式检查、修改训练入口 Ruff 检查、`git diff --check` 通过 |

单卡 11 次更新中有 1 次 BC 非零，双卡 11 次更新中有 3 次 BC 非零；两者 teacher 重试总数均为 0。
例如单卡第 4 步：KD=1.820964、BC=3.127750，有效 BC 动作数为 10；其他 rollout 样本 BC=0。
这些数值用于确认训练分支实际执行，不用于评价策略质量。

## Checkpoint 内容核验

以 `model.action_out_proj.weight` 为例，对通用 base 的最大绝对变化：

| Checkpoint | 最大绝对变化 |
| --- | ---: |
| 单卡 step 10 | 0.0005196035 |
| 单卡 step 11 | 0.0005209558 |
| 双卡 step 10 | 0.0005381554 |
| 双卡 step 11 | 0.0005391538 |

step 10 到 step 11 仍有非零更新；保存的 training_step 分别为 10/11。
四个 checkpoint 中抽查的冻结视觉 patch embedding 权重与 base 完全相同。
单卡和双卡恢复前后的 180 项 processor 统计数值均完全相同。
现有 processor 序列化会将其中 54 项长度为 1 的辅助统计转成标量；7D action、8D state 的归一化向量未变。

验证产物位于上述 hw 临时目录：

- `single.log`、`dual.log`、`resume_single.log`、`resume_dual.log`
- `output_single/checkpoints/{000010,000011}`、`output_dual/checkpoints/{000010,000011}`
- `output_single/distillation_metrics.jsonl`、`output_dual/distillation_metrics.jsonl`
- `verification.json`

## 多数据集验证（2026-09-11）

新增 `dataset.sources` 后再次在 hw 验证。为验证同 repo_id、独立 root 和独立 episode 选择，
将上述小型导出复制到临时 `multi_source_a`、`multi_source_b` 两个目录，分别选择 episode 0/1。
两源共 239+223=462 帧，不重复采样相同 episode；原始采集文件和服务器正式源码未改动。
两份数据来自同一次真实采集；不同均值、重复局部索引和不同 task 文本另由合成单元测试覆盖。

| 检查 | 结果 |
| --- | --- |
| 18 项蒸馏测试 | 通过，含跨源边界、相同 repo_id/局部索引、子集统计合并、来源顺序/内容/选择指纹、配置解析 |
| 原有 DatasetConfig 的 5 项测试 | 全部通过，单数据集配置保持兼容 |
| 真实数据 + 2 个 DataLoader workers | 完整读取两源各 239/223 帧，边界 BC mask 正确 |
| 合并归一化与所选原始帧直接计算 | action mean/std 最大误差分别 4.2e-8/2.3e-8；state mean/std 分别 1.1e-7/1.3e-6，符合存储统计的浮点精度 |
| 单卡训练及恢复 | 10 步训练、保存，再恢复到第 11 步并保存 |
| 双卡训练及恢复 | 每 rank batch_size=1、bf16；10 步训练、保存，再恢复到第 11 步并保存 |
| 恢复后的参数更新 | action_out_proj.weight 最大变化：单卡 2.13e-6，双卡 2.32e-6；抽查冻结视觉权重不变 |
| 恢复归一化 | 单卡/双卡各 40 项保存的 processor 统计数值不变，action/state 统计匹配合并统计 |
| 老单数据集 checkpoint | 新加载入口生成的指纹与旧 checkpoint 保存的指纹一致 |
| 调换来源顺序后 resume | 在模型创建前按预期拒绝，提示数据内容/选择变化 |
| 现有 student adapter | 多数据集单卡第 11 步 checkpoint 可加载，真实观测推理得到有限 `(10,7)` |
| 本地与 hw 源码一致性 | 10 个源码/测试文件的 SHA-256 一致（归一化换行），清单保存为 `multi_source_hashes.json` |

单卡 11 次更新中 BC 非零 1 次，双卡 3 次；teacher 重试均为 0。
静态 Ruff 检查、格式检查及源码 `git diff --check` 通过。

首次单卡训练已完成 10 次更新，但 `/tmp` 所在 overlay 文件系统耗尽配额，保存失败。
后续完整重跑使用独立共享内存临时目录 `/dev/shm/codex-smolvla-multi-01a08b28`，
由原验证目录的 `multi_outputs` 符号链接访问；这些 checkpoint 为临时产物，重启后不保证保留。
没有删除旧验证 checkpoint 或修改正式仓库来腾出空间。

验证目录 `/tmp/codex-smolvla-distill-01a08b28` 中的新增产物：

- `multi_single.json`、`multi_dual.json`
- `multi_unit.log`、`multi_config.log`、`multi_prepare.log`
- `multi_single_retry.log`、`multi_single_resume.log`、`multi_dual.log`、`multi_dual_resume.log`
- `multi_outputs/{single,dual}/checkpoints/{000010,000011}` 及各自 `distillation_metrics.jsonl`
- `multi_data_verification.json`、`multi_training_verification.json`、`multi_resume_reordered.log`

## 无 collect 示教数据混训验证（2026-09-18）

新增来源参数 `supervision="demonstration"`，默认仍为 `autodagger`。
使用同一个 hw 隔离源码目录，在 `/dev/shm/codex-smolvla-demo-20260918` 创建测试数据和输出。
普通示教测试源由之前的小型 LeRobot 导出复制后移除 collect 列、对应 feature 和 AutoDAgger 审计文件构造，
仅用于验证加载与监督机制，并非新的人类示教数据或策略效果实验。原数据不变。

- 22 项回归测试全部通过，新增覆盖无 collect/无审计加载、两种来源排列下的 batch 拼接、
  普通示教 BC、episode/数据集边界、padding、错误监督模式及禁止覆盖已有 collect。
- 实际混合读取 239 帧 AutoDAgger episode 和 223 帧无标签示教测试 episode，共 462 帧。
- 显式构造一个 rollout 样本和一个示教样本的 batch：两者均参与 KD；
  rollout 的 BC mask 全零，示教的 10 步 BC mask 全真。BC 标签与原始数据动作的归一化结果一致，
  KD 标签与模拟 teacher 动作的归一化结果一致；检查梯度仅流向允许的 BC 样本。
- 本次 hw 原 teacher 8000 端口拒绝连接，因此短训练使用独立临时 WebSocket 模拟服务，
  返回固定的有限 `[10,7]` 动作，checkpoint 的 teacher_id 为 `MOCK_VALIDATION_ONLY`。
  测试完成会关闭该服务。本轮不能作为真实 OpenPI teacher 或策略质量验证。
- 3 个本次修改的源码/测试文件与 hw 验证副本的 SHA-256 一致（归一化换行）；Ruff 及 diff 检查通过。
- 使用真实 SmolVLA base 权重、tokenizer 和 processors，单卡与双卡均完成 3 步训练、保存，
  再恢复到第 4 步并保存。每 rank batch_size=2，DataLoader workers=2，双卡使用 bf16。
  恢复后的 action_out_proj.weight 最大更新分别为 2.4885e-6/2.4997e-6，processor 统计数值均不变。
  单卡 4 次更新中 BC 非零 3 次，双卡 4 次均非零，训练 loss 均有限。

本次日志位于 `/tmp/codex-smolvla-distill-01a08b28` 的 `demo_unit.log`、`demo_prepare.log`、
`demo_mock_verify.log`、`demo_{single,dual}_{train,resume}.log`，汇总为 `demo_verification.json`。
checkpoint 位于上述共享内存目录的 `{single,dual}/checkpoints`，为临时测试产物。

## hw 原训练配置的 HWC 元信息兼容修复（2026-09-18）

用户实际配置为服务器正式仓库的 `configs/distill_smolvla.json`，数据目录为
`/home/ma-user/work/dataset/lerobot/libero_lerobot30`，273,465 帧、1,693 个 episode。
视频元信息使用 `[256,256,3]` 及 `height/width/channel`，torchcodec 实际解码输出 `[3,256,256]`。
旧校验错误地要求元信息本身为 CHW；数据同时没有 collect，而正式仓库尚未更新 demonstration 模式。

修复按 LeRobot 的轴名称规则校验解码后的形状，支持 CHW/HWC 两种元信息，兼容多来源混合。
在确认服务器两个源码文件与本地旧版完全一致后，备份并更新 `configs/default.py` 和
`datasets/autodagger_distillation.py`；将服务器训练配置改为单项 sources，显式声明 demonstration。
其余训练参数保留，数据集文件未改写。

- 23 项蒸馏回归测试通过，包含 HWC/CHW 来源混合、无标签示教、禁止错误分辨率和不修改元信息。
- 在隔离代码和更新后的正式源码中，分别通过完整数据集索引、episode 边界和文件内容指纹校验。
- 抽查图像张量 `[3,256,256]`，teacher 输入图像 `[256,256,3]`；首帧有效 BC 10 步，最终帧仅 1 步。
- 数据文件指纹两次一致，`info.json` 的 SHA-256 不变。
- 本轮没有启动长训练或真实 teacher；复核时 8000 端口未监听。

原源码及配置备份：`/tmp/codex-smolvla-distill-01a08b28/backups/hwc-production-20260918-203126`。
同一隔离目录下保存 `hwc_unit.log`、`hwc_preflight.json`、`hwc_install.json`、`hwc_production_verified.json`。

## 两张卡各自运行真实 OpenPI teacher（2026-09-18）

在隔离目录验证新的 `train_distill_smolvla.sh`，随后备份并安装到服务器正式仓库。
使用真实 `pi05_libero_bl/29999` 权重，以及用户配置的数据集（273,465 帧普通示教）。
测试设置为每 rank batch_size=1、2 步、不保存 checkpoint；没有启动用户的 100,000 步训练。

- 24 项蒸馏测试通过，新增双 WebSocket 服务的 rank 路由测试；脚本语法及修改源码 Ruff 检查通过。
- 特意使用 `CUDA_VISIBLE_DEVICES=1,0`，验证 rank 0 对应物理 GPU 1 / 18700，rank 1 对应 GPU 0 / 18701。
  读取 teacher 进程环境确认卡号，两个真实服务均收到训练连接。
- 两步 KD/BC 均有限，分别为 2.539512/2.670636 和 5.805365/2.751995；teacher 重试均为 0。
- 首步 teacher 请求约 30.37 秒（包含编译），第二步约 0.21 秒；这不是正式 batch_size=32 的性能测量。
- 训练退出后本次 teacher 进程全部消失、18700/18701 端口释放；原有 8000 服务仍保持监听。
- 占用端口测试确认脚本在加载模型前失败。OpenPI 进程清除 student 的 PYTHONPATH 后，旧版 lerobot.common 依赖正常加载。

证据保存在 `/tmp/codex-smolvla-distill-01a08b28`：`dual_teacher_unit.log`、
`dual_teacher_launch_retry.log`、`dual_teacher_processes.json`、`dual_teacher_verified.json`。
服务日志为 `dual_teacher_logs/run-YD7oDLSa/teacher_rank{0,1}.log`。
正式目录安装的启动脚本及三个 Python 文件均校验哈希一致，旧文件备份于
`backups/dual-teacher-production-20260918-210620`，安装清单为 `dual_teacher_install_report.json`。

## 启动脚本精简（2026-09-19）

按用户要求将 Bash 精简为 78 行，直接启动两个 teacher，再用 curl 查询 OpenPI 原生 `/healthz`，
最后启动 Accelerate。保留端口检查、超时、teacher 退出检测和本次进程组清理。

精简过程中完成真实双 teacher / 双卡两步训练；最终 healthz 版本另完成一次单步双卡训练，
使用 GPU 顺序 `1,0`，每 rank batch_size=1。最终 KD=2.541899、BC=2.670636，teacher 重试为 0。
脚本退出状态为 0，测试端口 18700–18703 全部释放，已有 8000 服务仍在运行。
本轮没有启动正式长训练，未验证正式 batch_size=32 的显存余量。

正式脚本已经替换并校验哈希。旧版备份位于
`/tmp/codex-smolvla-distill-01a08b28/backups/simple-launcher-20260919-162802`。
同一隔离根目录保存 `simple_teacher_train.log`、`simple_healthz_train.log` 和 `simple_launcher_verified.json`。

## 可直接编辑、可变卡数的新脚本（2026-09-19）

新增 `train_distill_smolvla_simple.sh`，在 Bash 顶部直接修改 GPU 列表、起始端口、训练 JSON 和 teacher 路径。
无命令行参数或环境变量覆盖入口。GPU 数量由列表长度计算，每张 GPU 启动一个 teacher，端口依次递增。
旧启动脚本保留。新脚本已安装到 hw 正式目录，本地与服务器 SHA-256 一致。

真实单卡（GPU 1）与双卡（GPU 顺序 1,0）各完成一步训练，每 rank batch_size=1，退出状态均为 0。
单卡 KD/BC 为 4.717989/5.347788，双卡为 2.539018/2.670636，teacher 重试均为 0。
测试端口 18710、18712、18713 均释放。四卡只验证了地址生成，未进行四卡硬件训练。
日志位于 `/tmp/codex-smolvla-distill-01a08b28/configurable_{one,two}/train.log`。

## 结论边界

已验证真实 teacher 下的训练、分布式归约、保存恢复和 checkpoint 推理链路。
未运行完整 LIBERO policy rollout，也未测量蒸馏前后成功率；这些短训练 checkpoint 不是经过效果验证的生产策略。

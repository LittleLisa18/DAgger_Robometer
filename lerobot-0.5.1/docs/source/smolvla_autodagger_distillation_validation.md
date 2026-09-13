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

## 结论边界

已验证真实 teacher 下的训练、分布式归约、保存恢复和 checkpoint 推理链路。
未运行完整 LIBERO policy rollout，也未测量蒸馏前后成功率；这些短训练 checkpoint 不是经过效果验证的生产策略。

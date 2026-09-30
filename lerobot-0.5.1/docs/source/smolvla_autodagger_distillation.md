# SmolVLA：AutoDAgger 在线动作蒸馏

实际运行结果见 [hw 验证记录](smolvla_autodagger_distillation_validation.md)。

Student 使用通用 `lerobot/smolvla_base` 权重，teacher 使用独立 OpenPI LIBERO policy server。
训练不加载 OpenPI 模型，不启动 Robometer 或模拟器。输入支持 AutoDAgger 导出及显式声明的普通示教 LeRobot v3 数据；
原始 `trajectory.npz` 需先通过现有 `autodagger_libero.export_dataset` 导出。

## 标签与损失

| 当前帧 collect | 在线 KD | 实际动作 BC |
| --- | --- | --- |
| rollout | 使用 | 不使用，即使未来窗口跨入 teacher 段 |
| teacher | 使用 | 仅同 episode 的有效 teacher 动作位置 |

每次 teacher 必须返回有限值 `[10,7]`。Student `chunk_size` 支持 1–10，默认 10；
取 teacher 的前 H 步，不插值、不补齐。`n_action_steps` 必须在 1–H 之间。
LIBERO 是 10 Hz，所以 H=10 代表未来 1 秒；采集重新规划间隔独立设置。

Teacher 请求使用原始双相机、8D state 和 task，图像不二次旋转；server 返回 LIBERO 环境坐标的 7D 动作。
沿用采集器的 `[-1,1]` 裁剪，并使用当前训练数据集的统计转换到 student 归一化空间。
Teacher 自身输入输出归一化由 server 负责。

KD 将 teacher 最终动作 a 当作 flow-matching 标签：采样噪声 e 和时间 t，
构造 `x_t=(1-t)*a+t*e`，监督 `v_student(o,x_t,t)` 拟合 `e-a`。
BC 用同一训练形式，但 a 是已执行的 teacher 动作。
两项各自按有效动作元素取均值，默认 `loss = kd_loss + bc_loss`。
这与根目录 OpenPI `train_distill.py` 的 ViT/LLM/速度场蒸馏不同。

## 配置与运行

若希望直接在 Bash 中修改参数，使用 `train_distill_smolvla_simple.sh`。
该脚本没有命令行参数或环境变量覆盖入口；只需编辑顶部的 GPU、端口、训练 JSON 和 teacher 配置：

```bash
GPUS=(0 1)        # 单卡改为 (0)，四卡改为 (0 1 2 3)
TEACHER_PORT=18700
MASTER_PORT=29500
CONFIG="configs/distill_smolvla.json"
```

然后执行 `bash train_distill_smolvla_simple.sh`。卡数自动取 GPU 列表长度，teacher 端口从起始端口递增；
每个 local rank 连接列表中同一位置的 GPU 上的 teacher。单卡不传 `--multi_gpu`，多卡自动启用。
下述 `train_distill_smolvla.sh` 保留原有环境变量与命令行入口，供需要这些入口时使用。

编辑 `configs/distill_smolvla_autodagger.json` 的数据目录、teacher 标识和输出目录。
`teacher_id` 填 checkpoint 路径/版本与服务配置的稳定标识，用于审计，不是自动验证 server 权重的证明。
`policy.pretrained_path` 可用 Hub ID 或本地通用 SmolVLA 目录；示例采用标准 base 架构默认值，
非标准模型需使用匹配的 policy 配置。首次运行需要能获取 checkpoint、VLM 配置及 tokenizer。

```bash
# 自动在两张卡分别启动 teacher，再启动双卡 student。
bash train_distill_smolvla.sh configs/distill_smolvla_autodagger.json

# 需要保留已有 8000 服务时，选择其他端口。
CUDA_VISIBLE_DEVICES=0,1 TEACHER_PORTS=18700,18701 bash train_distill_smolvla.sh

# 手动管理 teacher 时，也可直接启动 student。batch_size 是每个 rank 的大小。
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes=2 --num_machines=1 --mixed_precision=bf16 \
  --dynamo_backend=no --module lerobot.scripts.lerobot_train \
  --config_path=configs/distill_smolvla_autodagger.json
```

启动脚本不传配置参数时使用 `configs/distill_smolvla.json`。默认 GPU 为 `0,1`，
teacher 端口为 `8000,8001`：local rank 0 连接第一张卡的 teacher，rank 1 连接第二张卡的 teacher。
脚本通过 `distillation.teacher_urls` 命令行列表传入地址，并按 `accelerator.local_process_index` 选择；
没有配置列表时，手动启动仍兼容原有的单一 `teacher_url`。

Teacher 默认沿用服务器 `better_openpi/start_server.sh` 的 `pi05_libero`、
`checkpoints/pi05_libero_bl/29999` 和 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`。
可通过 `OPENPI_ROOT`、`TEACHER_CONFIG`、`TEACHER_CHECKPOINT` 覆盖；切换权重时也需更新训练配置的 teacher_id。
每个 teacher 使用 OpenPI 自身的 uv 环境，清除继承的 VIRTUAL_ENV/PYTHONPATH，避免 LeRobot 版本冲突。
脚本先检查端口，再用 curl 等待两个 teacher 的原生 `/healthz` 接口返回成功，最后启动 Accelerate。
正常退出、报错或 Ctrl-C 会清理本次启动的进程组。已有服务不会被终止，端口被占用时直接报错。
Teacher 日志位于 `outputs/teacher_servers/run-*/teacher_rank{0,1}.log`。

短训练可复制 JSON，设置新 output_dir、`steps=10`、`save_freq=10`、`log_freq=1`、
`batch_size=1`、`num_workers=0`、`policy.scheduler_warmup_steps=1`。
每 rank 串行请求各样本，现有 server 不需要 batch API；大 batch 会增加等待时间。
日志包含 KD、BC、全局有效动作步数及本 rank 的 teacher 延迟/重试次数。
JSONL 指标保存在输出目录的 `distillation_metrics.jsonl`，不会上传模型或数据。

## 多数据集

使用 `configs/distill_smolvla_autodagger_multi.json`，把单数据集的 `repo_id/root/episodes`
替换为 `dataset.sources`；其余训练参数和启动命令相同：

```json
"dataset": {
  "sources": [
    {"repo_id": "local/round1", "root": "/data/autodagger_round1"},
    {"repo_id": "local/round2", "root": "/data/autodagger_round2", "episodes": [0, 2, 5]}
  ],
  "image_transforms": {"enable": false},
  "use_imagenet_stats": false,
  "video_backend": "pyav",
  "streaming": false
}
```

`episodes` 是各自数据集内的编号，省略或为 null 表示该源所有 episode。
默认每个源都要求 AutoDAgger 审计和 collect 标注；普通无标签示教数据需显式设置下文的 demonstration 模式。
各源的模型输入输出 feature schema、动作含义和 FPS 必须一致。
允许不同目录使用同一 repo_id；拒绝重复目录及顶层单数据集参数与 sources 混用。
此入口只用于 AutoDAgger 蒸馏；普通 LeRobot 训练的多数据集入口没有启用。

拼接后均匀随机采样帧，因此默认各数据源的采样比例等于所选帧数占比，不提供源权重配置。
每个源先独立构造 action chunk 和 BC mask，不跨 dataset 或 episode。
当前帧为 rollout 时依旧只有 KD，当前帧为 teacher 时才对有效 teacher 动作位置计算 BC。
多数据集样本附带 `dataset_index`、`source_index`、`source_episode_index`；
`index/episode_index` 转换成全局编号，`task` 保留来源任务文本，`task_index` 仍为源内编号。

状态、动作及相机统计按 count 合并，方差包含数据源均值差异。
指定 episode 子集时，只读取所选 episode 的统计；不把未选 episode 算进去。
初次训练保存合并统计，恢复时直接加载 checkpoint processors。
支持默认 MEAN_STD，以及 MIN_MAX/IDENTITY；不支持多数据集 QUANTILES/QUANTILE10，
因为简单平均分位数不能得到合并分布的真实分位数。
单数据集配置和旧单数据集 checkpoint 的指纹算法保持兼容。

## 混入没有 collect 的普通示教数据

使用 `configs/distill_smolvla_mixed.json`，为普通示教来源显式设置
`"supervision": "demonstration"`，例如：

```json
"sources": [
  {"repo_id": "local/autodagger", "root": "/data/autodagger", "supervision": "autodagger"},
  {"repo_id": "local/demos", "root": "/data/demos", "supervision": "demonstration"}
]
```

`autodagger` 是默认值，继续要求 collect 和 AutoDAgger 审计文件。
`demonstration` 表示用户明确将该源声明为示教数据：不要求 collect 或 AutoDAgger 审计文件，
每帧都查询 teacher 计算 KD，并以该数据集中原有的示教动作计算 BC。
BC 仍排除 padding 和跨 episode 的位置；默认两项权重都是 1，由全局 kd_weight/bc_weight 控制。
示教 BC 的目标是数据动作，不是在线 teacher 返回的动作。
若该源实际含 collect，demonstration 模式会拒绝加载，以避免覆盖已有标签；请使用 autodagger 模式。

示教数据仍须满足当前 LIBERO 的 10 FPS、双相机、8D state、7D action 及相同动作语义，
并具有完整 LeRobot 元数据。模型输入输出 feature schema 必须匹配；来源专属辅助字段只保留公共部分，
因此混合 batch 不要求示教数据拥有 AutoDAgger 的 collect 等辅助列。
图像元信息支持 CHW `[3,256,256]`，也支持旧数据的 HWC `[256,256,3]`
（names 为 height/width/channel 或 channels）。校验按解码后的 CHW 形状进行；
不修改数据集元信息，也不对 reader 已输出的 CHW 图像再次转置。
也可以只配置一个 demonstration source。省略 supervision 不会自动推断为示教数据。
示教模式参与数据指纹，不能把原来的 AutoDAgger checkpoint 改成另一种监督语义后直接恢复。

## 保存和恢复

输出标准 LeRobot checkpoint，含 policy、processors、优化器、scheduler、训练步数和训练配置。
通过已保存的训练配置恢复，保留原来的多卡启动前缀（如有）：

```bash
PYTHONPATH=src .venv/bin/python -m lerobot.scripts.lerobot_train \
  --config_path=outputs/train/smolvla_autodagger_distill/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

恢复校验 teacher_id、student chunk 和数据内容指纹。可以变更 teacher URL 以连接同一模型的新地址。
多数据集指纹还包含有序数据源及各自的内容/episode 选择；添加、删除、调换数据源或修改内容后需新开训练。
恢复使用保存的归一化统计；teacher 随机生成的标签不保证逐位复现，数据采样位置也不保证原样续接。
输出模型可由现有 AutoDAgger student 服务加载，图像/状态/动作接口保持一致。

## 约束与验证

AutoDAgger 来源必须有 `meta/autodagger_episodes.json`、`meta/autodagger_run.json` 和合法 collect 字段；
拒绝测试 episode、未通过最终评分的 episode、不完整 episode 或无 teacher 接管数据。
显式声明的 demonstration 来源不要求这两份审计文件和 collect，但同样检查 episode 完整性和边界。
初次启动对数据文件计算内容指纹，较大数据集需要额外读盘时间。
暂不支持流式数据、图像随机增强、RA-BC、PEFT、RTC、Aloha 变换或相机重命名。

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/distillation -v
PYTHONPATH=src .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/distillation/check_ddp.py
```

Teacher 断线最多重试配置次数；非法输出立即报错。任一 rank 请求失败，全部 rank 在反向传播前停止，
不能用 rollout 动作替代 teacher 标签。多卡分别对两项 loss 做全局有效元素归约，
覆盖某个 rank 没有 BC、全部 rank 没有 BC 的情况。
短训练验证只能证明训练链路可用；策略质量需要独立 LIBERO rollout 评估。

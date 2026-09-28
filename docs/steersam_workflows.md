# SteerSAM 工作流与维护边界

更新：2026-09-28。本文列出当前入口，不把历史实验的分辨率、batch、样本上限或显存结果当作当前默认参数。实际参数以 Hydra 合成配置和实验目录中的配置快照为准。

## 1. 当前实现的六个部分

| 部分 | 入口/模块 | 职责 |
|---|---|---|
| 模型 | `sam3/model/steering.py`、`steersam_*.py`、`model_builder.py` | 文本的 contextual 1024 维特征在 ViT 中进行 early steering；256 维特征继续走原 late fusion |
| 监督与训练 | `steersam_patch_loss.py`、trainer/optimizer 接入 | 策略二训练 adapter、FPN、patch head；冻结原 ViT/语言/任务栈，但保留输入梯度；任务 loss 加 patch loss |
| 数据 | `build_steersam_pairs.py`、`unified_pair_loader.py` | 六来源质量筛选、去重、压缩 RLE、SQLite 随机访问、稳定抽样 |
| 配置 | `steersam_six_dataset*.yaml`、六个 `steersam_internal_val_*.yaml` | 联合训练、mini、来源级内部 patch loss/PMASS 诊断 |
| 正式评测 | COCO/LVIS 原生配置、Gold/COCO-O runner、官方 evaluator | 原始协议和原始划分；不把过滤后的训练索引作为官方 GT |
| 验证与说明 | `tests/`、`docs/` | CPU 回归、接口初始化、历史 GPU 验收和使用说明 |

原始数据 → 离线筛选/去重 → SQLite → dataset/transforms/collator → SteerSAM → 任务 loss + patch loss。训练 checkpoint 分别供内部验证与正式评测使用，两条路径不能混为一谈。

## 2. 默认入口与保留的历史入口

- 当前联合训练：`configs/steersam/steersam_six_dataset.yaml`。它是独立配置，不继承旧 COCO/PhraseCut 训练配置。`max_train_pairs` 是每来源训练 pair 上限，是否全量取决于实际值，不取决于文件名。
- 小规模实验：`configs/steersam/steersam_six_dataset_mini.yaml`。当前每来源 200 条、val 合计 60 条、10 epoch；其余参数继承主配置，包括 checkpoint 保存。不是原来的每来源 2 条接口 smoke，也不是正式评测。
- 六个内部 val：只报告 patch loss、PMASS、有效监督比例，读取训练 checkpoint；不能当作来源的官方指标。
- 旧 COCO 训练/smoke/frozen-backbone 配置保留用于历史回归。旧 frozen-backbone baseline 的可训练模块不同于策略二，不能直接当作只有 steering 开关不同的严格对照。
- 原始 JSON loader 保留：baseline、正式评测和模型测试需要它。SQLite loader 不替代这些用途。
- 模型子类中复用/镜像原 forward 是为保护原 SAM3 行为；不要为了减少代码行数删除或随意合并，修改原类时同步运行 parity 测试。

## 3. 数据索引及抽样报告

完整索引、partial 调试索引与运行时抽样是不同概念：

- 新构建完成后内嵌 `build_state=complete/partial` 与候选上限；未完成索引不可读取。
- partial 索引默认拒绝，只有 smoke 显式设置 loader `allow_partial_index: true` 才允许。
- 旧索引从相邻 `.audit.json` 确认完整性，兼容现有索引且不改写数据。复制旧 SQLite 时同时复制审计文件。
- `max_train_pairs` 在合格索引上做稳定抽样，不改变索引，也不限制 val/test。
- 报告脚本解析 Hydra defaults，因此可直接使用 mini 配置。使用独立 `--output`，不要覆盖历史报告。

```bash
CUDA_VISIBLE_DEVICES="" python scripts/report_steersam_selection.py \
  --config sam3/train/configs/steersam/steersam_six_dataset_mini.yaml \
  --output /tmp/steersam_mini_selection.json
```

不要在训练时覆盖正在使用的 SQLite 文件。新版本使用新的文件名，再显式修改新实验的 `pair_index`。

## 4. 正式评测边界与一条命令入口

### COCO / LVIS

原生 SAM3：`configs/steersam/eval_official_sam3_{coco,lvis}_{1008,672}.yaml`。672 配置轻量覆盖 1008 的分辨率、batch 和输出目录，其余协议相同。

训练后 SteerSAM 的旧 COCO 独立配置 `configs/coco/coco2017_steersam_mask_eval.yaml` 仍保留，但运行前必须确认 checkpoint 路径与训练分辨率。它使用一图一 query 的 exhaustive loader，以原始 image/category ID 计算 bbox/mask AP；`max_images` 在类别展开前生效。默认不计算 COCO prompt-pair cgF1。

COCO prompt-pair helper 仅为可选诊断与测试，不是官方表格指标。旧 LVIS positive JSON adapter、虚拟 pair loader/GT/evaluator 已移除；不得继续引用这些历史接口来声称复现论文 cgF1。

当前六来源内部 val 不是 RefCOCO/+/g 或 PhraseCut 的正式 referring-expression 评测。正式评测接入仍需独立实现，不能以 PMASS 替代。

### SA-Co/Gold

项目 wrapper 依次调用七个官方 Gold 配置，再调用未修改的 `scripts/eval/gold/eval_sam3.py`。预检查全部 a/b/c 标注。`--gt-folder` 同时传到推理配置和汇总，`--pred-folder` 同时控制推理输出与汇总读取；图像根目录仍来自 `eval_base.yaml`。此入口默认评测原生 SAM3，不自动选择 SteerSAM checkpoint。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python scripts/eval/steersam/run_official_saco_gold.py --num-gpus 2 --use-cluster 0
```

可以先加 `--dry-run` 只打印命令和检查标注，不启动 GPU 推理。

### COCO-O

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore \
python scripts/eval/steersam/run_official_coco_o.py --gpus 2,3 --num-gpus 2 --use-cluster 0
```

- `--output-root` 同时控制六个域的实际输出和汇总目录。
- `--dry-run` 不推理，也不读取旧结果。
- `--aggregate-only` 只读取结果；trainer 的 `val_stats.json` 是 JSONL，使用最后一个非空记录；缺少域或 bbox 指标会失败。
- `--smoke-images N` 是小样本流程测试，结果不能声称复现全量 APo；不要与全量实验共用输出目录。
- 两个 runner 仅支持同步本地 `--use-cluster 0`；集群任务必须单独提交、等待完成后汇总，不在未完成时误报结果。

## 5. 实验隔离与验证

改变分辨率、数据上限或训练目标时使用新输出目录。Trainer 会自动恢复目录中的 `checkpoint.pt`；从不同分辨率目录直接 resume 还可能遇到 RoPE buffer shape 不匹配。加载预训练/评测权重与恢复完整训练状态是不同入口，不能混用。每个实验保留配置快照、selection 报告和评测范围。

CPU 回归（`sam3` 环境）：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONWARNINGS=ignore python -m pytest -q tests
```

新增测试覆盖 loader 构造参数、继承配置、JSONL 读取、runner 路径贯通/dry-run/cluster 拒绝、partial/旧索引兼容。CPU 验证不代替 GPU 峰值显存、DDP 或长程训练验证。

待用户在 GPU 空闲时验证：先按实际分辨率降低 mini 的物理 batch，再执行下列命令。检查 train/val 前向、有限 loss、反向、保存/恢复和显存；不要用小样本 PMASS 声称正式指标。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py \
  -c configs/steersam/steersam_six_dataset_mini.yaml \
  --use-cluster 0 --num-gpus 2
```

正式推理先 dry-run，再运行上面的 Gold/COCO-O 命令，检查六/七个子集的输出目录、完整结果汇总与重复运行 JSONL 的读取。此前 GPU 记录见 `steersam_gpu_checks.md`，仅作为带日期的历史记录。

## 6. 本次维护范围

前期整理修复入口接口与数据完整性保护、移除无调用的旧 LVIS pair 实现、让正式 COCO 仅输出 AP、精简六数据集 YAML 的未使用后处理/输入框扰动/重复过滤、折叠重复的 672 评测配置、同步文档。保留原 SAM3 与 SteerSAM 核心结构、正式 evaluator、旧 baseline/smoke 回归入口和数据软链接；整理不改训练预算、学习率或采样上限，不覆盖训练产物。后续按用户授权分组提交上述工作，Git 提交阶段仍仅使用 CPU 验证。

后续可按模型与训练支持、六数据集链路、评测工作流、历史清理和文档分组提交。每组需可导入并通过对应测试；共享文件按实际差异分块，不能只按文件名划分依赖。

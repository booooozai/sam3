# SteerSAM 六数据集 image–query pair 数据链路

本实现把 COCO 2017、LVIS v1、RefCOCO、RefCOCO+、RefCOCOg 和 PhraseCut 转成统一的 image–query pair：每个样本只有一张图、一条文本，以及该文本对应的一个或多个框和 mask。它用于 SteerSAM 的 positive-only 图像训练；不改变原始数据或 `data/` 中的软链接。

入口、历史配置和正式评测边界统一见 [steersam_workflows.md](./steersam_workflows.md)。训练参数以实际 Hydra 合成配置及实验目录保存的配置快照为准，不以历史 smoke 数值作为默认值。

## 构建

在仓库根目录、`sam3` conda 环境中运行：

```bash
CUDA_VISIBLE_DEVICES="" python scripts/build_steersam_pairs.py \
  --output data/steersam_pairs/pairs.sqlite
```

构建器读取现有 `data/coco`、`data/lvis`、`data/PhraseCutDataset`，产生 SQLite 随机访问索引和同名的 `pairs.audit.json`。图像路径相对 `data/`，不会复制图像。完整索引应包含 COCO/LVIS 的 train、val，RefCOCO/RefCOCO+ 的 UNC train、val、testA、testB，RefCOCOg 的 UMD train、val、test，PhraseCut 的 train、val、test。COCO/LVIS 不生成无真值的 test pair。

所有 split 使用同一准入规则：图像可打开且尺寸与标注一致；文本非空，按 SAM3 自带 CLIP-BPE 计数加 SOT/EOT 不超过 32；每个目标有有效框、可解码非空 mask；丢弃 crowd；每条 pair 至少一个目标且至多 200 个目标。PhraseCut/VG 的框偶有少量坐标越出图像边界，若每个方向的越界不超过对应图像尺寸的 5%，先裁到图像范围，并在目标中保留 `original_bbox`；越界更大或裁后面积为零则剔除。审计记录裁框目标数。polygon 和未压缩 RLE 在构建时转成压缩 RLE。不会从框伪造 mask。一个必要目标损坏时，整条 pair 被剔除。PhraseCut 的 `ann_ids` 是源注释 ID，未必逐一区域对应；逐区域框与 polygon 才是目标列表，源 ID 原样保留供追溯。

同 split、同图、同规范化文本的记录：目标标注完全相同则只保留先遇到的；目标数量不同则保留更多目标的版本；数量相同但标注不同则保留两条并在审计中计数。规范化只用于去重键，不改写实际训练文本。跨 split 不因同图而去重；只有训练 pair 与 val/test 的图像、规范化文本、目标标注全相同时，才删除训练副本。内部 val/test 经过质量过滤，不能直接用其结果声称复现各数据集官方全量指标。

`pairs.audit.json` 记录源标注文件路径、字节数和修改时间，各来源各 split 的候选数、准入前/最终数量、BPE 长度分布、剔除原因，以及重复和冲突计数。训练上限属于配置而非索引；修改 YAML 中的上限后，运行下面的命令生成 `pairs.selection.json`，记录每来源上限、合格训练 pair 数和实际选中数：

```bash
CUDA_VISIBLE_DEVICES="" python scripts/report_steersam_selection.py \
  --config sam3/train/configs/steersam/steersam_six_dataset.yaml
```

`--max-source-pairs 10` 仅用于构建链路的快速 CPU 检查，生成的是不完整索引，绝不可用于正式训练或评测。

新索引在 SQLite metadata 内记录 `build_state` 与构建上限：完整构建为 `complete`，限制原始候选的调试构建为 `partial`；未完成构建为 `building`。loader 默认拒绝 partial/未完成索引。调试 partial 索引必须显式设置 loader 的 `allow_partial_index: true`，不是修改 `max_train_pairs`（后者只对完整索引做运行时抽样）。旧完整索引继续通过相邻 `pairs.audit.json` 的完整性标记读取，无须改写或重建；复制旧索引时要同时复制审计文件。不要覆盖正在训练进程使用的索引；构建新版本时使用新的输出文件名。

同图同文本、目标数相同但标注不同的记录不会被偷偷合并。若要逐组复核这些冲突，执行 `CUDA_VISIBLE_DEVICES="" python scripts/inspect_steersam_conflicts.py`，得到包含 pair ID、来源、图像路径和原始目标 ID 的 `pairs.conflicts.jsonl`；详细框和 RLE 仍可凭 pair ID 回查 SQLite `payload`。

## 训练与抽样

独立配置为 [steersam_six_dataset.yaml](../sam3/train/configs/steersam/steersam_six_dataset.yaml)。构建全量索引后，运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py \
  -c configs/steersam/steersam_six_dataset.yaml \
  --use-cluster 0 --num-gpus 2
```

`scratch.max_train_pairs` 有六个来源参数：`null` 使用全部合格训练 pair，`0` 关闭该来源，正整数以 `scratch.pair_selection_seed` 和稳定 pair ID 做确定性抽样。它们在构建后的质量筛选和去重之后生效；不设额外来源均衡权重，所以每个 epoch 的来源比例就是选中 pair 的数量比例。当前实验可能对任意来源显式设置上限，不应仅根据文件名判断是全量训练。训练启动时 loader 打印每来源实际数量。val/test 不受这些训练上限影响。selection 报告脚本支持 Hydra `defaults`，mini 等继承配置也可传入；为不同实验指定不同的 `--output`，避免覆盖历史抽样记录。

模型仍使用当前 strategy-2 冻结设置：固定 ViT trunk、语言 backbone 和 post-FPN 任务栈，训练 steering adapters、patch head 和 FPN。训练 loss 是原 SAM3 任务 loss 加 patch loss；内部联合 val 在 eval-mode 下计算 patch loss 和 PMASS。一个 epoch 遍历全部选中 pair，故全量六来源 epoch 可能非常长。训练期 val 是过滤后的合并 val；如需单来源诊断，使用同一个索引但为 val loader 指定 `source`，不要把该内部指标当作官方协议成绩。

六个 `steersam_internal_val_{source}.yaml`（`source` 为 `coco`、`lvis`、`refcoco`、`refcoco_plus`、`refcocog`、`phrasecut`）提供单来源过滤后 val。它们要求训练输出目录存在 `checkpoints/checkpoint.pt` 并加载它；没有 checkpoint 时直接报错，避免把随机初始化的 steering 模块误当训练结果。这里只计算内部 patch loss/PMASS，不计算官方 AP 或 referring 指标。例如：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py \
  -c configs/steersam/steersam_internal_val_phrasecut.yaml \
  --use-cluster 0 --num-gpus 2
```

## 验收和边界

CPU 单元测试：

```bash
CUDA_VISIBLE_DEVICES="" pytest -q tests/train/test_unified_pair_index.py
```

在全量索引构建完成后，可先跑双 GPU mini。mini 是小规模实验，不是官方评测：当前每来源上限为 200 条、内部 val 取 60 条，训练 10 epoch；batch、resolution、checkpoint 保存设置继承主配置。它不再等同于历史“不保存 checkpoint、每来源 2 条”的 smoke。若只验证接口，另行降低每来源上限、batch 和 epoch，并设置 `skip_saving_ckpts: true`；已有 mini 输出目录有 checkpoint 时会自动恢复，重新开始应使用新的实验目录。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py \
  -c configs/steersam/steersam_six_dataset_mini.yaml \
  --use-cluster 0 --num-gpus 2
```

完整性审计重点查看 `pairs.audit.json` 的 `final_counts`、`text_over_32_bpe`、`missing_corrupt_or_size_mismatch_image`、mask/框相关剔除、`equal_count_different_annotation_pairs_retained` 和 `cross_split_exact_training_duplicates_removed`。训练前还应各抽一条 COCO 多实例、LVIS 非穷尽类别、RefCOCO 单实例、PhraseCut 多区域样本，核对文本、实例数、mask 与框，并检查 collator 的 `img_ids == arange(P)`。

正式 COCO/LVIS/RefCOCO/PhraseCut/SA-Co-GOLD 比较，继续使用各自原始评测协议及完整原始划分。此索引只服务统一训练和内部诊断。

## 当前全量构建的验收记录（2026-09-28）

| 来源 | train | val | test / testA / testB |
|---|---:|---:|---|
| COCO | 325,408 | 13,855 | 不生成 |
| LVIS | 354,865 | 69,760 | 不生成 |
| RefCOCO | 112,602 | 10,265 | testA 5,349；testB 4,794 |
| RefCOCO+ | 103,932 | 9,418 | testA 4,801；testB 4,455 |
| RefCOCOg | 79,690 | 4,881 | test 9,575 |
| PhraseCut | 308,439 | 19,477 | test 14,343 |

总计 1,455,909 条 pair（训练 1,284,936；内部 val 127,656；test 类划分合计 43,317）。SQLite `quick_check` 通过，18 个来源/split 的实际计数与审计一致。所有记录的目标数组长度、1–200 目标上限、32 BPE 上限及来源/split 字段已逐条核对；3,047,066 个目标的框范围、正面积及 mask 尺寸元数据均无违例。额外完成了 200,083 个不同图像路径的完整像素解码检查，失败数为 0；构建器今后也直接使用 `Image.load()`，而非只检查图像文件头。

构建验收时相关 CPU 测试 20/20 通过；最终索引的六来源 collator 检查满足 `img_ids == arange(6)`，真实 PhraseCut 裁框双目标样本产生两个有效 mask。全量索引抽样在同种子下完全一致，换种子会改变选样。历史双 GPU mini 已完成每卡 3 个训练 step 和 3 个内部 val step；另用临时训练 checkpoint 验证了单来源 val 的恢复链路。这些是验收时的短跑记录，不代表随后修改参数后的配置也完成了同等 GPU 验证；后续正式训练状态以实际输出目录的日志为准。

当前冲突明细包含 68,742 个最终保留的同图同文本、同目标数但不同标注的组（138,509 条 pair）。审计中的冲突计数是构建时发生的比较次数，包含后来被更多目标版本替换的记录，不能等同于最终冲突组数。

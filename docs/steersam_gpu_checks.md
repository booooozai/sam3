# SteerSAM GPU 验证清单

> 状态：2026-09-24 已完成 Check A、B、C；正式训练 Check D 仍由实验计划决定是否启动。
> 环境：`conda activate sam3`，在仓库根目录 `/home/chenshengbo/workspace/sam3` 下运行。
> CPU 回归（已通过）：`CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q` → 55 passed。

> 上述日期、测试数量和显存为历史 COCO smoke 记录，不是当前六数据集配置参数。当前入口见 [steersam_workflows.md](./steersam_workflows.md)；改变分辨率、batch 或冻结策略后需要新的 GPU 验证。本轮维护仅执行 CPU 测试，没有重跑这些 GPU 检查。

## 已执行结果（GPU 2、3）

- Check A：普通 SAM3 checkpoint 成功装入 672 分辨率的 SteerSAM。关闭 steering 时与 baseline **逐元素完全一致**；`alpha=0` 时最大绝对误差为 `0.0`；将 gate 设为非零后输出发生变化，交换文本条件也产生不同输出（最大绝对差 `0.5138338`）；所有输出有限。峰值 allocated/reserved 显存分别约 `6.847/6.943 GiB`。
- Check B：2 卡、2 epoch、共 92 个训练 step 完成；epoch 平均 loss 从 `434.2033` 降至 `364.6846`，未出现非有限 loss/gradient。训练峰值显存约 `25 GiB/GPU`，稳定值约 `23 GiB/GPU`；截断 val 正常跑完 46 step，约 `10 GiB/GPU`。
- checkpoint 审计：4 个 steering gate 均离开零点；epoch 1→2 的 adapter query 投影、FPN 最大权重变化分别约 `1.33e-3`、`1.56e-4`，冻结的 ViT trunk 与 language backbone 权重变化均为 `0.0`；optimizer 含 439 个状态项、5 个参数组。
- Check C：原命令成功恢复 `/tmp/steersam_smoke/checkpoints/checkpoint.pt`，恢复状态为 `epoch=2`，随后完成 46-step final val 并以退出码 0 结束。各 worker 现会在成功结束时同步并显式销毁 process group，未再出现 NCCL/TCPStore 退出告警。
- 训练环境变量日志现会对 API key、token、password、secret、credential、auth 等凭据类变量进行脱敏。

## Check A：checkpoint 加载 + 零门控 parity + prompt sensitivity（单卡，约 5 分钟）

目的：验证设计文档 §7.11/§12 的三条验收标准——

1. 普通 SAM3 checkpoint 装入 SteerSAM 后，只允许两类 missing key：新增的 `steering_adapters.*`，以及加载前主动跳过、由 672 模型重新生成的 `*.freqs_cis`；不得出现其他不兼容 key；
2. `alpha=0` 时，带 steering 输入的 backbone 输出与基线 SAM3 在容差内一致（steering_factor=0 与关闭开关两条路径都检查）；
3. gate 非零后，不同文本应产生不同视觉输出。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python - <<'EOF'
import torch
from sam3.model_builder import build_sam3_image_model

CKPT = "pretrained_weights/sam3/sam3.pt"
BPE = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
common = dict(bpe_path=BPE, checkpoint_path=CKPT, load_from_HF=False,
              device="cuda", eval_mode=True, resolution=672, enable_segmentation=False)

base = build_sam3_image_model(**common)
steer = build_sam3_image_model(**common, enable_steering=True)
base.eval(); steer.eval()

torch.manual_seed(0)
img = torch.randn(2, 3, 672, 672, device="cuda")
text = torch.randn(2, 32, 1024, device="cuda")
pad = torch.zeros(2, 32, dtype=torch.bool, device="cuda"); pad[0, 24:] = True

with torch.no_grad():
    out_base = base.backbone.forward_image(img)["vision_features"]
    out_off  = steer.backbone.forward_conditioned_image(img)["vision_features"]
    out_zero = steer.backbone.forward_conditioned_image(
        img, steering_text=text, steering_padding_mask=pad)["vision_features"]

print("steering-off == baseline :", torch.equal(out_off, out_base),
      "| allclose:", torch.allclose(out_off, out_base))
print("alpha=0      == baseline :", torch.allclose(out_zero, out_base, atol=1e-4, rtol=1e-4))

with torch.no_grad():
    for a in steer.backbone.vision_backbone.trunk.steering_adapters.values():
        a.alpha += 0.5
    out_steer = steer.backbone.forward_conditioned_image(
        img, steering_text=text, steering_padding_mask=pad)["vision_features"]
print("alpha>0  != baseline      :", not torch.allclose(out_steer, out_base))
print("all finite:", torch.isfinite(out_steer).all().item())
EOF
```

预期：三行依次为 `True`、`True`、`True`（外加 `all finite: True`）。
构建阶段预期列出 32 个主动跳过的 `*.freqs_cis` 和 28 个新增 adapter key；SteerSAM checkpoint 校验不应报告 unsupported incompatibilities。released checkpoint 中未构造的 `sam2_convs.*`，以及本检查显式关闭的 `segmentation_head.*`，是按模型构建状态允许的 extra key。

## Check B：短程训练 smoke run（2 卡，约 30–60 分钟，含一次截断 val plumbing）

目的：验证设计文档 §7.9/§7.10 的 GPU 侧链路——bf16 AMP、DDP、trainable-only 优化器参数组、梯度裁剪、val forward/loss 和 checkpoint 保存。

以下命令使用保留的 COCO smoke 入口。当前文件为 train 32 个正 pair、val 32 个正 pair、1 epoch、每卡 batch 16，输出目录为 `/tmp/steersam_strategy2_b16_smoke_20260925`；本页开头的 2048 pair/2 epoch 等结果属于更早的配置快照，不是此命令当前的参数。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py -c configs/coco/coco2017_steersam_mask_smoke.yaml --use-cluster 0 --num-gpus 2
```

预期检查：
- 启动日志：`... steering adapters 16,785,412/16,785,412 trainable`，参数组只有 2 个 lr 规则生效（无 frozen 0-LR 组）；
- 每步 `Losses/train_all_loss` 为有限值且总体下降；无 "Non-finite gradient norm" 告警；
- 检查 ViT 原参数无梯度、FPN 与 gate 有梯度；零 gate 下第一步 CA 投影梯度为零属于预期，gate 更新后再确认 CA 投影梯度/权重开始变化；
- `Mem (GB)` 数值（衡量 adapter 带来的额外显存，对比同机器 frozen-backbone run 的 ~31–40 GB）；
- epoch 结束后当前配置选中的 32 个 positive pair 的 val forward/loss 正常完成；该截断集合不输出、也不解释为 COCO AP；
- 当前 `paths.experiment_log_dir` 下的 `checkpoints/checkpoint.pt` 生成；使用新实验目录，避免意外恢复历史运行。

## Check C：断点续训（2 卡，几分钟）

目的：验证 SteerSAM checkpoint 可完整 resume（含 adapter/gate 与优化器状态）。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py -c configs/coco/coco2017_steersam_mask_smoke.yaml --use-cluster 0 --num-gpus 2
```

（Check B 结束后原样重跑一次即可。）若 checkpoint 已完成配置中的全部 epoch，预期跳过额外训练并完成 final val，而不是从头开始。恢复日志的 epoch 以实际 checkpoint 为准，不能套用历史 `epoch: 2`。若要验证“恢复后继续优化”的 loss 连续性，将 `trainer.max_epochs` 调整到超过已完成 epoch 的值后再运行；不要删除原 checkpoint。

## Check D：正式实验启动前确认（人工）

- smoke 通过后，用当前六数据集配置启动。train 和 train-time val 均使用 positive pair；联合内部 val 报告 patch loss/PMASS/有效监督比例，不在训练进程内运行正式 AP：
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore python sam3/train/train.py -c configs/steersam/steersam_six_dataset.yaml --use-cluster 0 --num-gpus 2`；
- 先按实际 resolution 和 batch 测量显存，接近上限时降低 batch。标准 `collate_fn_api` 不支持只增大 `scratch.gradient_accumulation_steps`；必须保持 1，或先配套实现返回 micro-batch list 的 collator；
- 训练若干百步后，可抽查 gate 是否离开零点：
  `CUDA_VISIBLE_DEVICES="" python -c "import torch; ck=torch.load('/tmp/steersam_strategy2_b16_smoke_20260925/checkpoints/checkpoint.pt', map_location='cpu'); sd=ck.get('model', ck); print({k: float(v) for k,v in sd.items() if k.endswith('.alpha')})"`
  （键名结构以实际 checkpoint 为准；预期 alpha 逐步偏离 0）。

## Check E：独立正式评估（2 卡，长时间任务）

独立 eval 穷举每图全部 80 类（400,000 pairs），只报告标准 COCO bbox/mask AP。旧 prompt-pair cgF1 分支已从正式配置退出：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 PYTHONWARNINGS=ignore \
python sam3/train/train.py -c configs/coco/coco2017_steersam_mask_eval.yaml \
  --use-cluster 0 --num-gpus 2
```

运行前确认配置的 checkpoint 指向实际训练结果，默认值可能仍是历史实验路径。配置使用 `val_batch_size=8`；若 mask 后处理仍接近显存上限，可以只降低 eval 配置的 batch，不影响训练 effective batch。标准 AP 分支在线保留每张源图 top 100；正式入口不再包含 prompt-pair 指标分支。

## 可选 Check F：`enable_steering=False` 回归

目的：确认改动后原 SAM3 基线完全不受影响。任选一个现有配置短跑，例如：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 PYTHONWARNINGS=ignore \
python sam3/train/train.py -c configs/coco/coco2017_full_ft_mask_frozen_backbone.yaml \
  --use-cluster 0 --num-gpus 1
```

预期：构建日志与改动前一致（无 steering 行），前若干步 loss 行为与正在运行的 frozen-backbone run 同量级。

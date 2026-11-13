# TripoSG vs TripoSR 对比（衣服质感用哪个？）

> 结论先说：**衣服质感 TripoSR 更合适**（能出带贴图的 GLB），但两个都只能一次生成
> “穿着衣服的整个人”，都生成不了单独的衣服模型，也都不带骨骼。所以换装变体
> 的做法不变：同一人物用不同服装参考图各生成一个着装变体。
> **后端没有改**——当前默认仍是 TripoSG；要不要加 TripoSR 后端，等用户在
> 4060 Ti 上实测后再定。

## 对比

| | TripoSG（当前默认） | TripoSR（候选） |
|---|---|---|
| 仓库 | `VAST-AI-Research/TripoSG` | `VAST-AI-Research/TripoSR`（Tripo AI + Stability AI） |
| 许可证 | MIT | MIT（代码和权重都是） |
| 模型 | 约 15 亿参数，rectified flow transformer | 约 4.2 亿参数，`model.ckpt` 1.68 GB，单次前向 LRM |
| 显存 | 至少 8 GB（4060 Ti 8GB 刚好够，一次一个人） | 默认约 6 GB（8GB 卡更宽松） |
| 速度 | 扩散式，慢（分钟级） | A100 上 <0.5 秒/张；消费卡约几秒 |
| 贴图 | **无**：只出形状，GLB 没有贴图 | **有**：默认是顶点色 `.obj`；加 `--bake-texture`（+ `--model-save-format glb`）可出带贴图的 GLB，贴图分辨率默认 2048 |
| 几何质量 | 更好（官方定位高于 TripoSR） | 较弱，细节少，三角面多（约 20 万+），进 DCC 前要清理/重拓扑 |
| 入口 | `python -m scripts.inference_triposg --image-input <图> --output-path <out.glb>` | `python run.py <图> --bake-texture --model-save-format glb --output-dir output/` |
| 安装注意 | Python 3.10，conda 环境，权重首次运行从 Hugging Face 下载 | Python 3.8+；`requirements.txt` 会编译 `torchmcubes` CUDA 扩展，装时要能编译 |
| 骨骼 | 无 | 无 |

## 对衣服质感意味着什么

- TripoSG 的变体 GLB 是“素模”：衣服的形状对，但颜色/花纹出不来，装配区只能看个轮廓。
- TripoSR + `--bake-texture` 的变体 GLB 带贴图：衣服颜色和图案能看出来，换装变体在装配区才真的像“换了件衣服”。
- 代价是几何更糙：脸和身体细节不如 TripoSG，衣服边角可能糊；还要多花 CPU/内存做 xatlas 展 UV、烘焙 2048 贴图。

## 待定项（要用户在 Windows + 4060 Ti 上实测后才能定）

1. TripoSR 在 4060 Ti 8GB 上跑全身人物图是否稳定（`--bake-texture` + `--mc-resolution 512` 时显存/内存够不够）。
2. 同一张全身照，TripoSR 和 TripoSG 出的变体哪个更像本人、衣服还原度如何（脸漂移是已知风险）。
3. 烘焙贴图的变体 GLB 能否被现有 `validate_glb` 和装配流程直接吃下（应可以，都是标准 GLB，但没实测）。
4. 要不要把 TripoSR 做成第二个后端（配置 `model3d.backend: triposr` + `triposr_repo`），还是只作为手动出变体的备选——等上面三条有结果再定，**不要先改后端**。

## 共同的诚实边界（两个后端都一样）

- 只能整身生成，不能单独生成衣服；衣服 = 换装变体。
- 不带骨骼：生成的 GLB 不能直接用于身体重渲染（需要带骨骼的模型或先自动绑骨，另行试点）。
- 人物全身照质量决定一切：抠像干净、正面、光线均匀的全身照效果最好；侧身、遮挡、多人同框的图不要拿来生成。

# FedGCDR-AKF

FedGCDR 与 FR-CSU 知识融合的组合实验工作区，研究用户或场景条件下的动态融合权重。

## 当前状态

已拉取 FedGCDR 上游，固定到 `09c63ef5b36903dfbadee86b31aabdcee7f98205`，保留 22 条提交历史；原始工作区不作研究修改。项目专用 `.venv` 已安装依赖。`research/fedgcdr_akf/` 已实现上游 GAT/MLP 模块适配、CSV 时间划分、联邦模拟、固定/全局可学习/用户级及物品条件动态 gamma、验证选模与最终测试。具体验证结果见 `results/implementation_validation.json`。

本地 `ratings-full/` 有 16 个品类 CSV，合计 9,024,418,271 字节。初始清单检查每文件前 10,000 条；运行试验另按配置读取 3 个品类各前 100,000 条。试运行不代表全量实验或已验证的新方法，项目名称仍为临时工作名称。

## 快速运行

首次获取项目时，一并拉取固定版本的上游子模块：

```bash
git clone --recurse-submodules https://github.com/llllllhui/FedGCDR-AKF.git
cd FedGCDR-AKF
python3.14 -m venv .venv
```

已有克隆可运行 `git submodule update --init --recursive` 补齐上游代码。原始评分数据需自行放入 `ratings-full/`，不随仓库分发。

在项目根目录运行，使用 Python 3.14 的项目独立环境；本次验证平台为 macOS arm64 / CPU。

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m research.fedgcdr_akf prepare --config configs/pilot.json --output data/my_pilot
python -m research.fedgcdr_akf run --config configs/pilot.json --prepared data/my_pilot/prepared.json --output runs/my_pilot
```

输出目录必须尚不存在，避免覆盖已有数据与结果。当前 `configs/pilot.json` 是有界运行配置：Movies_and_TV 为目标域，Books 和 CDs_and_Vinyl 为源域，每域前 100,000 行、256 名根据训练数据选定的目标用户，仅用于实现验证。不能据此选择正式实验的域组合或宣称方法有效。

上游来源和文件哈希见 `upstream/source_manifest.json`，方法接口、运行产物及适配边界见 `docs/implementation.md`。依赖锁记录本次实际环境，不代表已验证其他 Python、CUDA 或服务器环境。

## 本地小规模对比

`configs/local_small.json` 在固定数据划分上比较 7 种方法和 3 个训练种子，最多同时运行两个本地 CPU 进程。每域前 1,000,000 行、最多 2,048 名训练用户；目标域 Movies_and_TV，源域 Books 与 CDs_and_Vinyl。协议见 `docs/local_comparison_protocol.md`。

```bash
source .venv/bin/activate
python -m research.fedgcdr_akf prepare --config configs/local_small.json --output data/my_small
python -m scripts.local_comparison --config configs/local_small.json --prepared data/my_small/prepared.json --output runs/my_small --report results/my_small
```

固定权重两组分别在预先规定的 gamma 候选值上按验证集选择，每方法每种子仅对选定模型执行一次最终测试。聚合报告包含全部方法、种子均值与标准差、有源知识分组和训练流行度参考；JSON 保留选模次数审计。原始日志、映射和检查点仍保存在忽略的运行目录中。

2026-10-28 已完成本轮比较：2,048 名训练用户、229 名验证用户、217 名测试用户，7 种模型 × 3 个种子，固定权重两组各搜索 3 个候选。结果与解释见 [本地比较报告](results/local_small_20261028.md)，机器可读明细见 `results/local_small_20261028.json`。用户级动态 gamma 相对固定权重有本子集内的改善，但未超过图迁移适配对照；物品条件增益不稳定。完整运行保存在 `runs/local_small_20261028/`。

用户随后要求增加轮次并仅在本机训练。新增 `configs/local_long.json` 将各域预训练提高到 10 轮、目标融合提高到 50 轮，复用原划分和其余实验条件。该轮以本机 CPU 后台任务运行，最多两个训练进程；运行状态以 `runs/local_long_20261028/status.json` 为准，启动记录见 `results/local_long_20261028_launch.json`，协议见 `docs/local_long_protocol.md`。完成后自动汇总到 `results/local_long_20261028.md`、`.csv` 和 `.json`。

## 已确认的研究方向

- 联邦推荐、跨域、多源域。
- 借鉴 FR-CSU 的本地知识与迁移知识融合思路，保留固定 gamma 作为基线。
- 将数据驱动的动态 gamma 正式列为研究目标：已实现用户级权重及候选物品条件权重，正式效果实验尚待完成。当前每次运行只有一个目标域，因此域条件通过各目标域独立模型表达，尚无跨目标域共享的域嵌入。
- 区分原方法与本项目扩展：当前参考记录中 FR-CSU 的 gamma 为超参数；动态权重是本项目的研究扩展，不能归因于 FR-CSU 或将实现验证描述为效果验证。
- 用户已放弃此前的设定；完全无重叠用户、每域一个客户端、所有域同时受益均不作为本项目的默认约束。

## 候选组合方式

1. 使用 FedGCDR 提取、映射和聚合多源跨域用户知识。
2. 保留目标域本地推荐分支，学习本地用户与物品偏好。
3. 借鉴 FR-CSU，以融合层组合本地预测与跨域补充预测，先建立固定 gamma 对照：

   `prediction = f2(local_score + gamma * f1(concat(transferred_user, target_item)))`

4. 将 gamma 替换为可学习门控。当前用户级输入为本地用户向量、映射源知识扩展后的目标图用户向量和 `log(1 + 本地训练交互数)`；物品条件模式额外使用候选物品向量。目标域由本轮模型确定；只有模拟协议允许的表示和本地训练统计进入门控。

这是待正式实验验证的组合设计，并非任一原论文公式的逐字复写。FedGCDR 已有本地与源域图信息融合，因此新增预测分支、动态权重的独立贡献与创新性仍需核验。实现已提供固定 gamma、全局可学习 gamma 和参数量接近的固定权重对照。

## 已到位的数据

- 原始文件：`ratings-full/` 下 16 个品类 CSV，保留原文件，不纳入版本控制。
- 已检查样本：每文件前 10,000 条记录，共 160,000 条；均为四列，样本评分范围为 1–5，未发现结构或数值解析错误。
- 字段暂按 `item_id, user_id, rating, unix_timestamp_seconds` 解释，语义及具体数据来源、版本仍需核验。
- 全量交互数、用户与物品数、重复记录、时间范围和跨域用户重叠尚未统计，不能用文件头样本推断。
- 数据清单与检查边界见 `docs/data_inventory.md`，机器可读记录见 `results/dataset_inventory.json`。

## 待确定的实验条件

- 在现有 16 个品类中选择源域与目标域；先完成数据来源、属性及全量统计核验。
- 用户重叠比例和跨域对齐方式。
- 联邦客户端定义，以及哪些参数允许交换。
- 正式实验的门控输入消融、验证集调参预算及多目标域共享方式。
- 是否将训练完成后新用户的本地适配列为主要研究问题。
- 源域模型冻结与更新策略、融合层的训练流程。

## 目录

- `research/`：研究实现。
- `upstream/FedGCDR/`：固定版本的上游原始仓库与历史。
- `configs/`：实验配置。
- `scripts/`：数据准备与运行入口。
- `docs/`：方法记录和实验方案。
- `data/`：本地数据，不纳入版本控制。
- `ratings-full/`：用户已提供的原始评分 CSV，不纳入版本控制。
- `runs/`：运行日志、完整结果与检查点，不纳入版本控制。
- `results/`：可供审阅的小型结果汇总。
- `tests/`：动态门控、数据时间边界、聚合和端到端验证。

## 环境与研究约定

安装依赖、运行项目或测试前，在本项目下创建并激活 `.venv`，确认解释器和依赖路径；不向全局运行时安装依赖。保留上游代码与来源，研究改动放在 `research/`。超参数和检查点只根据训练/验证数据选择，最终测试集保持独立。联邦部署本身不等于形式化隐私保证。Git 提交使用中文标题和分条正文。

## 参考

- FedGCDR：<https://arxiv.org/abs/2410.08249>
- 官方代码：<https://github.com/LafinHana/FedGCDR>
- FR-CSU：用户上传论文《Personalized Federated Recommendation for Cold-Start Users via Adaptive Knowledge Fusion》。当前参考文件为匿名稿，发表信息待核验。

初步实验比较见 `docs/experiment_plan.md`。

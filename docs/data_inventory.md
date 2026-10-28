# 本地数据清单

检查日期：2026-10-27。原始文件位于 `ratings-full/`，共 16 个 CSV，合计 9,024,418,271 字节，约 9.02 GB（8.40 GiB）。文件已到位不代表已经完成数据清洗、跨域对齐或训练。

## 检查范围

- 对每文件前 10,000 条 CSV 记录进行结构及数值解析检查，共检查 160,000 条记录。
- 这些样本均为四列，无表头；暂按 `item_id, user_id, rating, unix_timestamp_seconds` 解释。字段语义与数据来源、版本尚待核验。
- 每个文件的样本评分范围均为 1–5，未发现列数、空标识、非有限评分或时间戳解析错误。
- 文件头样本不是随机样本，不用于推断全量质量、评分分布、活跃度、时间范围或跨域重叠。
- 检查使用本项目 `.venv` 和 Python 标准库，未安装第三方依赖。

机器可读清单及样本时间范围见 `../results/dataset_inventory.json`；其中所有带 `sample_` 的统计仅描述上述文件头样本。

## 文件

| 品类文件 | 文件大小（字节） |
| --- | ---: |
| Automotive.csv | 333,489,688 |
| Books.csv | 2,140,933,459 |
| CDs_and_Vinyl.csv | 189,332,114 |
| Cell_Phones_and_Accessories.csv | 420,031,077 |
| Clothing_Shoes_and_Jewelry.csv | 1,347,817,661 |
| Electronics.csv | 876,247,561 |
| Grocery_and_Gourmet_Food.csv | 211,781,981 |
| Home_and_Kitchen.csv | 915,245,469 |
| Kindle_Store.csv | 238,886,692 |
| Movies_and_TV.csv | 365,735,607 |
| Office_Products.csv | 232,942,890 |
| Patio_Lawn_and_Garden.csv | 218,538,436 |
| Pet_Supplies.csv | 273,071,383 |
| Sports_and_Outdoors.csv | 541,795,338 |
| Tools_and_Home_Improvement.csv | 376,265,482 |
| Toys_and_Games.csv | 342,303,433 |

## 全量审计与实验准备待办

1. 核验数据集来源、版本、字段含义和跨域用户标识的一致性。
2. 计算文件 SHA-256、全量行数、用户数、物品数、缺失及异常记录、重复用户物品记录、完整时间范围。
3. 统计跨域用户重叠矩阵；不得用前 10,000 条样本的交集替代全量结果。
4. 明确联邦客户端定义、用户对齐方式及允许交换的信息，据训练数据确定多源域与目标域。
5. 固定时间划分和反馈处理规则，门控特征只使用预测时可用的训练信息。

用户级动态 gamma 可以在核实后的评分交互与可用表示上设计。候选物品或目标域条件可作为场景扩展；设备、地点等上下文需要额外数据，当前不默认具备。

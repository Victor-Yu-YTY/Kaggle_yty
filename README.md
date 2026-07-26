## Spaceship Titanic 项目以及代码说明

比赛本质为：二分类问题

主要处理带有缺失值、数值字段和文字字段的表格数据

## data文件说明

背景：时间设定在公元 2912 年。名为 Spaceship Titanic 的星际客轮搭载乘客前往多个系外行星。在飞船经过半人马座阿尔法星附近时，遭遇时空异常。飞船本身没有被摧毁，但接近一半的乘客被传送到了另一个维度。你的任务是利用飞船系统中保存的乘客信息，预测哪些乘客被传送了。

本质上是一个标准的监督学习问题：

```commandline
乘客信息 → 机器学习模型 → 是否被传送
```

目标变量：Transported

任务：对 test.csv 中的每一名乘客，预测：
- Transported = True
- Transported = False

任务分解：
- 输入 X：乘客的个人信息、舱位、行程和消费记录
- 输出 y：Transported

最终比赛以准确率进行评分：
```commandline
准确率 = 预测正确的乘客数 ÷ 总乘客数
```

#### 训练集 train.csv

| 字段名称 | 中文名称 | 数据类型 | 字段含义 |
|---|---|---|---|
| `PassengerId` | 乘客编号 | 字符串 | 每位乘客的唯一编号，格式为 `gggg_pp`。其中 `gggg` 表示旅行小组编号，`pp` 表示乘客在该小组中的编号。同一小组的乘客通常一起旅行。 |
| `HomePlanet` | 出发星球 | 类别型 | 乘客出发前长期居住的星球，主要取值包括 `Earth`、`Europa` 和 `Mars`。 |
| `CryoSleep` | 是否进入低温休眠 | 布尔型 | 表示乘客在航行期间是否进入低温休眠状态。取值为 `True` 或 `False`。进入休眠的乘客通常不会产生船上消费。 |
| `Cabin` | 舱位编号 | 字符串 | 乘客所在的舱位，格式为 `Deck/Number/Side`，例如 `B/0/P`。其中 `Deck` 表示甲板，`Number` 表示舱位编号，`Side` 表示飞船所在一侧。 |
| `Destination` | 目的地 | 类别型 | 乘客计划前往的目的星球，主要包括 `TRAPPIST-1e`、`55 Cancri e` 和 `PSO J318.5-22`。 |
| `Age` | 年龄 | 数值型 | 乘客的年龄。 |
| `VIP` | 是否为 VIP | 布尔型 | 表示乘客是否在航程中购买了 VIP 特殊服务。取值为 `True` 或 `False`。 |
| `RoomService` | 客房服务消费 | 数值型 | 乘客在客房服务方面产生的消费金额。 |
| `FoodCourt` | 餐饮区消费 | 数值型 | 乘客在飞船餐饮区产生的消费金额。 |
| `ShoppingMall` | 购物中心消费 | 数值型 | 乘客在飞船购物中心产生的消费金额。 |
| `Spa` | SPA 消费 | 数值型 | 乘客在飞船 SPA 服务方面产生的消费金额。 |
| `VRDeck` | 虚拟现实娱乐消费 | 数值型 | 乘客在虚拟现实娱乐设施方面产生的消费金额。 |
| `Name` | 乘客姓名 | 字符串 | 乘客的姓名。后续可以从中提取姓氏，用于识别家庭成员或同行关系。 |
| `Transported` | 是否被传送 | 布尔型 | 模型需要预测的目标变量。`True` 表示乘客被传送到另一个维度，`False` 表示乘客没有被传送。该字段只存在于训练集 `train.csv` 中。 |

## 01_inspect_data 检查数据

数据规模：
- 训练集形状： (8693, 14)
- 测试集形状： (4277, 13)
- 提交示例形状： (4277, 2)

#### 缺失值分析

| 字段名称 | 中文名称 | 缺失数量 | 缺失比例 |
|---|---|---:|---:|
| `CryoSleep` | 是否进入低温休眠 | 217 | 2.50% |
| `ShoppingMall` | 购物中心消费 | 208 | 2.39% |
| `VIP` | 是否为 VIP | 203 | 2.34% |
| `HomePlanet` | 出发星球 | 201 | 2.31% |
| `Name` | 乘客姓名 | 200 | 2.30% |
| `Cabin` | 舱位编号 | 199 | 2.29% |
| `VRDeck` | 虚拟现实娱乐消费 | 188 | 2.16% |
| `Spa` | SPA 消费 | 183 | 2.11% |
| `FoodCourt` | 餐饮区消费 | 183 | 2.11% |
| `Destination` | 目的地 | 182 | 2.09% |
| `RoomService` | 客房服务消费 | 181 | 2.08% |
| `Age` | 年龄 | 179 | 2.06% |
| `PassengerId` | 乘客编号 | 0 | 0.00% |
| `Transported` | 是否被传送 | 0 | 0.00% |

1. 除 `PassengerId` 和 `Transported` 外，其余字段均存在一定数量的缺失值。
2. 各字段的缺失比例均较低，大约在 `2.06%` 到 `2.50%` 之间。
3. 不建议直接删除所有包含缺失值的乘客记录，否则可能同时删除较多有效数据。
4. 后续需要根据字段类型分别处理缺失值：
   - 数值型字段可使用中位数填补；
   - 类别型字段可使用众数或 `Unknown` 填补；
   - `Cabin`、`PassengerId` 等结构化字段可先拆分，再分别处理缺失值；
   - 消费字段可结合 `CryoSleep` 判断是否适合填补为 `0`。
5. 目标变量 `Transported` 不存在缺失值，因此训练标签完整。

## 探索性数据分析（EDA）

主要目的：观察不同特征与 Transported 的关系强弱差异

#### 特征：CryoSleep（是否进入低温休眠）
![02_cryo_sleep](outputs/eda/02_cryo_sleep.png)

说明：进入低温休眠的乘客，被传送概率明显更高

#### 特征：NoSpend
![08_no_spend](outputs/eda/08_no_spend.png)

说明：完全没有消费的乘客更容易被传送

但是注意：NoSpend 与 CryoSleep 高度相关

因为处于低温休眠状态的乘客通常不会使用客房服务、餐厅、商场、SPA 和 VR 设施，因此总消费很可能为 0。

#### 特征：CabinDeck（不同甲板之间的传送率差异）
![05_cabin_deck](outputs/eda/05_cabin_deck.png)

#### 特征：CabinSide
![06_cabin_side](outputs/eda/06_cabin_side.png)

说明飞船左右两侧可能受到异常事件的影响不同。

#### 特征：HomePlanet
![01_home_planet](outputs/eda/01_home_planet.png)

Europa 乘客的传送率明显较高，Earth 乘客最低

#### 特征强弱总结

| 特征 | 观察到的差异 | 预测价值 | 后续建议 |
|---|---|---|---|
| `CryoSleep` | 约 82% 对 33% | 很强 | 必须保留 |
| `NoSpend` | 约 79% 对 30% | 很强 | 必须构造 |
| `CabinDeck` | 约 20%～73% | 较强 | 拆分 `Cabin` |
| `HomePlanet` | 约 42%～66% | 较强 | 类别编码 |
| `GroupSize` | 约 40%～64% | 中等 | 从 `PassengerId` 构造 |
| `CabinSide` | 约 45%～56% | 中等 | 拆分 `Cabin` |
| `Destination` | 约 47%～61% | 中等偏弱 | 类别编码 |
| `VIP` | 约 38%～51% | 较弱 | 可以保留 |

1. `CryoSleep` 和 `NoSpend` 与乘客是否被传送的关系最明显，是后续模型中的核心特征。
2. `CabinDeck` 不同类别之间的传送率差异较大，因此不能直接删除原始 `Cabin` 字段，应先将其拆分。
3. `HomePlanet` 具有较明显的区分能力，需要作为类别变量进行编码。
4. `GroupSize` 与传送结果存在非线性关系，可以从 `PassengerId` 中提取小组编号并计算同行人数。
5. `CabinSide` 和 `Destination` 有一定预测能力，但弱于 `CryoSleep`、`NoSpend` 和 `CabinDeck`。
6. `VIP` 的区分能力相对较弱，但仍可以保留，让模型判断其是否提供额外信息。

#### 原始字段处理方式

| 原始字段 | 处理方式 |
|---|---|
| `PassengerId` | 提取小组编号和小组人数后删除原始值 |
| `Cabin` | 拆分为 `Deck`、`Number`、`Side` 后删除原始字段 |
| `Name` | 第一个模型先删除 |
| 五项消费字段 | 保留，并增加 `TotalSpend` 和 `NoSpend` |
| 类别型缺失值 | 填充为 `Unknown` |
| 数值型缺失值 | 使用中位数填补 |

#### 具体字段处理说明

| 字段或字段组 | 具体处理 |
|---|---|
| `PassengerId` | 提取 `GroupId` 和 `GroupSize`，还可以构造 `IsAlone` |
| `Cabin` | 拆分为 `CabinDeck`、`CabinNumber` 和 `CabinSide` |
| `Name` | 基线模型中删除；后续可提取姓氏或家庭信息 |
| `RoomService` | 保留原始消费金额 |
| `FoodCourt` | 保留原始消费金额 |
| `ShoppingMall` | 保留原始消费金额 |
| `Spa` | 保留原始消费金额 |
| `VRDeck` | 保留原始消费金额 |
| 五项消费总和 | 构造 `TotalSpend` |
| 是否无消费 | 构造 `NoSpend`，当 `TotalSpend` 等于 0 时取 `True` |
| `HomePlanet` | 缺失值填充为 `Unknown`，然后进行类别编码 |
| `CryoSleep` | 缺失值填充为 `Unknown` 或使用众数填补，然后进行类别编码 |
| `Destination` | 缺失值填充为 `Unknown`，然后进行类别编码 |
| `VIP` | 缺失值填充为 `Unknown` 或使用众数填补，然后进行类别编码 |
| `CabinDeck` | 缺失值填充为 `Unknown`，然后进行类别编码 |
| `CabinSide` | 缺失值填充为 `Unknown`，然后进行类别编码 |
| `Age` | 使用训练集年龄中位数填补 |
| `CabinNumber` | 使用训练集舱位编号中位数填补 |
| 消费数值字段 | 基线模型中使用训练集对应字段的中位数填补 |

## 逻辑回归基线模型

### 模型设置

| 项目 | 设置 |
|---|---|
| 模型 | `LogisticRegression` |
| 验证方法 | 训练集随机划分 80%/20% |
| 随机种子 | `42` |
| 类别平衡 | 使用 `stratify=y` 保持标签比例 |
| 数值缺失值 | 使用中位数填补 |
| 类别缺失值 | 填充为 `Unknown` |
| 类别编码 | `OneHotEncoder` |
| 数值缩放 | `StandardScaler` |

### 使用的特征

| 特征类型 | 特征 |
|---|---|
| 原始类别特征 | `HomePlanet`、`CryoSleep`、`Destination`、`VIP` |
| 原始数值特征 | `Age`、`RoomService`、`FoodCourt`、`ShoppingMall`、`Spa`、`VRDeck` |
| 乘客小组特征 | `GroupMemberNumber`、`GroupSize`、`IsAlone` |
| 舱位特征 | `CabinDeck`、`CabinNumber`、`CabinSide` |
| 消费构造特征 | `TotalSpend`、`NoSpend`、`ServicesUsed` |

### 本地验证结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.79068 |
| False Precision | 0.7918 |
| False Recall | 0.7845 |
| False F1-score | 0.7881 |
| True Precision | 0.7896 |
| True Recall | 0.7968 |
| True F1-score | 0.7932 |

### 混淆矩阵

|  | 预测 False | 预测 True |
|---|---:|---:|
| 实际 False | 677 | 186 |
| 实际 True | 178 | 698 |

### 测试集预测分布

| 预测结果 | 人数 | 比例 |
|---|---:|---:|
| `True` | 2264 | 52.93% |
| `False` | 2013 | 47.07% |


## V2：CatBoost 分组交叉验证模型

### 模型改进

- 使用 `CatBoostClassifier` 替代逻辑回归；
- 使用五折 `StratifiedGroupKFold`；
- 按 `GroupId` 分组，避免同一旅行小组被分到训练折和验证折；
- 增加姓氏与家庭人数特征；
- 增加舱位区域和组合类别特征；
- 增加消费总额、消费类别数量和消费交互特征；
- 使用五个折模型对测试集概率取平均；
- 根据 OOF 预测搜索分类阈值。

### 本地验证结果

| 指标 | 结果 |
|---|---:|
| 五折平均准确率 | 0.81559 |
| 最佳阈值 OOF 准确率 | 0.81755 |
| 最佳分类阈值 | 0.480 |
| 逻辑回归基线准确率 | 0.79068 |
| 相对基线提升 | 约 2.49 个百分点 |


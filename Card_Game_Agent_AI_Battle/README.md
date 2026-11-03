## 项目介绍

宝可梦对战模拟对战界面：

![对战回放](output/可视化/对战回放.png)

## data 给的 deck 文件

为卡组文件，这里我选用盖欧卡超级暴雪王卡组作为展示例子

![超级雪笠怪](对局回放/卡组/超级雪笠怪.png)

## data 给的 main.py 内容

main.py 是官方给的最基础的随机决策AI

他的代码的逻辑为：
1. 第一次调用时返回60张牌
2. 正常对战时随机选择合法动作

但仅仅是随机选择，需要进行优化

## 构建这样一个ai对战的代码思路

main.py 文件主要的思路流程：

#### 1、主体工作流程

读取当前局面 → 找到所有合法动作 → 给每个动作打分 → 选择分数最高的动作

#### 2、卡组介绍（以超级暴雪王卡组为例）

主要卡组构成：雪笠怪、超级暴雪王 ex、盖欧卡和水能量体系

#### 3、决策连

当前是否需要提交卡组？
- 是：读取 deck.csv（卡组文件）

是否为正常回合主操作？
- 是：choose_main_action()
  - score_main_action()
  - score_attack()
  - score_energy_attachment()
- 否：choose_other_selection()
  - 选择初始宝可梦
  - 选择新的出战宝可梦
  - 处理搜索卡
  - 处理效果目标

sanitize_action()：检查最终返回值是否合法

##### 1）choose_main_action() 正常回合主操作

例如：
- 打出宝可梦；
- 使用训练家卡；
- 贴能量；
- 进化；
- 撤退；
- 攻击；
- 结束回合。

##### 2）choose_other_selection() 卡牌效果产生的二次选择

例如：
- 从牌库中选择一张卡；
- 给某只宝可梦贴能量；
- 撤退后选择新的出战宝可梦；
- 宝可梦被击倒后选择新的出战宝可梦；
- 开局选择基础宝可梦。

#### 4、动作评分系统

把不同类型的动作转换成可以比较的统一分数

例如当前合法动作有：
- 进化超级暴雪王 ex	10000
- 给盖欧卡贴水能量	7600
- 攻击造成200伤害	6950
- 结束回合	0

程序会先选择：进化超级暴雪王 ex	

引擎执行后会再次调用 agent()

再根据新得分选择得分最高的动作执行

#### 5、贴能量评分的的设计：score_energy_attachment()

主要考虑三个信息：
- target      # 贴给哪只宝可梦
- target_area # 出战区还是备战区
- energy_count

##### 1）优先防止出战宝可梦零能量卡场

if target_area == AREA_ACTIVE and energy_count == 0:

这意味着当出战宝可梦一颗能量都没有时，第一颗能量会获得很高优先级。

##### 2）超级暴雪王 ex 优先充到三能量

![超级雪笠怪](对局回放/卡组/超级雪笠怪.png)

```commandline
if card_id == MEGA_ABOMASNOW_EX:
    if energy_count < 3:
        return 8500 + active_bonus - energy_count * 120
```

但是注意：

第 0 颗到第 3 颗之前都具有较高优先级，但每多一颗能量，分数会略微降低：

- 0能量：最高
- 1能量：稍低
- 2能量：再稍低

这样可以避免机械地一直给同一只宝可梦贴能量，而完全忽视其他宝可梦。

##### 3）盖欧卡即使三能量以后仍有一定价值

![盖欧卡](对局回放/卡组/盖欧卡.png)

```commandline
if card_id == KYOGRE:
    if energy_count < 3:
        return 7600 + active_bonus - energy_count * 120

    return 6100 + active_bonus
```

原因：Extra energy helps recover from Swirling Waves' discard cost.

即盖欧卡的攻击可能需要弃能量，因此多余能量不是完全浪费，而是后续攻击资源。

#### 6、如何设定攻击评分

基本评分公式是：

```commandline
score = (
    4300
    + estimated_damage * 10
    + reliability_bonus
    - resource_penalty
)
```

即：攻击分数 = 基础攻击价值+ 预计伤害价值+ 稳定性价值- 资源消耗风险

##### 1)以盖欧卡的技能Swirling Waves为例

![盖欧卡](对局回放/卡组/盖欧卡.png)

```commandline
estimated_damage = 130
```

但它有资源惩罚：

```commandline
resource_penalty = 500 if energy_count <= 3 else 250
```

也就是说：
- 只有三颗能量时发动，资源压力较大；
- 有额外能量时发动，损失相对较小。

##### 2)以雪笠怪形态的攻击为例

![雪笠怪](对局回放/卡组/雪笠怪.png)

ATTACK_SNOVER_ICY_SNOW 得分估算为：30

并带有少量稳定性加分：reliability_bonus = 100

它们主要是过渡攻击，评分不会高于完成进化和关键充能。

##### 3)以超级暴雪王 ex 的 Hammer-lanche技能为例

![超级雪笠怪](对局回放/卡组/超级雪笠怪.png)

这个攻击不是固定伤害，而是和牌库中水能量比例相关。

通过estimate_hammer_lanche_damage()估计伤害

假设卡组原始水能量总量为：35

先计算已知区域中的水能量：

```commandline
known_water_outside_deck = count_visible_water(player)
```

再估算未知区域中还有多少水能量：

```commandline
estimated_water_in_unknown_zones = max(
    0,
    35 - known_water_outside_deck,
)
```

最后估算翻六张时命中水能量的数量：

```commandline
expected_hits = 6 * estimated_water_in_deck / deck_count
```

每命中一张按 100 伤害计算：

```commandline
round(expected_hits * 100)
```

##### Hammer-lanche 的牌库风险

攻击会消耗牌库，因此代码根据剩余牌库数量施加惩罚：

```commandline
if deck_count <= 12:
    resource_penalty = 5000
elif deck_count <= 18:
    resource_penalty = 1800
else:
    resource_penalty = 350
```

- 牌库充足时，可以放心使用；
- 牌库较少时，谨慎使用；
- 牌库只剩 12 张以内时，强烈避免使用。

##### 4）击倒奖励

如果预计伤害足以击倒对手：

固定攻击获得：score += 6000

而随机性更大的 Hammer-lanche 只增加：score += 2500

#### 7、主回合不同动作如何评分

##### 1）进化

```commandline
if action_type == ACTION_EVOLVE:
    if card_id == MEGA_ABOMASNOW_EX:
        return 10000
    return 9000
```

超级暴雪王 ex 的进化几乎是最高优先级。

##### 2)搜索卡

如果场上没有等待进化的雪笠怪,使用搜索卡会得到负分

这样可以避免 AI 为了“能用卡”而无意义地浪费训练家卡。

##### 3）Waitress

![Waitress](对局回放/卡组/Waitress.png)

```commandline
if card_id == WAITRESS:
    return 8350 if any_pokemon_needs_energy(player) else -500
```

只有在场上有需要充能的宝可梦时才使用。

否则留在手中，不浪费效果。

##### 4)莉莉艾的决心

![莉莉艾](对局回放/卡组/莉莉艾.png)

这部分规则比较复杂，因为它同时考虑：

- 剩余奖赏卡数量；
- 手牌数量；
- 是否没有后备宝可梦；
- 手牌中是否已有基础宝可梦；
- 是否有进化渠道；
- 牌库是否快耗尽。

例如：牌库太少时避免继续大量抽牌。

```commandline
if deck_count <= 8:
    return -1200
```

如果没有后备宝可梦，同时手中也没有基础宝可梦：

```commandline
if bench_count == 0 and not basic_in_hand:
    return 9950 if prize_count == 6 else 9700
```

#### 8、交换与撤退

判断是否应该换出当前出战宝可梦

##### 1）情况一：超级暴雪王 ex 被对方效果封锁

如果超级暴雪王 ex 无法对当前对手造成伤害，并且备战区有盖欧卡或雪笠怪，使用交换卡

##### 2）情况二：当前出战宝可梦血量过低

```commandline
active_hp * 4 <= active_max_hp
```

同时备战区存在较成熟的攻击手时，考虑换人。

#### 9)攻击前的准备

如果还有得分达到 6000 的有效准备动作，就先不攻击。

准备动作包括：

- 进化；
- 贴能量；
- 放置后备宝可梦；
- 使用有效搜索；
- 使用有效抽牌；
- 附加工具；
- 必要撤退。

#### 10）二次选择的单独处理

choose_other_selection() 相当于一个针对不同游戏提示的状态机。

##### 1）开局放置额外基础宝可梦

```commandline
preferred = [SNOVER, KYOGRE]
```

优先雪笠怪，其次盖欧卡

并且当备战区已经有两只宝可梦时，不会把基础宝可梦全部铺满，避免手牌和备战位浪费。

##### 2）撤退或交换后选择新出战宝可梦

- 能量更多的宝可梦更适合上场；
- 当前血量更高的宝可梦更适合上场。

#### 11）区分不同区域中的卡牌 ID

Index 与 位置 相对应：
- 1	牌库
- 2	手牌
- 3	弃牌区
- 4	出战区
- 5	备战区
- 12	临时查看区域

通过 index（不是卡牌 ID）来区分某个区域里的位置
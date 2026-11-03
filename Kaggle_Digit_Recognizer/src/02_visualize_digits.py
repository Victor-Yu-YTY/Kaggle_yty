from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# 1. 设置文件路径
# =========================

# 当前文件位于 src 文件夹中
# parent 是 src，parent.parent 是 DigitRecognizer 项目根目录
project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.csv"
output_dir = project_root / "outputs"

# 如果 outputs 文件夹不存在，就自动创建
output_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取训练数据
# =========================

train_data = pd.read_csv(train_path)

print("训练集大小：", train_data.shape)


# =========================
# 3. 分离标签和像素
# =========================

# label 是每张图片对应的正确数字
labels = train_data["label"]

# 删除 label 后，剩下的784列都是像素
pixels = train_data.drop(columns=["label"])

print("标签数量：", labels.shape)
print("像素数据大小：", pixels.shape)

print("像素最小值：", pixels.min().min())
print("像素最大值：", pixels.max().max())


# =========================
# 4. 显示前10张图片
# =========================

fig, axes = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(10, 5)
)

# 把二维的axes转换成一维，方便循环
axes = axes.flatten()

for index in range(10):
    # 取出第index张图片的784个像素
    image_pixels = pixels.iloc[index].to_numpy()

    # 将784个像素重新变成28×28的图片
    image = image_pixels.reshape(28, 28)

    # 显示图片
    axes[index].imshow(image, cmap="gray")

    # 显示正确标签
    axes[index].set_title(f"Label: {labels.iloc[index]}")

    # 隐藏横轴和纵轴
    axes[index].axis("off")


plt.tight_layout()


# =========================
# 5. 保存图片
# =========================

save_path = output_dir / "sample_digits.png"

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

print(f"图片已经保存到：{save_path}")


# =========================
# 6. 显示图片窗口
# =========================

plt.show()
from pathlib import Path
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


# ==================================================
# 1. 固定随机种子，使结果尽量可以复现
# ==================================================

RANDOM_SEED = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==================================================
# 2. 设置项目路径
# ==================================================

project_root = Path(__file__).resolve().parent.parent

train_path = project_root / "data" / "train.csv"
model_dir = project_root / "models"
output_dir = project_root / "outputs"

model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

best_model_path = model_dir / "cnn_best_validation.keras"
history_path = output_dir / "cnn_training_history.csv"
curve_path = output_dir / "cnn_training_curves.png"
validation_path = output_dir / "cnn_validation_predictions.csv"


# ==================================================
# 3. 检查训练文件
# ==================================================

if not train_path.exists():
    raise FileNotFoundError(f"没有找到训练数据：{train_path}")


# ==================================================
# 4. 显示TensorFlow运行设备
# ==================================================

print("=" * 65)
print("TensorFlow版本：", tf.__version__)

gpu_devices = tf.config.list_physical_devices("GPU")

if gpu_devices:
    print("检测到GPU：", gpu_devices)
else:
    print("没有检测到GPU，将使用CPU训练")


# ==================================================
# 5. 读取训练数据
# ==================================================

print("\n正在读取训练数据……")

train_data = pd.read_csv(train_path)

print("原始训练集大小：", train_data.shape)


# ==================================================
# 6. 分离图片和标签
# ==================================================

y = train_data["label"].to_numpy(dtype=np.int64)

X = train_data.drop(
    columns=["label"]
).to_numpy(dtype=np.float32)

print("原始特征大小：", X.shape)
print("标签大小：", y.shape)


# ==================================================
# 7. 像素归一化
# ==================================================

X /= 255.0

print("归一化后的像素范围：", X.min(), "～", X.max())


# ==================================================
# 8. 将784个像素恢复为28×28图片
# ==================================================

X = X.reshape(-1, 28, 28, 1)

print("恢复图片后的数据大小：", X.shape)

# 删除原始DataFrame，节省内存
del train_data


# ==================================================
# 9. 划分训练集和验证集
# ==================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\n训练部分大小：", X_train.shape)
print("验证部分大小：", X_valid.shape)
print("训练标签大小：", y_train.shape)
print("验证标签大小：", y_valid.shape)


# ==================================================
# 10. 创建轻微数据增强
# ==================================================

data_augmentation = keras.Sequential(
    [
        # 轻微旋转，避免旋转幅度太大导致6和9混淆
        layers.RandomRotation(
            factor=0.03,
            fill_mode="constant"
        ),

        # 水平和竖直方向轻微平移
        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08,
            fill_mode="constant"
        ),

        # 轻微放大或缩小
        layers.RandomZoom(
            height_factor=(-0.08, 0.08),
            width_factor=(-0.08, 0.08),
            fill_mode="constant"
        ),
    ],
    name="data_augmentation"
)


# ==================================================
# 11. 创建CNN模型
# ==================================================

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),

        # 只在模型训练时进行数据增强
        data_augmentation,

        # 第一组卷积层
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.20),

        # 第二组卷积层
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # 第三组卷积层
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        layers.BatchNormalization(),

        # 将二维特征图压缩成一维特征
        layers.GlobalAveragePooling2D(),

        # 全连接分类层
        layers.Dense(
            units=128,
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.35),

        # 最终输出0～9十个类别的概率
        layers.Dense(
            units=10,
            activation="softmax"
        )
    ],
    name="digit_recognizer_cnn"
)


# ==================================================
# 12. 编译模型
# ==================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN模型结构：")
model.summary()


# ==================================================
# 13. 设置训练回调
# ==================================================

callbacks = [
    # 保存验证准确率最高的模型
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # 验证损失不再改善时，降低学习率
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    ),

    # 长时间不再改善时提前结束训练
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


# ==================================================
# 14. 训练CNN
# ==================================================

print("\n开始训练CNN模型……")
print("CPU训练时间可能较长，请等待每一轮epoch完成。")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=30,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

print(f"\n训练完成，耗时：{training_time:.2f}秒")
print(f"训练完成，耗时：{training_time / 60:.2f}分钟")


# ==================================================
# 15. 加载验证表现最好的模型
# ==================================================

print("\n正在加载验证准确率最高的CNN模型……")

best_model = keras.models.load_model(best_model_path)

print("最佳CNN模型加载成功")


# ==================================================
# 16. 验证集评估
# ==================================================

valid_loss, valid_accuracy = best_model.evaluate(
    X_valid,
    y_valid,
    batch_size=256,
    verbose=1
)

print("\n" + "=" * 65)
print(f"CNN验证集损失：{valid_loss:.5f}")
print(f"CNN验证集准确率：{valid_accuracy:.5f}")
print(f"CNN验证集正确率：{valid_accuracy * 100:.2f}%")


# ==================================================
# 17. 生成验证集预测结果
# ==================================================

valid_probabilities = best_model.predict(
    X_valid,
    batch_size=256,
    verbose=1
)

valid_predictions = np.argmax(
    valid_probabilities,
    axis=1
)

calculated_accuracy = accuracy_score(
    y_valid,
    valid_predictions
)

print(f"再次计算的准确率：{calculated_accuracy:.5f}")


# ==================================================
# 18. 输出各数字分类报告
# ==================================================

print("\n各数字分类结果：")

print(
    classification_report(
        y_valid,
        valid_predictions,
        digits=4
    )
)


# ==================================================
# 19. 保存验证集预测结果
# ==================================================

validation_result = pd.DataFrame(
    {
        "TrueLabel": y_valid,
        "PredictedLabel": valid_predictions,
        "Confidence": valid_probabilities.max(axis=1)
    }
)

validation_result["Correct"] = (
    validation_result["TrueLabel"]
    == validation_result["PredictedLabel"]
)

validation_result.to_csv(
    validation_path,
    index=False
)

print("验证集预测结果已保存到：")
print(validation_path)


# ==================================================
# 20. 保存训练历史
# ==================================================

history_data = pd.DataFrame(history.history)

history_data.to_csv(
    history_path,
    index=False
)

print("\n训练历史已保存到：")
print(history_path)


# ==================================================
# 21. 绘制训练曲线
# ==================================================

epochs_completed = range(
    1,
    len(history.history["loss"]) + 1
)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs_completed,
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_completed,
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    curve_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("训练曲线已保存到：")
print(curve_path)

print("\nCNN本地验证完成")
print("=" * 65)
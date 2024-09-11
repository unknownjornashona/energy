import tensorflow as tf
import numpy as np
import pandas as pd
import logging

# 设置日志
logging.basicConfig(filename='radar_gan_log.txt', level=logging.INFO)

class RadarDataGenerator:
    def __init__(self, log_file='radar_gan_log.txt'):
        self.generated_data = []
        self.log_file = log_file

    def add_data(self, data_point):
        self.generated_data.append(data_point)

    def save_to_csv(self, filename='generated_radar_data.csv'):
        if not self.generated_data:
            logging.error("没有数据可保存，请生成数据。")
            raise Exception("没有数据可保存。")
        
        # 数据框架
        radar_df = pd.DataFrame(self.generated_data, columns=['TargetType', 'CoordinateX', 'CoordinateY', 'Speed'])
        radar_df.to_csv(filename, index=False)
        logging.info(f"生成的雷达数据已保存至 '{filename}'")

    def log_epoch_completion(self, epoch):
        logging.info(f"Epoch {epoch} completed.")

# 生成器模型
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4)  # 假设有四个特征：目标类别、坐标 (x, y)、速度
    ])
    return model

# 判别器模型
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 二分类输出
    ])
    return model

def train_gan(epochs=10000, batch_size=32):
    radar_data_gen = RadarDataGenerator()

    generator = build_generator()
    discriminator = build_discriminator()

    # 编译判别器
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 构建 GAN 模型
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(100,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan_model = tf.keras.Model(gan_input, gan_output)
    gan_model.compile(optimizer='adam', loss='binary_crossentropy')

    for epoch in range(epochs):
        # 生成假数据
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        generated_data = generator.predict(noise)

        # 真实数据（这里随机生成，实际应当是雷达数据）
        real_data = np.random.rand(batch_size, 4) * np.array([3, 100, 100, 30])  # 假定特征范围
        
        # 合并数据并标记
        combined_data = np.concatenate([real_data, generated_data])
        labels = np.array([1] * batch_size + [0] * batch_size)

        # 训练判别器
        d_loss = discriminator.train_on_batch(combined_data, labels)

        # 训练生成器
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        labels_gan = np.array([1] * batch_size)  # 目标是让生成器生成真实样本
        g_loss = gan_model.train_on_batch(noise, labels_gan)

        # 保存生成的数据
        if epoch % 1000 == 0:
            radar_data_gen.log_epoch_completion(epoch)
            for g_data in generated_data:
                radar_data_gen.add_data(g_data)

    # 保存生成的雷达数据
    try:
        radar_data_gen.save_to_csv()
    except Exception as e:
        logging.error(f"发生异常：{str(e)}")
        print("数据保存过程中出现异常，请查看日志了解详细信息。")

if __name__ == "__main__":
    try:
        train_gan()
    except Exception as e:
        logging.error(f"训练过程中发生异常：{str(e)}")
        print("训练过程中出现异常，请查看日志了解详细信息。")

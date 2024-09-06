import logging
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 配置日志记录
logging.basicConfig(
    filename='drake_equation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DrakeEquation:
    def __init__(self, R_star, f_p, n_e, f_l, f_i, f_c, L):
        self.R_star = R_star  # 每年形成的新恒星的数量
        self.f_p = f_p        # 拥有行星系统的恒星比例
        self.n_e = n_e        # 每个星系中适合生命发展的行星平均数量
        self.f_l = f_l        # 能够支持生命的行星比例
        self.f_i = f_i        # 发展出智能生命的行星比例
        self.f_c = f_c        # 能发出信号的文明比例
        self.L = L            # 文明发出信号的年限

    def calculate_civilizations(self):
        try:
            N = self.R_star * self.f_p * self.n_e * self.f_l * self.f_i * self.f_c * self.L
            logging.info(f'计算结果 N: {N}')
            return N
        except Exception as e:
            logging.error(f'计算过程中发生异常: {e}')
            raise

    def save_to_csv(self, file_name):
        data = {
            'R_star': self.R_star,
            'f_p': self.f_p,
            'n_e': self.n_e,
            'f_l': self.f_l,
            'f_i': self.f_i,
            'f_c': self.f_c,
            'L': self.L
        }
        df = pd.DataFrame([data])
        df.to_csv(file_name, index=False)

    def plot_results(self, filename):
        params = [self.R_star, self.f_p, self.n_e, self.f_l, self.f_i, self.f_c, self.L]
        labels = ['R_star', 'f_p', 'n_e', 'f_l', 'f_i', 'f_c', 'L']

        # 散点图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(labels, params, color='blue')
        plt.title('参数散点图')
        plt.xlabel('参数')
        plt.ylabel('值')
        plt.grid()

        # 曲线图
        plt.subplot(1, 2, 2)
        plt.plot(labels, params, marker='o', linestyle='-', color='orange')
        plt.title('参数曲线图')
        plt.xlabel('参数')
        plt.ylabel('值')
        plt.grid()

        # 保存图表为 PNG 文件
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# 定义简单的GAN模型
class SimpleGAN:
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(100,)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(1,)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def train(self, epochs=10000):
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (32, 100))
            generated_data = self.generator.predict(noise)
            real_data = np.random.rand(32, 1)

            # 训练判别器
            self.discriminator.train_on_batch(real_data, np.ones((32, 1)))
            self.discriminator.train_on_batch(generated_data, np.zeros((32, 1)))

            # 训练生成器
            noise = np.random.normal(0, 1, (32, 100))
            self.discriminator.trainable = False
            self.generator.train_on_batch(noise, np.ones((32, 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='德雷克方程计算与GAN模型')
    parser.add_argument('--R_star', type=float, default=10, help='每年形成的新恒星的数量 (建议增大)')
    parser.add_argument('--f_p', type=float, default=0.7, help='拥有行星系统的恒星比例 (建议增大)')
    parser.add_argument('--n_e', type=float, default=5, help='适合生命发展的行星平均数量 (建议增大)')
    parser.add_argument('--f_l', type=float, default=0.5, help='能够支持生命的行星比例 (建议增大)')
    parser.add_argument('--f_i', type=float, default=0.1, help='发展出智能生命的行星比例 (建议增大)')
    parser.add_argument('--f_c', type=float, default=0.3, help='能发出信号的文明比例 (建议增大)')
    parser.add_argument('--L', type=int, default=5000, help='文明发出信号的年限 (建议增大)')
    parser.add_argument('--csv', type=str, default='drake_results.csv', help='输出CSV文件名')
    parser.add_argument('--plot', type=str, default='drake_plot.png', help='输出PNG图表文件名')

    args = parser.parse_args()

    drake_eq = DrakeEquation(args.R_star, args.f_p, args.n_e, args.f_l, args.f_i, args.f_c, args.L)

    try:
        result = drake_eq.calculate_civilizations()
        print(f"可通信外星文明的数量 N: {result}")
        drake_eq.save_to_csv(args.csv)
        drake_eq.plot_results(args.plot)
    except Exception as e:
        print(f"计算发生错误: {e}")

    # 初始化并训练GAN
    gan = SimpleGAN()
    gan.train(epochs=1000)  # 训练GAN模型

#python script.py --R_star 10 --f_p 0.7 --n_e 5 --f_l 0.5 --f_i 0.1 --f_c 0.3 --L 5000 --csv drake_results.csv --plot drake_plot.png

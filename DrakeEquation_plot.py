import logging
import argparse
import pandas as pd
import numpy as np
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
        # 创建参数的网格
        R_star_range = np.linspace(1, 20, 100)
        f_p_range = np.linspace(0.1, 1.0, 100)
        N_values = np.zeros((len(R_star_range), len(f_p_range)))  # 初始化 N_values 数组

        for i, R_star in enumerate(R_star_range):
            for j, f_p in enumerate(f_p_range):
                N_values[i, j] = (R_star * f_p * self.n_e * self.f_l * self.f_i * self.f_c * self.L)

        # 生成彩色云图
        plt.figure(figsize=(12, 6))
        plt.contourf(R_star_range, f_p_range, N_values.T, cmap='viridis')  # 借助色彩渐变生成云图
        plt.colorbar(label='文明数量 N')
        plt.title('R_star 与 f_p 对 N 的影响')
        plt.xlabel('R_star (每年形成的新恒星数量)')
        plt.ylabel('f_p (拥有行星系统的恒星比例)')
        plt.grid()

        # 保存图表为 PNG 文件
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='德雷克方程计算与可视化')
    parser.add_argument('--R_star', type=float, default=10, help='每年形成的新恒星的数量')
    parser.add_argument('--f_p', type=float, default=0.7, help='拥有行星系统的恒星比例')
    parser.add_argument('--n_e', type=float, default=5, help='适合生命发展的行星平均数量')
    parser.add_argument('--f_l', type=float, default=0.5, help='能够支持生命的行星比例')
    parser.add_argument('--f_i', type=float, default=0.1, help='发展出智能生命的行星比例')
    parser.add_argument('--f_c', type=float, default=0.3, help='能发出信号的文明比例')
    parser.add_argument('--L', type=int, default=5000, help='文明发出信号的年限')
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
        
#python script.py --R_star 10 --f_p 0.7 --n_e 5 --f_l 0.5 --f_i 0.1 --f_c 0.3 --L 5000 --csv drake_results.csv --plot drake_plot.png


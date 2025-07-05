#!/usr/bin/env python3
"""
LDPC数据生成器 - SJ.py
使用LDPC_coder生成的H矩阵，为神经网络译码器生成训练数据
主要使用全零码字策略，确保数据有效性
"""

import numpy as np
import torch
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LDPCDataGenerator:
    """
    LDPC数据生成器
    为神经网络译码研究生成训练数据
    """

    def __init__(self, H_path=None, H_matrix=None, device='cpu'):
        """
        初始化数据生成器
        H_path: H矩阵文件路径
        H_matrix: 直接提供的H矩阵
        """
        self.device = device

        # 加载H矩阵
        if H_matrix is not None:
            self.H = H_matrix
        elif H_path is not None:
            self.H = self.load_h_matrix(H_path)
        else:
            raise ValueError("必须提供H_path或H_matrix")

        # 转换为numpy和torch格式
        self.H_np = self.H.astype(int)
        self.H_torch = torch.tensor(self.H, dtype=torch.float32, device=device)

        # LDPC码参数
        self.m, self.n = self.H.shape  # m: 校验节点数, n: 变量节点数（码长）
        self.k = self.n - self.m  # 信息位长度
        self.rate = self.k / self.n  # 码率

        print(f"LDPC码参数:")
        print(f"  码长 n = {self.n}")
        print(f"  信息位 k = {self.k}")
        print(f"  校验位 m = {self.m}")
        print(f"  码率 R = {self.rate:.3f}")

        # 验证全零码字是否有效
        self._verify_all_zero_codeword()

    def load_h_matrix(self, h_path):
        """加载H矩阵"""
        if h_path.endswith('.npy'):
            H = np.load(h_path)
        elif h_path.endswith('.npz'):
            from scipy.sparse import load_npz
            H_sparse = load_npz(h_path)
            H = H_sparse.toarray()
        else:
            raise ValueError(f"不支持的文件格式: {h_path}")

        print(f"成功加载H矩阵: shape = {H.shape}")
        return H

    def _verify_all_zero_codeword(self):
        """验证全零码字是否满足校验方程"""
        zero_codeword = np.zeros(self.n, dtype=int)
        syndrome = (self.H_np @ zero_codeword) % 2

        if not np.all(syndrome == 0):
            raise ValueError("H矩阵错误：全零码字不满足校验方程 H*c = 0")
        print("✓ H矩阵验证通过：全零码字有效")

    def generate_codewords(self, num_samples, all_zero_ratio=0.5):
        """
        生成码字
        all_zero_ratio: 全零码字的比例（保证有效性）
        其余为随机生成的码字（可能无效，但增加多样性）
        """
        num_zeros = int(num_samples * all_zero_ratio)
        num_random = num_samples - num_zeros

        codewords = []
        labels = []  # 标记是否为有效码字

        # 1. 生成全零码字（保证有效）
        if num_zeros > 0:
            zero_codewords = np.zeros((num_zeros, self.n), dtype=np.float32)
            codewords.append(zero_codewords)
            labels.extend([1] * num_zeros)  # 标记为有效

        # 2. 生成随机码字（增加多样性）
        if num_random > 0:
            # 生成随机信息位
            random_messages = np.random.randint(0, 2, size=(num_random, self.k))

            # 简单的编码尝试：将信息位放在前k位，校验位随机
            random_codewords = np.zeros((num_random, self.n), dtype=np.float32)
            random_codewords[:, :self.k] = random_messages
            random_codewords[:, self.k:] = np.random.randint(0, 2, size=(num_random, self.m))

            # 检查哪些是有效的
            for i in range(num_random):
                syndrome = (self.H_np @ random_codewords[i].astype(int)) % 2
                if np.all(syndrome == 0):
                    labels.append(1)  # 偶然有效
                else:
                    labels.append(0)  # 无效

            codewords.append(random_codewords)

        # 合并并打乱顺序
        codewords = np.vstack(codewords)
        labels = np.array(labels)

        # 打乱顺序
        perm = np.random.permutation(num_samples)
        codewords = codewords[perm]
        labels = labels[perm]

        return codewords, labels

    def add_awgn_noise(self, codewords, snr_db):
        """
        添加AWGN噪声
        codewords: 二进制码字 (batch_size, n)
        snr_db: 信噪比（dB）
        """
        # BPSK调制: 0 -> +1, 1 -> -1
        modulated = 1.0 - 2.0 * codewords

        # 计算噪声标准差
        # SNR = Es/N0, 其中Es是符号能量（BPSK为1），N0/2是噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (2 * snr_linear))

        # 生成高斯噪声
        noise = np.random.normal(0, noise_std, modulated.shape)

        # 添加噪声
        received = modulated + noise

        return received, noise_std

    def compute_llr(self, received, noise_std):
        """
        计算对数似然比（LLR）
        LLR = log(P(r|c=0) / P(r|c=1))
        对于AWGN信道和BPSK调制：LLR = 2*r/σ²
        """
        llr = 2 * received / (noise_std ** 2)
        return llr

    def generate_dataset(self, num_samples, snr_db, all_zero_ratio=0.5):
        """
        生成完整的数据集
        """
        # 生成码字
        codewords, valid_labels = self.generate_codewords(num_samples, all_zero_ratio)

        # 添加噪声
        received, noise_std = self.add_awgn_noise(codewords, snr_db)

        # 计算LLR
        llr = self.compute_llr(received, noise_std)

        # 统计信息
        num_valid = np.sum(valid_labels)
        print(f"生成 {num_samples} 个样本:")
        print(f"  有效码字: {num_valid} ({num_valid / num_samples * 100:.1f}%)")
        print(f"  SNR: {snr_db} dB")
        print(f"  噪声标准差: {noise_std:.4f}")

        return {
            'codewords': codewords.astype(np.float32),
            'received': received.astype(np.float32),
            'llr': llr.astype(np.float32),
            'valid_labels': valid_labels.astype(np.float32),
            'snr_db': snr_db,
            'noise_std': noise_std
        }

    def save_dataset(self, dataset, save_path):
        """保存数据集"""
        # 转换为PyTorch张量
        torch_dataset = {
            'codewords': torch.from_numpy(dataset['codewords']),
            'llr': torch.from_numpy(dataset['llr']),
            'valid_labels': torch.from_numpy(dataset['valid_labels']),
            'snr_db': dataset['snr_db'],
            'noise_std': dataset['noise_std']
        }

        torch.save(torch_dataset, save_path)
        print(f"数据已保存到: {save_path}")

    def analyze_dataset(self, dataset):
        """分析数据集统计信息"""
        codewords = dataset['codewords']
        llr = dataset['llr']
        valid_labels = dataset['valid_labels']

        print("\n数据集统计分析:")
        print(f"样本数量: {len(codewords)}")
        print(f"有效码字比例: {np.mean(valid_labels):.3f}")

        # 码字统计
        hamming_weights = np.sum(codewords, axis=1)
        print(f"\n码字汉明重量统计:")
        print(f"  平均: {np.mean(hamming_weights):.2f}")
        print(f"  最小: {np.min(hamming_weights)}")
        print(f"  最大: {np.max(hamming_weights)}")
        print(f"  全零码字数: {np.sum(hamming_weights == 0)}")

        # LLR统计
        print(f"\nLLR统计:")
        print(f"  均值: {np.mean(llr):.3f}")
        print(f"  标准差: {np.std(llr):.3f}")
        print(f"  最小值: {np.min(llr):.3f}")
        print(f"  最大值: {np.max(llr):.3f}")

        # 估计BER（硬判决）
        hard_decisions = (llr < 0).astype(float)
        bit_errors = np.abs(hard_decisions - codewords)
        ber = np.mean(bit_errors)
        print(f"\n硬判决BER估计: {ber:.4f}")

        # 只计算有效码字的BER
        if np.sum(valid_labels) > 0:
            valid_indices = valid_labels == 1
            valid_ber = np.mean(bit_errors[valid_indices])
            print(f"有效码字BER: {valid_ber:.4f}")


def main():
    """主函数 - 生成完整的LDPC数据集"""

    print("=" * 60)
    print("LDPC数据生成器 - 神经网络译码训练数据")
    print("=" * 60)

    # 配置参数
    H_PATH = 'ldpc_data/small_regular_64_32_3_H.npy'  # H矩阵路径
    SAVE_DIR = 'ldpc_data'  # 保存目录

    # 数据集参数
    SNR_LIST = [0, 1, 2, 3, 4, 5]  # SNR列表（dB）
    NUM_TRAIN = 50000  # 训练样本数
    NUM_VAL = 10000  # 验证样本数
    NUM_TEST = 10000  # 测试样本数

    # 检查H矩阵是否存在
    if not os.path.exists(H_PATH):
        print(f"\n错误: 找不到H矩阵文件 {H_PATH}")
        print("请先运行 LDPC_coder.py 生成H矩阵")
        print("\n步骤:")
        print("1. 运行: python LDPC_coder.py")
        print("2. 选择: 1 (小规模规则LDPC码)")
        print("3. 保存生成的矩阵")
        return

    # 创建数据生成器
    generator = LDPCDataGenerator(H_path=H_PATH)

    # 为每个SNR生成数据
    for snr_db in SNR_LIST:
        print(f"\n{'=' * 50}")
        print(f"处理 SNR = {snr_db} dB")
        print(f"{'=' * 50}")

        # 创建SNR子目录
        snr_dir = os.path.join(SAVE_DIR, f'snr_{snr_db}dB')
        os.makedirs(snr_dir, exist_ok=True)

        # 生成训练集（50%全零码字，50%随机码字）
        print("\n生成训练集...")
        train_dataset = generator.generate_dataset(
            num_samples=NUM_TRAIN,
            snr_db=snr_db,
            all_zero_ratio=0.5
        )
        generator.save_dataset(train_dataset, os.path.join(snr_dir, 'train.pt'))

        # 生成验证集（70%全零码字，30%随机码字）
        print("\n生成验证集...")
        val_dataset = generator.generate_dataset(
            num_samples=NUM_VAL,
            snr_db=snr_db,
            all_zero_ratio=0.7
        )
        generator.save_dataset(val_dataset, os.path.join(snr_dir, 'val.pt'))

        # 生成测试集（100%全零码字，保证准确性）
        print("\n生成测试集...")
        test_dataset = generator.generate_dataset(
            num_samples=NUM_TEST,
            snr_db=snr_db,
            all_zero_ratio=1.0
        )
        generator.save_dataset(test_dataset, os.path.join(snr_dir, 'test.pt'))

        # 分析测试集
        generator.analyze_dataset(test_dataset)

        # 保存元信息
        meta_info = {
            'snr_db': snr_db,
            'num_train': NUM_TRAIN,
            'num_val': NUM_VAL,
            'num_test': NUM_TEST,
            'code_length': generator.n,
            'info_bits': generator.k,
            'code_rate': generator.rate,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(snr_dir, 'meta.json'), 'w') as f:
            json.dump(meta_info, f, indent=2)

    # 保存H矩阵到每个SNR目录（方便使用）
    print("\n复制H矩阵到各个目录...")
    H = generator.H
    for snr_db in SNR_LIST:
        snr_dir = os.path.join(SAVE_DIR, f'snr_{snr_db}dB')
        np.save(os.path.join(snr_dir, 'H.npy'), H)

    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)

    # 打印使用说明
    print("\n使用方法示例:")
    print("```python")
    print("import torch")
    print("import numpy as np")
    print("")
    print("# 加载数据")
    print("snr = 3  # 选择SNR")
    print("train_data = torch.load(f'ldpc_data/snr_{snr}dB/train.pt')")
    print("val_data = torch.load(f'ldpc_data/snr_{snr}dB/val.pt')")
    print("")
    print("# 获取数据")
    print("train_llr = train_data['llr']  # LLR值")
    print("train_codewords = train_data['codewords']  # 真实码字")
    print("train_valid = train_data['valid_labels']  # 是否为有效码字")
    print("")
    print("# 加载H矩阵")
    print("H = np.load(f'ldpc_data/snr_{snr}dB/H.npy')")
    print("```")

    # 可视化一个样本
    print("\n生成可视化图表...")
    visualize_sample_data(generator, SNR_LIST[len(SNR_LIST) // 2])


def visualize_sample_data(generator, snr_db=3):
    """可视化样本数据"""
    # 生成少量样本用于可视化
    dataset = generator.generate_dataset(100, snr_db, all_zero_ratio=0.5)

    plt.figure(figsize=(12, 8))

    # 子图1: LLR分布
    plt.subplot(2, 2, 1)
    plt.hist(dataset['llr'].flatten(), bins=50, alpha=0.7, density=True)
    plt.title(f'LLR分布 (SNR={snr_db}dB)')
    plt.xlabel('LLR值')
    plt.ylabel('概率密度')

    # 子图2: 码字汉明重量分布
    plt.subplot(2, 2, 2)
    hamming_weights = np.sum(dataset['codewords'], axis=1)
    plt.hist(hamming_weights, bins=np.arange(0, generator.n + 2) - 0.5)
    plt.title('码字汉明重量分布')
    plt.xlabel('汉明重量')
    plt.ylabel('数量')

    # 子图3: 有效/无效码字比例
    plt.subplot(2, 2, 3)
    valid_counts = [np.sum(dataset['valid_labels']), len(dataset['valid_labels']) - np.sum(dataset['valid_labels'])]
    plt.pie(valid_counts, labels=['有效码字', '无效码字'], autopct='%1.1f%%')
    plt.title('码字有效性分布')

    # 子图4: BER vs 位置
    plt.subplot(2, 2, 4)
    hard_decisions = (dataset['llr'] < 0).astype(float)
    bit_errors = np.abs(hard_decisions - dataset['codewords'])
    position_ber = np.mean(bit_errors, axis=0)
    plt.plot(position_ber, 'b-', linewidth=2)
    plt.title('各位置的比特错误率')
    plt.xlabel('比特位置')
    plt.ylabel('BER')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ldpc_data_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化图表已保存到: ldpc_data_visualization.png")
    plt.close()


if __name__ == "__main__":
    main()
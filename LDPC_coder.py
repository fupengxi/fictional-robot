#!/usr/bin/env python3
"""
LDPC码生成器 - 用于神经网络译码研究 (交互式版本)
生成LDPC校验矩阵和相关数据集
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import copy
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 修复Python 2到Python 3的兼容性问题
def find_smallest(array):
    if len(array) == 1:
        return 0
    elif len(array) == 2:
        if array[0] <= array[1]:
            return 0
        else:
            return 1
    else:
        arrayA = array[:len(array) // 2]  # Python 3 整除
        arrayB = array[len(array) // 2:]
        smallA = find_smallest(arrayA)
        smallB = find_smallest(arrayB)
        if arrayA[smallA] <= arrayB[smallB]:
            return smallA
        else:
            return len(arrayA) + smallB


class PEG:
    """Progressive Edge Growth算法的Python 3实现"""

    def __init__(self, nvar, nchk, degree_sequence):
        self.degree_sequence = degree_sequence
        self.nvar = nvar
        self.nchk = nchk
        self.H = np.zeros((nchk, nvar), dtype=np.int32)
        self.sym_degrees = np.zeros(nvar, dtype=np.int32)
        self.chk_degrees = np.zeros(nchk, dtype=np.int32)
        self.I_edge_chk2var = []
        self.I_edge_var2chk = []

        for chk in range(nchk):
            self.I_edge_chk2var.append([0] * nvar)
        for var in range(nvar):
            self.I_edge_var2chk.append([0] * nchk)

    def grow_edge(self, var, chk):
        self.I_edge_chk2var[chk][var] = 1
        self.I_edge_var2chk[var][chk] = 1
        self.H[chk, var] = 1
        self.sym_degrees[var] += 1
        self.chk_degrees[chk] += 1

    def bfs(self, var):
        var_list = np.zeros(self.nvar, dtype=np.int32)
        var_list[var] = 1
        cur_chk_list = [0] * self.nchk
        new_chk_list = [0] * self.nchk
        chk_Q = []
        var_Q = [var]

        while True:
            for _vars in var_Q:
                for i in range(self.nchk):
                    if self.H[i, _vars] == 1:
                        if cur_chk_list[i] == 0:
                            new_chk_list[i] = 1
                            chk_Q.append(i)
            var_Q = []

            for _chks in chk_Q:
                for j in range(self.nvar):
                    if self.H[_chks, j] == 1:
                        if var_list[j] == 0:
                            var_list[j] = 1
                            var_Q.append(j)
            chk_Q = []

            if new_chk_list.count(1) == self.nchk:
                new_chk = self.find_smallest_chk(cur_chk_list)
                return new_chk
            elif new_chk_list == cur_chk_list:
                new_chk = self.find_smallest_chk(cur_chk_list)
                return new_chk
            else:
                cur_chk_list = copy.copy(new_chk_list)

    def find_smallest_chk(self, cur_chk_list):
        index = []
        degree = []
        for i in range(len(cur_chk_list)):
            if cur_chk_list[i] == 0:
                index.append(i)
                degree.append(self.chk_degrees[i])
        if not index:  # 安全检查
            return 0
        return index[find_smallest(degree)]

    def progressive_edge_growth(self, verbose=True):
        for var in range(self.nvar):
            if verbose and var % 50 == 0:
                print(f"Edge growth at var {var}/{self.nvar}")
            for k in range(self.degree_sequence[var]):
                if k == 0:
                    smallest_degree_chk = find_smallest(list(self.chk_degrees))
                    self.grow_edge(var, smallest_degree_chk)
                else:
                    chk = self.bfs(var)
                    self.grow_edge(var, chk)


class LDPCDataGenerator:
    """LDPC数据生成器 - 专门用于神经网络译码研究"""

    def __init__(self):
        self.H = None
        self.G = None  # 生成矩阵
        self.n = None  # 码长
        self.k = None  # 信息位长度
        self.m = None  # 校验位长度

    def generate_regular_ldpc(self, n, m, dv, dc=None):
        """
        生成规则LDPC码
        n: 码长（变量节点数）
        m: 校验节点数
        dv: 变量节点度数
        dc: 校验节点度数（可选，自动计算）
        """
        if dc is None:
            dc = (n * dv) // m

        print(f"生成规则({dv},{dc}) LDPC码: n={n}, m={m}")

        # 创建度序列
        degree_sequence = [dv] * n

        # 使用PEG算法
        peg = PEG(n, m, degree_sequence)
        peg.progressive_edge_growth()

        self.H = peg.H
        self.n = n
        self.m = m
        self.k = n - m

        # 验证
        actual_dv = np.mean(np.sum(self.H, axis=0))
        actual_dc = np.mean(np.sum(self.H, axis=1))
        print(f"实际平均度数: dv={actual_dv:.2f}, dc={actual_dc:.2f}")

        return self.H

    def generate_irregular_ldpc(self, n, m, dv_distribution):
        """
        生成非规则LDPC码
        dv_distribution: 变量节点度分布字典 {度数: 节点数}
        """
        degree_sequence = []
        for degree, count in dv_distribution.items():
            degree_sequence.extend([degree] * count)

        if len(degree_sequence) != n:
            raise ValueError(f"度分布总和({len(degree_sequence)})不等于n({n})")

        print(f"生成非规则LDPC码: n={n}, m={m}")
        print(f"度分布: {dv_distribution}")

        peg = PEG(n, m, degree_sequence)
        peg.progressive_edge_growth()

        self.H = peg.H
        self.n = n
        self.m = m
        self.k = n - m

        return self.H

    def analyze_matrix(self):
        """分析生成的LDPC矩阵特性"""
        if self.H is None:
            raise ValueError("请先生成LDPC矩阵")

        analysis = {
            "dimensions": f"{self.m}x{self.n}",
            "code_rate": float(self.k / self.n),
            "density": float(np.sum(self.H) / (self.m * self.n)),
            "avg_variable_degree": float(np.mean(np.sum(self.H, axis=0))),
            "avg_check_degree": float(np.mean(np.sum(self.H, axis=1))),
            "min_variable_degree": int(np.min(np.sum(self.H, axis=0))),
            "max_variable_degree": int(np.max(np.sum(self.H, axis=0))),
            "min_check_degree": int(np.min(np.sum(self.H, axis=1))),
            "max_check_degree": int(np.max(np.sum(self.H, axis=1)))
        }

        # 计算girth（最小环长）- 简化版本
        analysis["estimated_girth"] = self._estimate_girth()

        return analysis

    def _estimate_girth(self, max_check=100):
        """估计矩阵的girth（最小环长）"""
        # 这是一个简化的估计，完整的girth计算比较复杂
        # 检查是否存在长度为4的环
        for i in range(min(max_check, self.m)):
            for j in range(i + 1, min(max_check, self.m)):
                common = np.sum(self.H[i] * self.H[j])
                if common >= 2:
                    return 4
        return ">4"

    def visualize_matrix(self, save_path=None):
        """可视化LDPC矩阵"""
        if self.H is None:
            raise ValueError("请先生成LDPC矩阵")

        plt.figure(figsize=(12, 8))

        # 子图1：矩阵稀疏模式
        plt.subplot(2, 2, 1)
        plt.spy(self.H, markersize=1)
        plt.title(f'LDPC矩阵稀疏模式 ({self.m}x{self.n})')
        plt.xlabel('变量节点')
        plt.ylabel('校验节点')

        # 子图2：变量节点度分布
        plt.subplot(2, 2, 2)
        var_degrees = np.sum(self.H, axis=0)
        plt.hist(var_degrees, bins=np.arange(var_degrees.min(), var_degrees.max() + 2) - 0.5)
        plt.title('变量节点度分布')
        plt.xlabel('度数')
        plt.ylabel('节点数')

        # 子图3：校验节点度分布
        plt.subplot(2, 2, 3)
        chk_degrees = np.sum(self.H, axis=1)
        plt.hist(chk_degrees, bins=np.arange(chk_degrees.min(), chk_degrees.max() + 2) - 0.5)
        plt.title('校验节点度分布')
        plt.xlabel('度数')
        plt.ylabel('节点数')

        # 子图4：统计信息
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.9, f'码率: {self.k / self.n:.3f}', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'密度: {np.sum(self.H) / (self.m * self.n):.4f}', transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'平均变量节点度: {np.mean(var_degrees):.2f}', transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'平均校验节点度: {np.mean(chk_degrees):.2f}', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('LDPC码统计信息')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()

    def save_for_nn_training(self, save_dir, dataset_name):
        """保存数据用于神经网络训练"""
        if self.H is None:
            raise ValueError("请先生成LDPC矩阵")

        os.makedirs(save_dir, exist_ok=True)

        # 保存校验矩阵
        np.save(os.path.join(save_dir, f'{dataset_name}_H.npy'), self.H)

        # 保存为稀疏格式（节省空间）
        try:
            from scipy.sparse import csr_matrix, save_npz
            H_sparse = csr_matrix(self.H)
            save_npz(os.path.join(save_dir, f'{dataset_name}_H_sparse.npz'), H_sparse)
        except ImportError:
            print("警告：未安装scipy，跳过稀疏矩阵保存")

        # 保存参数
        params = {
            'n': int(self.n),
            'k': int(self.k),
            'm': int(self.m),
            'code_rate': float(self.k / self.n),
            'analysis': self.analyze_matrix(),
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(save_dir, f'{dataset_name}_params.json'), 'w') as f:
            json.dump(params, f, indent=2)

        # 生成Tanner图的邻接表示（用于GNN）
        self._save_tanner_graph(save_dir, dataset_name)

        print(f"数据已保存到: {save_dir}")
        print(f"- {dataset_name}_H.npy: 校验矩阵")
        if 'scipy' in globals():
            print(f"- {dataset_name}_H_sparse.npz: 稀疏格式校验矩阵")
        print(f"- {dataset_name}_params.json: 参数文件")
        print(f"- {dataset_name}_tanner_*.npy: Tanner图表示")

    def _save_tanner_graph(self, save_dir, dataset_name):
        """保存Tanner图表示（用于图神经网络）"""
        # 变量节点到校验节点的边
        edges_v2c = []
        edges_c2v = []

        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    edges_v2c.append([j, i])  # 变量节点j到校验节点i
                    edges_c2v.append([i, j])  # 校验节点i到变量节点j

        edges_v2c = np.array(edges_v2c)
        edges_c2v = np.array(edges_c2v)

        np.save(os.path.join(save_dir, f'{dataset_name}_tanner_v2c.npy'), edges_v2c)
        np.save(os.path.join(save_dir, f'{dataset_name}_tanner_c2v.npy'), edges_c2v)


def display_menu():
    """显示主菜单"""
    print("\n" + "=" * 60)
    print("         LDPC码生成器 - 交互式版本")
    print("=" * 60)
    print("请选择要生成的LDPC码类型:")
    print("1. 小规模规则LDPC码 (64, 32, dv=3) - 适合快速测试")
    print("2. 中等规模规则LDPC码 (1000, 500, dv=3) - 标准测试")
    print("3. 5G-like LDPC码 (1944, 972, dv=3) - 实际应用")
    print("4. 非规则LDPC码 (1000, 500) - 高级研究")
    print("5. 自定义规则LDPC码 - 自定义参数")
    print("6. 自定义非规则LDPC码 - 自定义度分布")
    print("7. 批量生成预设码 - 一次生成多种")
    print("0. 退出")
    print("-" * 60)


def generate_small_regular():
    """生成小规模规则LDPC码"""
    print("\n生成小规模规则LDPC码...")
    generator = LDPCDataGenerator()
    H = generator.generate_regular_ldpc(n=64, m=32, dv=3)

    # 分析并显示结果
    analysis = generator.analyze_matrix()
    print(f"分析结果: {json.dumps(analysis, indent=2)}")

    # 询问是否保存
    save = input("\n是否保存生成的LDPC码? (y/n): ").lower().strip()
    if save == 'y':
        os.makedirs('ldpc_data', exist_ok=True)
        generator.save_for_nn_training('ldpc_data', 'small_regular_64_32_3')

        # 询问是否生成可视化
        viz = input("是否生成可视化图像? (y/n): ").lower().strip()
        if viz == 'y':
            generator.visualize_matrix('ldpc_small_visualization.png')

    return generator


def generate_medium_regular():
    """生成中等规模规则LDPC码"""
    print("\n生成中等规模规则LDPC码...")
    generator = LDPCDataGenerator()
    H = generator.generate_regular_ldpc(n=1000, m=500, dv=3)

    analysis = generator.analyze_matrix()
    print(f"分析结果: {json.dumps(analysis, indent=2)}")

    save = input("\n是否保存生成的LDPC码? (y/n): ").lower().strip()
    if save == 'y':
        os.makedirs('ldpc_data', exist_ok=True)
        generator.save_for_nn_training('ldpc_data', 'medium_regular_1000_500_3')

    return generator


def generate_5g_like():
    """生成5G-like LDPC码"""
    print("\n生成5G-like LDPC码...")
    generator = LDPCDataGenerator()
    H = generator.generate_regular_ldpc(n=1944, m=972, dv=3)

    analysis = generator.analyze_matrix()
    print(f"分析结果: {json.dumps(analysis, indent=2)}")

    save = input("\n是否保存生成的LDPC码? (y/n): ").lower().strip()
    if save == 'y':
        os.makedirs('ldpc_data', exist_ok=True)
        generator.save_for_nn_training('ldpc_data', '5g_like_1944_972')

    return generator


def generate_irregular():
    """生成非规则LDPC码"""
    print("\n生成非规则LDPC码...")
    generator = LDPCDataGenerator()

    # 度分布：大部分节点度为3，少数度为2和4
    dv_dist = {2: 100, 3: 700, 4: 200}
    H = generator.generate_irregular_ldpc(n=1000, m=500, dv_distribution=dv_dist)

    analysis = generator.analyze_matrix()
    print(f"分析结果: {json.dumps(analysis, indent=2)}")

    save = input("\n是否保存生成的LDPC码? (y/n): ").lower().strip()
    if save == 'y':
        os.makedirs('ldpc_data', exist_ok=True)
        generator.save_for_nn_training('ldpc_data', 'irregular_1000_500')

    return generator


def generate_custom_regular():
    """生成自定义规则LDPC码"""
    print("\n自定义规则LDPC码生成")
    print("请输入参数（按Enter使用默认值）:")

    try:
        n = input("码长 n (默认: 500): ").strip()
        n = int(n) if n else 500

        m = input(f"校验节点数 m (默认: {n // 2}): ").strip()
        m = int(m) if m else n // 2

        dv = input("变量节点度数 dv (默认: 3): ").strip()
        dv = int(dv) if dv else 3

        # 验证参数
        if n <= 0 or m <= 0 or dv <= 0:
            raise ValueError("参数必须为正数")
        if m >= n:
            raise ValueError("校验节点数必须小于码长")

        print(f"\n生成参数: n={n}, m={m}, dv={dv}")

        generator = LDPCDataGenerator()
        H = generator.generate_regular_ldpc(n=n, m=m, dv=dv)

        analysis = generator.analyze_matrix()
        print(f"分析结果: {json.dumps(analysis, indent=2)}")

        save = input("\n是否保存生成的LDPC码? (y/n): ").lower().strip()
        if save == 'y':
            name = input("输入数据集名称 (默认: custom_regular): ").strip()
            name = name if name else "custom_regular"
            os.makedirs('ldpc_data', exist_ok=True)
            generator.save_for_nn_training('ldpc_data', name)

        return generator

    except ValueError as e:
        print(f"参数错误: {e}")
        return None


def generate_custom_irregular():
    """生成自定义非规则LDPC码"""
    print("\n自定义非规则LDPC码生成")

    try:
        n = input("码长 n (默认: 500): ").strip()
        n = int(n) if n else 500

        m = input(f"校验节点数 m (默认: {n // 2}): ").strip()
        m = int(m) if m else n // 2

        print(f"\n需要定义{n}个变量节点的度分布")
        print("格式：度数:节点数，用逗号分隔")
        print("例如：2:100,3:300,4:100 表示100个度为2的节点，300个度为3的节点，100个度为4的节点")

        dist_str = input("度分布: ").strip()
        if not dist_str:
            # 默认分布
            dv_dist = {2: n // 5, 3: n * 3 // 5, 4: n // 5}
        else:
            dv_dist = {}
            for pair in dist_str.split(','):
                degree, count = pair.split(':')
                dv_dist[int(degree)] = int(count)

        # 验证度分布
        total_nodes = sum(dv_dist.values())
        if total_nodes != n:
            print(f"警告：度分布总节点数({total_nodes})不等于n({n})")
            if total_nodes < n:
                # 自动补充度为3的节点
                if 3 in dv_dist:
                    dv_dist[3] += (n - total_nodes)
                else:
                    dv_dist[3] = n - total_nodes
                print(f"自动调整后的度分布: {dv_dist}")
            else:
                raise ValueError("度分布总节点数超过n")

        print(f"\n生成参数: n={n}, m={m}, 度分布={dv_dist}")

        generator = LDPCDataGenerator()
        H = generator.generate_irregular_ldpc(n=n, m=m, dv_distribution=dv_dist)

        analysis = generator.analyze_matrix()
        print(f"分析结果: {json.dumps(analysis, indent=2)}")

        save = input("\n是否保存生成的LDPC码? (y/n): ").lower().strip()
        if save == 'y':
            name = input("输入数据集名称 (默认: custom_irregular): ").strip()
            name = name if name else "custom_irregular"
            os.makedirs('ldpc_data', exist_ok=True)
            generator.save_for_nn_training('ldpc_data', name)

        return generator

    except ValueError as e:
        print(f"参数错误: {e}")
        return None


def batch_generate():
    """批量生成预设的LDPC码"""
    print("\n批量生成模式")
    print("将生成以下预设的LDPC码：")
    print("1. 小规模规则LDPC码 (64, 32, dv=3)")
    print("2. 中等规模规则LDPC码 (1000, 500, dv=3)")
    print("3. 5G-like LDPC码 (1944, 972, dv=3)")
    print("4. 非规则LDPC码 (1000, 500)")

    confirm = input("\n确认批量生成? (y/n): ").lower().strip()
    if confirm != 'y':
        return

    os.makedirs('ldpc_data', exist_ok=True)

    # 生成各种码
    generators = []

    try:
        print("\n[1/4] 生成小规模规则LDPC码...")
        gen1 = LDPCDataGenerator()
        gen1.generate_regular_ldpc(n=64, m=32, dv=3)
        gen1.save_for_nn_training('ldpc_data', 'small_regular_64_32_3')
        generators.append(("小规模规则", gen1))

        print("\n[2/4] 生成中等规模规则LDPC码...")
        gen2 = LDPCDataGenerator()
        gen2.generate_regular_ldpc(n=1000, m=500, dv=3)
        gen2.save_for_nn_training('ldpc_data', 'medium_regular_1000_500_3')
        generators.append(("中等规模规则", gen2))

        print("\n[3/4] 生成5G-like LDPC码...")
        gen3 = LDPCDataGenerator()
        gen3.generate_regular_ldpc(n=1944, m=972, dv=3)
        gen3.save_for_nn_training('ldpc_data', '5g_like_1944_972')
        generators.append(("5G-like", gen3))

        print("\n[4/4] 生成非规则LDPC码...")
        gen4 = LDPCDataGenerator()
        dv_dist = {2: 100, 3: 700, 4: 200}
        gen4.generate_irregular_ldpc(n=1000, m=500, dv_distribution=dv_dist)
        gen4.save_for_nn_training('ldpc_data', 'irregular_1000_500')
        generators.append(("非规则", gen4))

        print("\n" + "=" * 50)
        print("批量生成完成！生成的LDPC码统计:")
        for name, gen in generators:
            analysis = gen.analyze_matrix()
            print(f"\n{name} LDPC码:")
            print(f"  - 维度: {analysis['dimensions']}")
            print(f"  - 码率: {analysis['code_rate']:.3f}")
            print(f"  - 密度: {analysis['density']:.4f}")

    except Exception as e:
        print(f"\n批量生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数 - 交互式LDPC码生成"""

    print("LDPC码生成器启动...")

    while True:
        display_menu()

        try:
            choice = input("请选择操作 (0-7): ").strip()

            if choice == '0':
                print("感谢使用LDPC码生成器！")
                break
            elif choice == '1':
                generate_small_regular()
            elif choice == '2':
                generate_medium_regular()
            elif choice == '3':
                generate_5g_like()
            elif choice == '4':
                generate_irregular()
            elif choice == '5':
                generate_custom_regular()
            elif choice == '6':
                generate_custom_irregular()
            elif choice == '7':
                batch_generate()
            else:
                print("无效选择，请重新输入！")

        except KeyboardInterrupt:
            print("\n\n用户中断操作")
            break
        except Exception as e:
            print(f"\n操作过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

        # 询问是否继续
        if choice != '0':
            continue_choice = input("\n是否继续使用生成器? (y/n): ").lower().strip()
            if continue_choice != 'y':
                print("感谢使用LDPC码生成器！")
                break


if __name__ == "__main__":
    main()
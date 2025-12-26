# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - https://herve.niderb.fr
# Alexis PLAQUET

# 标准库导入
from functools import cached_property  # 用于缓存属性计算结果
from itertools import combinations, permutations  # 用于生成组合和排列
from typing import Dict, Tuple  # 类型提示

# 第三方库导入
import scipy.special  # 用于计算二项式系数（组合数）
import torch  # PyTorch核心库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口（用于one_hot等操作）


class Powerset(nn.Module):
    """幂集编码/解码模块
    
    实现幂集（powerset）编码和多标签之间的转换。
    幂集编码是一种将多标签分类问题转换为单标签分类问题的方法。
    
    原理：
    - 将多个类别的组合（子集）映射为单个"幂集类别"
    - 例如：3个类别，最大集合大小为2
      - 空集 {} → 类别0
      - {0} → 类别1
      - {1} → 类别2
      - {2} → 类别3
      - {0,1} → 类别4
      - {0,2} → 类别5
      - {1,2} → 类别6
    
    优势：
    - 可以处理重叠标签（多个说话人同时说话）
    - 将多标签问题转换为单标签问题，简化训练
    
    参数
    ----------
    num_classes : int
        常规类别数量（例如：说话人数量）
    max_set_size : int
        每个集合中的最大类别数（例如：最大同时说话人数）
    
    使用场景
    --------
    主要用于说话人分离任务，处理重叠语音：
    - 训练时：将多标签（哪些说话人活跃）转换为幂集类别
    - 推理时：将幂集类别转换回多标签
    """

    def __init__(self, num_classes: int, max_set_size: int):
        """初始化幂集编码模块
        
        参数
        ----------
        num_classes : int
            常规类别数量（例如：说话人数量）
        max_set_size : int
            每个集合中的最大类别数（例如：最大同时说话人数）
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_set_size = max_set_size

        # 注册映射矩阵和基数张量作为缓冲区（不参与梯度计算，但会随模型移动设备）
        # persistent=False 表示这些缓冲区不会保存到模型checkpoint中
        self.register_buffer("mapping", self.build_mapping(), persistent=False)
        self.register_buffer("cardinality", self.build_cardinality(), persistent=False)

    @cached_property
    def num_powerset_classes(self) -> int:
        """计算幂集类别数量
        
        计算大小不超过max_set_size的所有子集的数量。
        使用组合数学：C(n,0) + C(n,1) + ... + C(n,k)
        其中n=num_classes，k=max_set_size
        
        示例
        -----
        当num_classes=3，max_set_size=2时：
        - C(3,0) = 1: {}
        - C(3,1) = 3: {0}, {1}, {2}
        - C(3,2) = 3: {0,1}, {0,2}, {1,2}
        总计：1 + 3 + 3 = 7个幂集类别
        
        返回
        -------
        int
            幂集类别总数
        """
        # 计算所有大小不超过max_set_size的子集数量
        # 使用二项式系数：C(num_classes, i) for i in [0, max_set_size]
        return int(
            sum(
                scipy.special.binom(self.num_classes, i)
                for i in range(0, self.max_set_size + 1)
            )
        )

    def build_mapping(self) -> torch.Tensor:
        """构建幂集到常规类别的映射矩阵
        
        创建一个映射矩阵，将幂集类别转换为多标签表示。
        矩阵的每一行对应一个幂集类别，每一列对应一个常规类别。
        
        返回
        -------
        torch.Tensor
            映射矩阵，形状为(num_powerset_classes, num_classes)
            mapping[i, j] == 1: 第j个常规类别是第i个幂集类别的成员
            mapping[i, j] == 0: 第j个常规类别不是第i个幂集类别的成员
        
        示例
        -------
        当num_classes=3，max_set_size=2时，返回：
        
            [0, 0, 0]  # 空集 {}
            [1, 0, 0]  # {0}
            [0, 1, 0]  # {1}
            [0, 0, 1]  # {2}
            [1, 1, 0]  # {0, 1}
            [1, 0, 1]  # {0, 2}
            [0, 1, 1]  # {1, 2}
        
        用途
        -----
        用于将模型输出的幂集类别（单标签）转换回多标签表示
        """
        # 初始化映射矩阵，全零
        mapping = torch.zeros(self.num_powerset_classes, self.num_classes)
        powerset_k = 0  # 当前幂集类别的索引
        
        # 遍历所有可能的集合大小：从0（空集）到max_set_size
        for set_size in range(0, self.max_set_size + 1):
            # 生成所有大小为set_size的组合（子集）
            # 例如：set_size=1时，生成{0}, {1}, {2}
            #      set_size=2时，生成{0,1}, {0,2}, {1,2}
            for current_set in combinations(range(self.num_classes), set_size):
                # 将当前组合对应的位置标记为1
                # current_set是一个元组，例如(0,1)，表示类别0和1都在这个幂集类别中
                mapping[powerset_k, current_set] = 1
                powerset_k += 1  # 移动到下一个幂集类别

        return mapping

    def build_cardinality(self) -> torch.Tensor:
        """计算每个幂集类别的基数（集合大小）
        
        计算每个幂集类别包含的常规类别数量。
        通过计算映射矩阵每一行的和来得到。
        
        返回
        -------
        torch.Tensor
            形状为(num_powerset_classes,)的张量
            每个元素表示对应幂集类别包含的常规类别数量
            
        示例
        -----
        当num_classes=3，max_set_size=2时：
        - 空集 {} → 基数 = 0
        - {0}, {1}, {2} → 基数 = 1
        - {0,1}, {0,2}, {1,2} → 基数 = 2
        
        返回: [0, 1, 1, 1, 2, 2, 2]
        """
        # 对映射矩阵的每一行求和，得到每个幂集类别包含的类别数
        return torch.sum(self.mapping, dim=1)

    def to_multilabel(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """将幂集空间的预测转换为多标签空间

        这是推理时使用的主要方法，将模型输出的幂集类别转换为多标签表示。
        
        参数
        ---------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            幂集空间的预测值，通常是log_softmax的输出
            即：log_softmax(logits)，形状为 (batch_size, num_frames, num_powerset_classes)
        soft : bool, optional
            是否返回软多标签预测。默认为False（即硬预测，one-hot）
            - True: 返回概率分布，每个类别有对应的概率值
            - False: 返回硬预测，只保留概率最高的幂集类别对应的多标签
            注意：当soft=True时，会对log_softmax输出进行exp操作得到概率

        返回
        -------
        multi_label : (batch_size, num_frames, num_classes) torch.Tensor
            多标签空间的预测值
            - 如果soft=False: 每个元素为0或1，表示该类别是否活跃
            - 如果soft=True: 每个元素为概率值，表示该类别活跃的概率
            
        工作原理
        --------
        1. 将幂集预测转换为概率分布（硬预测或软预测）
           - soft=False: 在log_softmax输出上直接argmax（log_softmax是单调的，argmax结果不变）
           - soft=True: 对log_softmax输出进行exp，得到softmax概率
        2. 通过矩阵乘法将幂集概率映射到多标签空间
        3. 映射公式：multi_label = powerset_probs @ mapping
           其中mapping的每一行表示一个幂集类别对应的多标签向量
        """

        if soft:
            # 软预测模式：将log_softmax输出转换为概率
            # exp(log_softmax(x)) = softmax(x)，得到归一化的概率分布
            powerset_probs = torch.exp(powerset)
        else:
            # 硬预测模式：选择概率最高的幂集类别，转换为one-hot编码
            # 1. argmax在log_softmax输出上找到概率最高的幂集类别索引
            #    注意：log_softmax是单调函数，argmax结果与在原始logits上相同
            # 2. one_hot将索引转换为one-hot向量
            powerset_probs = torch.nn.functional.one_hot(
                torch.argmax(powerset, dim=-1),
                self.num_powerset_classes,
            ).float()

        # 矩阵乘法：将幂集概率分布映射到多标签空间
        # powerset_probs: (batch_size, num_frames, num_powerset_classes)
        # mapping: (num_powerset_classes, num_classes)
        # 结果: (batch_size, num_frames, num_classes)
        return torch.matmul(powerset_probs, self.mapping)

    def forward(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """前向传播方法（`to_multilabel`的别名）
        
        作为nn.Module的forward方法，可以直接通过调用模块实例来使用。
        例如：multilabel = powerset_module(powerset_tensor)
        
        参数
        ---------
        powerset : torch.Tensor
            幂集空间的预测值，通常是log_softmax的输出
        soft : bool, optional
            是否返回软预测，默认为False
            
        返回
        -------
        torch.Tensor
            多标签空间的预测值
        """
        return self.to_multilabel(powerset, soft=soft)

    def to_powerset(self, multilabel: torch.Tensor) -> torch.Tensor:
        """将多标签空间的预测转换为幂集空间（硬预测）

        这是训练时使用的方法，将多标签标注转换为幂集类别。
        
        参数
        ---------
        multilabel : (batch_size, num_frames, num_classes) torch.Tensor
            多标签空间的预测值
            - 理想情况下应该是硬预测（0或1）
            - 也可以是软预测（概率值），但结果可能不准确

        返回
        -------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            幂集空间的硬预测（one-hot编码）
            每个时间步只有一个幂集类别为1，其余为0
            
        工作原理
        --------
        1. 通过矩阵乘法计算多标签与每个幂集类别的匹配度
           similarity = multilabel @ mapping.T
           其中mapping.T的每一列表示一个幂集类别对应的多标签向量
        2. 选择匹配度最高的幂集类别（argmax）
        3. 转换为one-hot编码
        
        注意
        ----
        如果multilabel是软预测（例如sigmoid输出的概率），该方法不会报错，
        但转换结果可能不准确。建议使用硬预测（0或1）作为输入。
        
        示例
        -----
        假设num_classes=3，max_set_size=2，mapping为：
        [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
        对应幂集类别：{}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}
        
        情况1：multilabel = [1, 0, 0]（表示类别0活跃）
        则similarity = [1,0,0] @ mapping.T = [0, 1, 0, 0, 1, 1, 0]
        最大值为1，对应索引1（argmax返回第一个最大值索引），即幂集类别{0}
        
        情况2：multilabel = [1, 1, 0]（表示类别0和1活跃）
        则similarity = [1,1,0] @ mapping.T = [0, 1, 1, 0, 2, 1, 1]
        最大值为2，对应索引4，即幂集类别{0,1}
        """
        # 计算多标签与每个幂集类别的匹配度
        # multilabel: (batch_size, num_frames, num_classes)
        # mapping.T: (num_classes, num_powerset_classes)
        # similarity: (batch_size, num_frames, num_powerset_classes)
        similarity = torch.matmul(multilabel, self.mapping.T)
        
        # 找到匹配度最高的幂集类别索引
        best_powerset_idx = torch.argmax(similarity, dim=-1)
        
        # 转换为one-hot编码
        return F.one_hot(
            best_powerset_idx,
            num_classes=self.num_powerset_classes,
        )

    def _permutation_powerset(
        self, multilabel_permutation: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """计算多标签空间排列对应的幂集空间排列（辅助函数）

        当多标签空间的类别顺序发生变化时（例如说话人标签重新编号），
        需要计算对应的幂集空间排列，以保持一致性。
        
        参数
        ----------
        multilabel_permutation : tuple of int
            多标签空间的排列
            例如：(1, 0, 2) 表示原来的类别0→新类别1，类别1→新类别0，类别2→新类别2

        返回
        -------
        powerset_permutation : tuple of int
            对应的幂集空间排列
            例如：(0, 2, 1, 3, 4, 6, 5) 表示原来的幂集类别0→新幂集类别0，等等
            
        工作原理
        --------
        1. 对映射矩阵的列进行排列，得到排列后的映射矩阵
        2. 使用二进制编码（2的幂次）为每个幂集类别创建唯一标识
        3. 计算排列前后的编码值
        4. 通过匹配编码值找到对应的排列关系
        
        示例
        -------
        >>> powerset = Powerset(3, 2)
        >>> powerset._permutation_powerset((1, 0, 2))
        (0, 2, 1, 3, 4, 6, 5)
        
        注意
        ----
        这是一个辅助函数，不缓存结果，每次调用都重新计算。
        主要用于构建permutation_mapping属性。
        """

        # 对映射矩阵的列进行排列（列对应多标签类别）
        # 例如：如果multilabel_permutation=(1,0,2)，则原来的列[0,1,2]变为[1,0,2]
        permutated_mapping: torch.Tensor = self.mapping[:, multilabel_permutation]

        # 创建2的幂次数组：[2^0, 2^1, 2^2, ...] = [1, 2, 4, ...]
        # 用于为每个类别创建唯一的二进制编码
        arange = torch.arange(
            self.num_classes, device=self.mapping.device, dtype=torch.int
        )
        # 将幂次数组扩展为矩阵，每行相同
        powers_of_two = (2**arange).tile((self.num_powerset_classes, 1))

        # 计算排列前的编码：每个幂集类别用二进制编码表示
        # 例如：{0,1} → 1*2^0 + 1*2^1 = 3
        # 这样每个幂集类别都有唯一的编码值
        before = torch.sum(self.mapping * powers_of_two, dim=-1)
        
        # 计算排列后的编码：使用排列后的映射矩阵
        after = torch.sum(permutated_mapping * powers_of_two, dim=-1)

        # 通过匹配编码值找到排列关系
        # before[None] == after[:, None] 创建一个比较矩阵
        # argmax找到每个排列前编码对应的排列后编码索引
        powerset_permutation = (before[None] == after[:, None]).int().argmax(dim=0)

        # 返回为元组格式
        return tuple(powerset_permutation.tolist())

    @cached_property
    def permutation_mapping(self) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        """多标签空间和幂集空间之间的排列映射字典

        预计算所有可能的多标签排列对应的幂集排列。
        这个映射在说话人分离任务中很有用，因为说话人标签的顺序可能发生变化。
        
        返回
        -------
        Dict[Tuple[int, ...], Tuple[int, ...]]
            字典，键为多标签空间的排列，值为对应的幂集空间排列
            键和值都是元组格式
            
        示例
        -------
        当num_classes=3，max_set_size=2时，返回所有6种排列的映射：
        
        {
            (0, 1, 2): (0, 1, 2, 3, 4, 5, 6),  # 无变化
            (0, 2, 1): (0, 1, 3, 2, 5, 4, 6),  # 类别1和2交换
            (1, 0, 2): (0, 2, 1, 3, 4, 6, 5),  # 类别0和1交换
            (1, 2, 0): (0, 2, 3, 1, 6, 4, 5),  # 循环排列
            (2, 0, 1): (0, 3, 1, 2, 5, 6, 4),  # 循环排列
            (2, 1, 0): (0, 3, 2, 1, 6, 5, 4)   # 完全反转
        }
        
        使用场景
        --------
        在模型训练或推理时，如果说话人标签的顺序发生变化（例如通过某种对齐算法），
        可以使用这个映射来调整幂集预测，保持一致性。
        
        注意
        ----
        这是一个缓存属性，首次访问时计算，之后直接返回缓存结果。
        排列数量为num_classes!（阶乘），当num_classes较大时计算量会很大。
        """
        permutation_mapping = {}

        # 遍历所有可能的多标签排列（共num_classes!种）
        for multilabel_permutation in permutations(
            range(self.num_classes), self.num_classes
        ):
            # 计算对应的幂集排列并存储
            permutation_mapping[
                tuple(multilabel_permutation)
            ] = self._permutation_powerset(multilabel_permutation)

        return permutation_mapping

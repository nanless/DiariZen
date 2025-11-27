# MIT License
#
# Copyright (c) 2020-2021 CNRS
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

"""
pyannote.audio 包初始化模块

本模块是pyannote.audio包的入口点，导出核心类和接口。
主要导出四个核心类：
- Audio: 音频I/O处理类
- Model: 模型基类
- Inference: 推理引擎类
- Pipeline: 处理管道基类
"""

# 尝试导入版本信息（如果可用）
try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


# 从核心模块导入主要类
from .core.inference import Inference  # 推理引擎：处理模型推理和滑动窗口
from .core.io import Audio  # 音频I/O：处理音频文件的读取、重采样和裁剪
from .core.model import Model  # 模型基类：所有音频模型的抽象基类
from .core.pipeline import Pipeline  # 管道基类：处理管道的抽象基类

# 定义包的公共API，只导出这四个核心类
__all__ = ["Audio", "Model", "Inference", "Pipeline"]

#!/bin/bash
# 简洁的说话人分离推理脚本启动器
# 自动使用 diarizen conda 环境运行

# 脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 使用 conda run 执行 Python 脚本
conda run -n diarizen python "${SCRIPT_DIR}/simple_diarize.py" "$@"


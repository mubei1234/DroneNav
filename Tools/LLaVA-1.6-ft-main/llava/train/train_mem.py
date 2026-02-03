import sys
from pathlib import Path

# 获取当前文件的父目录的父目录（项目根目录）
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

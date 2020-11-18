from __future__ import print_function
import torch


if __name__ == '__main__':
    # 초기화되지 않은 5x3 행렬을 생성합니다.
    x = torch.empty(5, 3)
    print(x)

    # 무작위로 초기화된 행렬을 생성합니다.
    y = torch.rand(5, 3)
    print(y)

    # dtype이 long이고 0으로 채워진 행렬을 생성합니다.
    z = torch.zeros(5, 3, dtype=torch.long)
    print(z)

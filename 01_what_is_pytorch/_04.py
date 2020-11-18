from __future__ import print_function
import torch

if __name__ == '__main__':
    # 크기 변경: tensor의 크기나 모양을 변경하고 싶다면 torch.view를 사용합니다.
    x = torch.randn(4, 4)
    y = x.view(16)
    # -1은 다른 차원에서 유추합니다.
    z = x.view(2, -1)
    print(x.size(), y.size(), z.size())

    # 만약 tensor에 하나의 값만 존재한다면, .item()을 사용하면 숫자 값을 얻을 수 있습니다.
    x = torch.randn(1)
    print(x)
    print(x.item())

from __future__ import print_function
import torch

if __name__ == '__main__':
    # 데이터로부터 tensor를 직접 생성합니다.
    x = torch.tensor([5.5, 3])
    print(x)

    # 또는 기존 tensor를 바탕으로 새로운 tensor를 만듭니다.
    # 이들 메소드(method)는 사용자로부터 새로운 값을 제공받지 않은 한,
    # 입력 tensor의 속성들(예. dtype)을 재사용합니다.

    # new_* 메소드는 크기를 받습니다.
    x = x.new_ones(5, 3, dtype=torch.double)
    print(x)

    # dtype을 오버라이드(Override) 합니다!
    x = torch.randn_like(x, dtype=torch.float)
    # 결과는 동일한 크기를 갖습니다.
    print(x)

    # 행렬의 크기를 구합니다.
    # 결과는 튜플이며, 모든 튜플 연산을 지원합니다.
    print(x.size())

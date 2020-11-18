from __future__ import print_function
import torch

if __name__ == '__main__':
    x = torch.rand(5, 3)
    y = torch.rand(5, 3)

    # 여러가지 덧셈 방식
    print(x + y)
    print(torch.add(x, y))

    # 덧셈 결과 tensor를 인자로 제공
    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print(result)

    # 바꿔치기(in-place) 방식
    # 바꿔치기(in-place) 방식으로 tensor의 값을 변경하는 연산 뒤에는 _``가 붙습니다.
    # 예: x.copy_(y), x.t_()는 x를 변경합니다.
    y.add_(x)
    print(y)

    # Numpy스러운 인덱싱 표기 방법을 사용할 수도 있습니다.
    print(y[:, 1])

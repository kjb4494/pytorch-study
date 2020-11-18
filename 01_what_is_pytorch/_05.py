from __future__ import print_function
import torch
import numpy as np

if __name__ == '__main__':
    """
    NumPy 변환(Bridge)
    Torch Tensor를 NumPy 배열(array)로 변환하거나, 그 반대로 하는 것은 매우 쉽습니다.
    (Torch Tensor가 CPU 상에 있다면) Torch Tensor와 NumPy 배열은 메모리 공간을 공유하기 때문에, 
    하나를 변경하면 다른 하나도 변경됩니다.
    """

    # Torch Tensor를 NumPy 배열로 변환하기
    a = torch.ones(5)
    print(a)
    b = a.numpy()
    print(b)

    # Numpy배열의 값도 같이 변함
    a.add_(1)
    print(a)
    print(b)

    print()

    # NumPy 배열을 Torch Tensor로 변환하기
    c = np.ones(5)
    d = torch.from_numpy(c)
    np.add(c, 1, out=c)
    print(c)
    print(d)

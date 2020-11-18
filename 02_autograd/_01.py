import torch

if __name__ == '__main__':
    # tensor를 생성하고 requires_grad=True 를 설정하여 연산을 기록합니다.
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    # tensor에 연산을 수행합니다.
    y = x + 2
    print(y)

    # y 는 연산의 결과로 생성된 것이므로 grad_fn 을 갖습니다.
    print(y.grad_fn)

    # y에 다른 연산을 수행합니다.
    z = y * y * 3
    out = z.mean()
    print(z, out)

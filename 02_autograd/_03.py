import torch

if __name__ == '__main__':
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()

    # 변화도(gradient)
    # 역전파(backprop) -> out은 하나의 스칼라 값만 가지고 있기 때문에,
    # out.backward()는 out.backward(torch.tensor(1.))과 동일합니다.
    out.backward()

    # 변화도 d(out)/dx를 출력합니다.
    print(x.grad)

    # 벡터-야코비안 곱의 예제
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print(y)

    # 이 경우 y는 더 이상 스칼라 값이 아닙니다.
    # torch.autograd는 전체 야코비안을 직접 계산할수는 없지만,
    # 벡터-야코비안 곱은 간단히 backward에 해당 벡터를 인자로 제공하여 얻을 수 있습니다
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)
    print(x.grad)

    # 또한 with torch.no_grad(): 로 코드 블럭을 감싸서
    # autograd가 .requires_grad=True인
    # Tensor들의 연산 기록을 추적하는 것을 멈출 수 있습니다.
    print(x.requires_grad)
    print((x ** 2).requires_grad)
    with torch.no_grad():
        print((x ** 2).requires_grad)

    print()

    # 또는 .detach() 를 호출하여 내용물(content)은 같지만
    # require_grad가 다른 새로운 Tensor를 가져옵니다:
    print(x.requires_grad)
    y = x.detach()
    print(y.requires_grad)
    print(x.eq(y).all())

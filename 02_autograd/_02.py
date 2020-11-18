import torch

if __name__ == '__main__':
    # .requires_grad_( ... ) 는 기존 Tensor의
    # requires_grad 값을 바꿔치기 (in-place)하여 변경합니다.
    # 입력값이 지정되지 않으면 기본값은 False 입니다.
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)

    a.requires_grad_(True)
    print(a.requires_grad)

    b = (a * a).sum()
    print(b.grad_fn)

from krnl import nn, Tensor


def test_linear():
    layer = nn.Linear(3, 5, bias=True)
    x = Tensor([[1.0, 2.0, 3.0]])
    y = layer(x)
    print("Linear output shape:", y.shape)
    assert y.shape == (1, 5)


def test_relu():
    relu = nn.ReLU()
    x = Tensor([[-1.0, 0.0, 1.0]])
    y = relu(x)
    print("ReLU output:", y.tolist())
    assert y.tolist() == [[0.0, 0.0, 1.0]]


def test_sigmoid():
    sigmoid = nn.Sigmoid()
    x = Tensor([[0.0]])
    y = sigmoid(x)
    print("Sigmoid output:", y.tolist())
    assert abs(y.tolist()[0][0] - 0.5) < 1e-5


def test_tanh():
    tanh = nn.Tanh()
    x = Tensor([[0.0]])
    y = tanh(x)
    print("Tanh output:", y.tolist())
    assert abs(y.tolist()[0][0]) < 1e-5


def test_gelu():
    gelu = nn.GELU()
    x = Tensor([[0.0, 1.0]])
    y = gelu(x)
    print("GELU output:", y.tolist())
    assert abs(y.tolist()[0][0]) < 1e-5


def test_silu():
    silu = nn.SiLU()
    x = Tensor([[0.0, 1.0]])
    y = silu(x)
    print("SiLU output:", y.tolist())
    assert abs(y.tolist()[0][0]) < 1e-5


def test_softmax():
    softmax = nn.Softmax()
    x = Tensor([[1.0, 2.0, 3.0]])
    y = softmax(x)
    print("Softmax output:", y.tolist())
    assert abs(sum(y.tolist()[0]) - 1.0) < 1e-5


def test_sequential():
    model = nn.Sequential(
        nn.Linear(2, 4, bias=True),
        nn.ReLU(),
        nn.Linear(4, 1, bias=True),
    )
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    print("Sequential output shape:", y.shape)
    assert y.shape == (1, 1)
    assert len(model._parameters) == 4


def test_residual():
    block = nn.Residual(nn.Linear(3, 3))
    x = Tensor([[1.0, 2.0, 3.0]])
    y = block(x)
    print("Residual output shape:", y.shape)
    assert y.shape == (1, 3)


def test_parameter():
    p = nn.Parameter([1.0, 2.0, 3.0])
    print("Parameter requires_grad:", p.requires_grad)
    assert p.requires_grad is True


def test_train_eval_mode():
    model = nn.Linear(2, 2)
    assert model.training is True
    model.eval()
    assert model.training is False
    model.train()
    assert model.training is True
    print("Train/eval mode: OK")


def test_zero_grad():
    model = nn.Linear(2, 1, bias=True)
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    y.sum().backward()
    assert model._parameters["weight"].grad is not None
    model.zero_grad()
    assert model._parameters["weight"].grad is None
    print("Zero grad: OK")


def test_state_dict():
    model = nn.Linear(2, 3)
    state = model.get_model_state_dict()
    assert "parameters" in state
    assert "weight" in state["parameters"]
    print("State dict: OK")


if __name__ == "__main__":
    test_linear()
    test_relu()
    test_sigmoid()
    test_tanh()
    test_gelu()
    test_silu()
    test_softmax()
    test_sequential()
    test_residual()
    test_parameter()
    test_train_eval_mode()
    test_zero_grad()
    test_state_dict()
    print("\nAll module tests passed!")

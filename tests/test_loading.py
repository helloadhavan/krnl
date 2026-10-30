import numpy as np
from krnl import save, load, nn, Device

# --- Create model and optimizer ---
model = nn.Linear(
    in_features=1,
    out_features=1,
    bias=True,
    device=Device("CPU"),
)

optimizer = nn.optim.SGD(
    model._parameters.values()
)

save(model.get_model_state_dict(), "model")
save(optimizer.get_state_dict(), "optim")

print("Saved model and optimizer state dicts.")

model_state = load("model")
optim_state = load("optim")

print(f"Loaded model state keys: {list(model_state.keys())}")
print(f"Loaded optim state keys: {list(optim_state.keys())}")

model_params = model_state["parameters"]
print(f"Model parameter names: {list(model_params.keys())}")
print(f"Weight shape: {model_params['weight'].shape}")
print(f"Bias shape: {model_params['bias'].shape}")

optim_params = optim_state["parameters"]
print(f"Optimizer parameter count: {len(optim_params)}")
print(f"Optimizer state: {optim_state['state']}")

original_weight = model._parameters["weight"].data
loaded_weight = model_params["weight"].data
weight_match = np.allclose(original_weight, loaded_weight)
print(f"Weight data round-trip match: {weight_match}")

original_bias = model._parameters["bias"].data
loaded_bias = model_params["bias"].data
bias_match = np.allclose(original_bias, loaded_bias)
print(f"Bias data round-trip match: {bias_match}")

model2 = nn.Linear(
    in_features=1,
    out_features=1,
    bias=True,
    device=Device("CPU"),
)
model2.register_state_dict(model_state)

restored_weight_match = np.allclose(model2._parameters["weight"].data, original_weight)
restored_bias_match = np.allclose(model2._parameters["bias"].data, original_bias)
print(f"Restored model weight match: {restored_weight_match}")
print(f"Restored model bias match: {restored_bias_match}")

print("\nAll loading tests passed.")

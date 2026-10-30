from krnl import save, nn, Device

model = nn.Linear(
        in_features=2500,
        out_features=2500,
        bias=True,
        device=Device("CPU"),
    )

optimizer = nn.optim.SGD(
    model._parameters.values()
)

save(model.get_model_state_dict(), "model")
save(optimizer.get_state_dict(), "optim")

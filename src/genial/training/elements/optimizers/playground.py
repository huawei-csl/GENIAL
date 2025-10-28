import torch
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path

from genial.training.elements.lit.models import AbstractLitModule


# Define the model
class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class LitDummyModel(AbstractLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = kwargs.get("meta_model")
        self.all_train_losses = []
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.training_step_outputs = []
        x, y = batch
        y_hat = self.model(x)
        # Simulate a plateau in loss after certain epochs to trigger ReduceLROnPlateau
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("loss/train_loss", loss, on_epoch=True)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.all_train_losses.append(torch.stack(self.training_step_outputs).mean())


# Playground Configuration
# The keys are the same as in AbstractLitModule, the values is the name of the lr scalar logged by tensorboard
scheduler_name_logger_map = {
    "cyclic_lr": "lr/cyclic",
    "warmup_constant_plateau_lr": "lr/wup_c_rop",
    "warmup_constant_lindec_lr": "lr/wup_c_lindec",
}
scheduler_type = "warmup_constant_lindec_lr"

# Create dummy data
x = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

# Instantiate the model
model_config_dict = {
    "max_scratch_lr": 1e-3,
    "lr_scheduler_type": scheduler_type,
}
model = LitDummyModel(meta_model=DummyModel(), model_config=model_config_dict)

# Prepare to log learning rates
lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

# Set up the trainer with total_epochs
trainer = L.Trainer(
    max_epochs=100,  # Use total_epochs variable
    callbacks=[lr_monitor],
    logger=True,  # Disable logging for simplicity
    enable_checkpointing=False,
    devices=[0],
)

# Train the model
trainer.fit(model, dataloader)

ax0: Axes
ax1: Axes
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6))

lrs = lr_monitor.lrs[scheduler_name_logger_map[scheduler_type]]
ax0.plot(range(len(lrs)), lrs)
# ax0.plot(range(len(lr_monitor.lrs["warmup_constant_scheduler"])), lr_monitor.lrs["warmup_constant_scheduler"])
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Learning Rate")
plt.grid(True)

ax1.plot(range(len(model.all_train_losses)), torch.stack(model.all_train_losses).cpu().detach())
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
plt.grid(True)
outdir_path = Path("debug")
if not outdir_path.exists():
    outdir_path.mkdirs()

filepath = Path("debug") / "lr_scheduler.png"
plt.suptitle(f"Learning Rate Schedule with {scheduler_type}")
plt.savefig(filepath)
plt.close()
print(f"Figure saved in {filepath}")

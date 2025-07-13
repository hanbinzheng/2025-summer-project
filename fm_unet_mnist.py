import torch
from datasets import get_mnist_dataloader
from paths import GaussianConditionalPath, LinearAlpha, LinearBeta
from models import UNetVelocity
from trainers import CFGTrainerFM
from utils import generate_samples_and_save
import matplotlib.pyplot as plt
from vector_fields import CFGVectorField
from simulators import EulerSimulator
from torchvision.utils import make_grid
from matplotlib.axes._axes import Axes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataloader, _ = get_mnist_dataloader()
print("dataloader is done")

path = GaussianConditionalPath(
    p_init_shape = (1, 32, 32),
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

unet = UNetVelocity(
    channels = [1, 32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 64,
    y_embed_dim = 64,
    num_categories = 10
).to(device)

trainer = CFGTrainerFM(
    path = path,
    model = unet,
    eta = 0.1
)

trainer.train(
    num_epochs = 50,
    device = device,
    dataloader = dataloader,
    project_name = "mnist"
)

generate_samples_and_save(
    model = unet,
    path = path,
    device = device,
    save_path = "results/mnist/test.png",
    labels = [i for i in range(1, 11)],
    samples_per_label = 8,
    simulator_type = "heun"
)

## Simple test
samples_per_class = 10
num_timesteps = 100
guidance_scales = [1.0, 3.0, 5.0]

# Graph
fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

for idx, w in enumerate(guidance_scales):
    # Setup ode and simulator
    vector_field = CFGVectorField(unet, guidance_scale=w)
    simulator = EulerSimulator(vector_field)

    # Sample initial conditions
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
    num_samples = y.shape[0]
    x0 = path.p_init.sample(num_samples) # (num_samples, 1, 32, 32)

    # Simulate
    ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
    x1 = simulator.simulate(x0, ts, y=y)

    # Plot
    grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
    axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    axes[idx].axis("off")
    axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
plt.savefig("results/mnist/new_visualize.png", dpi=300)

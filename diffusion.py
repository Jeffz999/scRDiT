import torch
from tqdm import tqdm
from unet import Unet1d
import logging
from diffusers import DPMSolverMultistepScheduler

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DiffusionGene:
    def __init__(self, gene_size=2000, device="cuda", shift=1.0):
        self.gene_size = gene_size
        self.device = device

        # --- NEW: Initialize a Hugging Face Diffusers Scheduler ---
        # We replace the manual beta schedule with a modern scheduler.
        # DPMSolverMultistepScheduler is a great choice for speed and quality.
        self.scheduler = DPMSolverMultistepScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            num_train_timesteps=1000,
            # For this model, which predicts noise (epsilon), this setting is standard.
            prediction_type="epsilon",
            # This corresponds to the "sigma_shift" in SD3's configs
            trained_betas=None,
        )

    def noise_genes(self, x, t):
        """Add noise to the genes using the scheduler's method."""
        # x is the original data (genes)
        # t is the tensor of timesteps
        noise = torch.randn_like(x)
        noisy_x = self.scheduler.add_noise(x, noise, t)
        return noisy_x, noise

    def sample_timesteps(self, n):
        """Generate random timesteps for training."""
        return torch.randint(low=0, high=self.scheduler.config.num_train_timesteps, size=(n,), device=self.device)

    def sample(self, model, n: int, num_inference_steps: int = 25):
        """
        --- NEW: Modern sampling method using the diffusers scheduler ---
        This method replaces the old `sample` and `sample_ddim` methods.

        Args:
            model: The trained noise prediction model (Unet1d or DiT).
            n: The number of samples to generate (batch size).
            num_inference_steps: How many steps to run the reverse diffusion.
                                 Fewer steps are much faster. (e.g., 20-50).
        """
        logging.info(f"Sampling {n} new genes with DPMSolverMultistepScheduler...")
        model.eval()

        # Set the number of inference steps. This is a key parameter for speed vs. quality.
        self.scheduler.set_timesteps(num_inference_steps)

        # 1. Start with random noise
        # The shape needs to match what the model expects: (batch, channels, sequence_length)
        x = torch.randn((n, 1, self.gene_size), device=self.device)

        # The scheduler needs to scale the initial noise.
        x *= self.scheduler.init_noise_sigma

        with torch.no_grad():
            # 2. Denoising loop
            for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
                # --- FIX START ---
                # The model (both DiT and Unet) expects a batch of timesteps (N,),
                # but the scheduler's loop provides a single scalar timestep.
                # We need to expand the scalar `t` to a tensor of shape (N,)
                # to match the batch size of `x`.
                timestep_batch = torch.full((n,), t, device=self.device, dtype=torch.long)

                # Predict the noise for the current noisy sample, using the batched timestep.
                predicted_noise = model(x, timestep_batch)
                # --- FIX END ---

                # 3. Use the scheduler's `step` method to compute the previous sample.
                # The scheduler's step function itself expects the scalar timestep `t`.
                x = self.scheduler.step(predicted_noise, t, x).prev_sample

        model.train()
        return x.cpu()

if __name__ == '__main__':
    # Test code.
    diffusion = DiffusionGene()
    # This will now work with either Unet1d or DiT
    from transformer import DiT
    model = DiT(depth=3, patch_size=10).to('cuda')
    # model = Unet1d().to('cuda')

    logging.info("Testing corrected sample method...")
    X = diffusion.sample(model, n=4, num_inference_steps=10)
    X = X.to('cpu')
    print("Sampled shape:", X.shape)
    logging.info("Test complete. If no errors, the fix is working.")


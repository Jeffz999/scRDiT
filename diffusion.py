import torch
from tqdm import tqdm
from unet import Unet1d
import logging
from diffusers import DPMSolverMultistepScheduler

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DiffusionGene:
    def __init__(self, gene_size=2000, device="cuda"):
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
            prediction_type="epsilon"
        )


    def noise_genes(self, x, t):
        """add noise"""
        # x is the original data (genes)
        # t is the tensor of timesteps
        noise = torch.randn_like(x)
        noisy_x = self.scheduler.add_noise(x, noise, t)
        return noisy_x, noise

    def sample_timesteps(self, n):
        """
        generate timesteps
        :param n:
        :return:
        """
        return torch.randint(low=1, high=self.scheduler.config.num_train_timesteps, size=(n,))

    def sample(self, model, n: int, num_inference_steps: int = 25):
        """
        --- NEW: Modern sampling method using the diffusers scheduler ---
        This method replaces the old `sample` and `sample_ddim` methods.

        Args:
            model: The trained noise prediction model (Unet1d or DiT).
            n: The number of samples to generate.
            num_inference_steps: How many steps to run the reverse diffusion.
                                 Fewer steps are much faster. (e.g., 20-50).
        """
        logging.info(f"Sampling {n} new genes with DPMSolverMultistepScheduler...")
        model.eval()

        # Set the number of inference steps. This is a key parameter for speed vs. quality.
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 1. Start with random noise
        x = torch.randn((n, 1, self.gene_size)).to(self.device)
        # The scheduler needs to scale the initial noise.
        x *= self.scheduler.init_noise_sigma

        with torch.no_grad():
            # 2. Denoising loop
            for t in tqdm(timesteps):
                # The model input needs to be on the correct device
                t_tensor = torch.tensor([t]).to(self.device)

                # Predict the noise for the current noisy sample
                predicted_noise = model(x, t_tensor)

                # 3. Use the scheduler's `step` method to compute the previous sample
                # This one line replaces the complex math of the old samplers.
                x = self.scheduler.step(predicted_noise, t, x).prev_sample

        model.train()
        return x

if __name__ == '__main__':
    # Test code.
    diffusion = DiffusionGene()
    model = Unet1d()
    model.to('cuda')
    X = diffusion.sample(model, 1)
    X = X.to('cpu')
    print(X.shape)


import dearpygui.dearpygui as dpg
from dpg_system.conversion_utils import *
from dpg_system.node import Node
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os


def register_vae_nodes():
    Node.app.register_node("vae", VAENode.factory)
    # Node.app.register_node("smpl_pose_to_joints", SMPLPoseToJointsNode.factory)
    # Node.app.register_node("smpl_body", SMPLBodyNode.factory)


class VAENode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = VAENode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.input_dim = 784
        self.hidden_dim = 512
        self.latent_dim = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(args) > 0:
            if is_number(args[0]):
                self.input_dim = any_to_int(args[0])

        if len(args) > 1:
            if is_number(args[1]):
                self.hidden_dim = any_to_int(args[1])

        if len(args) > 2:
            if is_number(args[2]):
                self.latent_dim = any_to_int(args[2])

        self.input_in = self.add_input('input in', triggers_execution=True)
        self.forward_in = self.add_input('forward in', triggers_execution=True)
        self.latent_in = self.add_input('latent in', triggers_execution=True)

        self.input_dim_label = self.add_label('input dim: ' + str(self.input_dim))
        self.hidden_dim_label = self.add_label('hidden dim: ' + str(self.hidden_dim))
        self.latent_dim_label = self.add_label('latent dim: ' + str(self.latent_dim))
        self.model_path_in = self.add_input('model path', callback=self.load_model)
        self.distribution_out = self.add_output('distribution out')
        self.latents_out = self.add_output('latents out')
        self.decoded_out = self.add_output('decoded out')

        self.dataloader = None
        self.optimizer = None

        self.model = VAE(self.input_dim, self.hidden_dim, self.latent_dim)

    def load_model(self):
        path = self.model_path_in()
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model = self.model.to(self.device)

    def execute(self):
        if self.active_input == self.latent_in:
            self.decode_input()
        elif self.active_input == self.input_in:
            self.process()
        elif self.active_input == self.forward_in:
            self.forward()

    def forward(self):
        # self.model.eval()

        with torch.no_grad():
            data = self.forward_in()
            t = type(data)
            if t == torch.Tensor:
                data = data.to(device=self.device, dtype=torch.float32)
            elif t == np.ndarray:
                data = torch.from_numpy(data).to(device=self.device, dtype=torch.float32)
            else:
                data = any_to_tensor(data, self.device, torch.float32)

            try:
                data = data.view(-1, self.input_dim)
                results = self.model(data)
                self.latents_out.send(results.z_sample)
                self.decoded_out.send(results.x_recon)
            except Exception as e:
                print(e)

    def decode_input(self):
        self.model.eval()

        with torch.no_grad():
            data = self.latent_in()
            t = type(data)
            if t == torch.Tensor:
                data = data.to(device=self.device, dtype=torch.float32)
            elif t == np.ndarray:
                data = torch.from_numpy(data).to(device=self.device, dtype=torch.float32)
            else:
                data = any_to_tensor(data, self.device, torch.float32)

            try:
                data = data.view(-1, self.latent_dim)
                output = self.model.decode(data)
                self.latents_out.send(data)
                self.decoded_out.send(output)
            except Exception as e:
                print(e)
            # data should be in shape(<batch>, <input_dim>)

    def process(self):
        self.model.eval()

        with torch.no_grad():
            data = self.input_in()
            t = type(data)
            if t == torch.Tensor:
                data = data.to(device=self.device, dtype=torch.float32)
            elif t == np.ndarray:
                data = torch.from_numpy(data).to(device=self.device, dtype=torch.float32)
            else:
                data = any_to_tensor(data, self.device, torch.float32)

            try:
                data = data.view(-1, self.input_dim)
                latent_distribution = self.model.encode(data)
                sampled_latent = latent_distribution.loc[0]
                # sampled_latent = self.model.reparameterize(latent_distribution)
                reconstruction = self.model.decode(sampled_latent)
                self.distribution_out.send(latent_distribution.loc[0])
                self.latents_out.send(sampled_latent)
                self.decoded_out.send(reconstruction)
            except Exception as e:
                print(e)

    # def prepare_training(self):
    #     self.previous_updates = 0
    #     self.epoch = 0
    #
    # def training_epoch(self):
    #     self.train_output.send(['epoch', (self.epoch + 1) / self.num_epochs])
    #     self.previous_updates = self.train(self.previous_updates)
    #     self.test()
    #
    # def train(self, prev_updates):
    #     """
    #     Trains the model on the given data.
    #
    #     Args:
    #         model (nn.Module): The model to train.
    #         dataloader (torch.utils.data.DataLoader): The data loader.
    #         loss_fn: The loss function.
    #         optimizer: The optimizer.
    #     """
    #     self.model.train()  # Set the model to training mode
    #
    #     # for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
    #     for batch_idx, (data, target) in enumerate(self.dataloader):
    #         n_upd = prev_updates + batch_idx
    #
    #         data = data.to(self.device)
    #
    #         self.optimizer.zero_grad()  # Zero the gradients
    #
    #         output = self.model(data)  # Forward pass
    #         loss = output.loss
    #
    #         loss.backward()
    #
    #         if n_upd % 100 == 0:
    #             # Calculate and log gradient norms
    #             total_norm = 0.0
    #             for p in self.model.parameters():
    #                 if p.grad is not None:
    #                     param_norm = p.grad.data.norm(2)
    #                     total_norm += param_norm.item() ** 2
    #
    #             self.train_output.send(['error', loss.item(), output.loss_recon.item(), output.loss_kl.item()])
    #
    #         # gradient clipping
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #
    #         self.optimizer.step()  # Update the model parameters
    #
    #     return prev_updates + len(self.dataloader)
    #
    # def test(self):
    #     """
    #     Tests the model on the given data.
    #
    #     Args:
    #         model (nn.Module): The model to test.
    #         dataloader (torch.utils.data.DataLoader): The data loader.
    #         cur_step (int): The current step.
    #         writer: The TensorBoard writer.
    #     """
    #     self.model.eval()  # Set the model to evaluation mode
    #     test_loss = 0
    #     test_recon_loss = 0
    #     test_kl_loss = 0
    #
    #     with torch.no_grad():
    #         for data, target in self.dataloader:
    #             data = data.to(self.device)
    #             data = data.view(data.size(0), -1)  # Flatten the data
    #
    #             output = self.model(data, compute_loss=True)  # Forward pass
    #
    #             test_loss += output.loss.item()
    #             test_recon_loss += output.loss_recon.item()
    #             test_kl_loss += output.loss_kl.item()
    #
    #     test_loss /= len(self.dataloader)
    #     test_recon_loss /= len(self.dataloader)
    #     test_kl_loss /= len(self.dataloader)
    #
    #     self.train_output.send(['test error', test_loss, output.loss_recon.item(), output.loss_kl.item()])
    #
    #     # print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    #
    #     # if writer is not None:
    #     #     writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
    #     #     writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
    #     #     writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)
    #     #
    #     #     # Log reconstructions
    #     #     writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
    #     #     writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)
    #     #
    #     #     # Log random samples from the latent space
    #     #     z = torch.randn(16, latent_dim).to(device)
    #     #     samples = model.decode(z)
    #     #     writer.add_images('Test/Samples',


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim),  # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # compute loss terms
        loss_recon = F.binary_cross_entropy(recon_x, x + 0.5, reduction='none').sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )

@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.

    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor

    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


def train(model, dataloader, optimizer, prev_updates, writer=None):
    """
    Trains the model on the given data.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode

    # for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
    for batch_idx, (data, target) in enumerate(dataloader):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        output = model(data)  # Forward pass
        loss = output.loss

        loss.backward()

        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            print(
                f'Step {n_upd:,} (N samples: {n_upd * batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()  # Update the model parameters

    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, writer=None):
    """
    Tests the model on the given data.

    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data

            output = model(data, compute_loss=True)  # Forward pass

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()

    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')

    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)

        # Log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)

        # Log random samples from the latent space
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)

import dearpygui.dearpygui as dpg
from kornia.utils import map_location_to_cpu

from dpg_system.conversion_utils import *
from dpg_system.node import Node
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import glob
import os
import os.path as osp
from omegaconf import OmegaConf
import platform

def register_vae_nodes():
    Node.app.register_node("vae", VAENode.factory)
    Node.app.register_node('vposer', VPoserNode.factory)

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


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * torch.logical_not(mask_d0_d1)
    mask_c2 = torch.logical_not(mask_d2) * mask_d0_nd1
    mask_c3 = torch.logical_not(mask_d2) * torch.logical_not(mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def quaternion_to_angle_axis(quaternion) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    bs = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot, [0, 1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot)
    return pose


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))

class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)

def exprdir2model(expr_dir, model_cfg_override: dict = None):
    if not os.path.exists(expr_dir): raise ValueError(f'Could not find the experiment directory: {expr_dir}')

    model_snapshots_dir = osp.join(expr_dir, 'snapshots')
    available_ckpts = sorted(glob.glob(osp.join(model_snapshots_dir, '*.ckpt')), key=osp.getmtime)
    assert len(available_ckpts) > 0, ValueError('No checkpoint found at {}'.format(model_snapshots_dir))
    trained_weights_fname = available_ckpts[-1]

    slash = '/'
    if platform.system() == 'Windows':
        slash = '\\'

    model_cfg_fname = glob.glob(osp.join(slash, slash.join(trained_weights_fname.split(slash)[:-2]), '*.yaml'))
    if len(model_cfg_fname) == 0:
        model_cfg_fname = glob.glob(osp.join(slash.join(trained_weights_fname.split(slash)[:-2]), '*.yaml'))

    model_cfg_fname = model_cfg_fname[0]
    model_cfg = OmegaConf.load(model_cfg_fname)
    if model_cfg_override:
        override_cfg_dotlist = [f'{k}={v}' for k, v in model_cfg_override.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)
        model_cfg = OmegaConf.merge(model_cfg, override_cfg)

    return model_cfg, trained_weights_fname

def load_model(expr_dir, model_code=None,
               remove_words_in_model_weights: str = None,
               load_only_cfg: bool = False,
               disable_grad: bool = True,
               model_cfg_override: dict = None,
               comp_device='gpu'):
    """

    :param expr_dir:
    :param model_code: an imported module
    from supercap.train.supercap_smpl import SuperCap, then pass SuperCap to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    """
    import torch

    model_cfg, trained_weights_fname = exprdir2model(expr_dir, model_cfg_override=model_cfg_override)

    if load_only_cfg: return model_cfg

    assert model_code is not None, ValueError('model_code should be provided')
    model_instance = model_code(model_cfg)
    if disable_grad:  # i had to do this. torch.no_grad() couldnt achieve what i was looking for
        for param in model_instance.parameters():
            param.requires_grad = False

    if comp_device == 'cpu' or not torch.cuda.is_available():
        print('No GPU detected. Loading on CPU!')
        state_dict = torch.load(trained_weights_fname, map_location=torch.device('cpu'))['state_dict']
    else:
        state_dict = torch.load(trained_weights_fname)['state_dict']
    if remove_words_in_model_weights is not None:
        words = '{}'.format(remove_words_in_model_weights)
        state_dict = {k.replace(words, '') if k.startswith(words) else k: v for k, v in state_dict.items()}

    ## keys that were in the model trained file and not in the current model
    instance_model_keys = list(model_instance.state_dict().keys())
    # trained_model_keys = list(state_dict.keys())
    # wts_in_model_not_in_file = set(instance_model_keys).difference(set(trained_model_keys))
    ## keys that are in the current model not in the training weights
    # wts_in_file_not_in_model = set(trained_model_keys).difference(set(instance_model_keys))
    # assert len(wts_in_model_not_in_file) == 0, ValueError('Some model weights are not present in the pretrained file. {}'.format(wts_in_model_not_in_file))

    state_dict = {k: v for k, v in state_dict.items() if k in instance_model_keys}
    model_instance.load_state_dict(state_dict, strict=False)
    # Todo fix the issues so that we can set the strict to true. The body model uses unnecessary registered buffers
    model_instance.eval()
    print(f'Loaded model in eval mode with trained weights: {trained_weights_fname}')
    return model_instance, model_cfg

class VPoserNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = VPoserNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.num_neurons = 512
        self.latent_dim = 32
        self.input_dim = 63

        if len(args) > 1:
            self.num_neurons = any_to_int(args[0])
            self.latent_dim = any_to_int(args[1])

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_built():
            self.device = 'mps'
        self.input_in = self.add_input('input in', triggers_execution=True)
        # self.forward_in = self.add_input('forward in', triggers_execution=True)
        self.latent_in = self.add_input('latent in', triggers_execution=True)

        self.model_path_in = self.add_input('model path', callback=self.load_model)
        self.latents_out = self.add_output('latents out')
        self.decoded_out = self.add_output('decoded out')

        self.dataloader = None
        self.optimizer = None
        config = vposer_config(self.num_neurons, self.latent_dim)
        self.model = VPoser(config)

    def load_model(self):
        path = any_to_string(self.model_path_in())
        if os.path.exists(path):
            self.model = load_model(path, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)[0]
            # self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model = self.model.to(self.device)

    def execute(self):
        if self.active_input == self.latent_in:
            self.decode_input()
        elif self.active_input == self.input_in:
            self.forward()

    def forward(self):
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
                results = self.model(data)
                self.latents_out.send(results['q_z_sample'])
                self.decoded_out.send(results['pose_body'])
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
                self.decoded_out.send(output['pose_body'])
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
                data = data.view(-1, 1, 21, 3)
                decoded_results = self.model.forward(data)
                pose = decoded_results['pose_body']
                sampled_latent = decoded_results['q_z_sample']
                # sampled_latent = latent_distribution.loc[0]
                # # sampled_latent = self.model.reparameterize(latent_distribution)
                # reconstruction = self.model.decode(sampled_latent)
                # self.distribution_out.send(latent_distribution.loc[0])
                self.latents_out.send(sampled_latent)
                self.decoded_out.send(pose)
            except Exception as e:
                print(e)

class vposer_config():
    def __init__(self, num_neurons, latentD):
        class model_params():
            def __init__(self, num_neurons, latentD):
                self.num_neurons = num_neurons
                self.latentD = latentD

        self.model_params = model_params(num_neurons, latentD)


class VPoser(nn.Module):
    def __init__(self, model_ps):
        super(VPoser, self).__init__()
        num_neurons, self.latentD = model_ps.model_params.num_neurons, model_ps.model_params.latentD

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_built():
            self.device = torch.device('mps')
        self.num_joints = 21
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        ).to(device=self.device)

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        ).to(device=self.device)

    def encode(self, pose_body):
        '''
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]

        prec = self.decoder_net(Zin)

        return {
            'pose_body': matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }

    def forward(self, pose_body):
        '''
        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        with torch.no_grad():
            q_z = self.encode(pose_body)
            q_z_sample = q_z.mean  # q_z.rsample()  #  #
            decode_results = self.decode(q_z_sample)
            decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z, 'q_z_sample': q_z_sample})
        return decode_results

    def sample_poses(self, num_poses, seed=None):
        np.random.seed(seed)

        some_weight = [a for a in self.parameters()][0]
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype, device=device)

        return self.decode(Zgen)


#
#
# def train(model, dataloader, optimizer, prev_updates, writer=None):
#     """
#     Trains the model on the given data.
#
#     Args:
#         model (nn.Module): The model to train.
#         dataloader (torch.utils.data.DataLoader): The data loader.
#         loss_fn: The loss function.
#         optimizer: The optimizer.
#     """
#     model.train()  # Set the model to training mode
#
#     # for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
#     for batch_idx, (data, target) in enumerate(dataloader):
#         n_upd = prev_updates + batch_idx
#
#         data = data.to(device)
#
#         optimizer.zero_grad()  # Zero the gradients
#
#         output = model(data)  # Forward pass
#         loss = output.loss
#
#         loss.backward()
#
#         if n_upd % 100 == 0:
#             # Calculate and log gradient norms
#             total_norm = 0.0
#             for p in model.parameters():
#                 if p.grad is not None:
#                     param_norm = p.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
#             total_norm = total_norm ** (1. / 2)
#
#             print(
#                 f'Step {n_upd:,} (N samples: {n_upd * batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')
#
#             if writer is not None:
#                 global_step = n_upd
#                 writer.add_scalar('Loss/Train', loss.item(), global_step)
#                 writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
#                 writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
#                 writer.add_scalar('GradNorm/Train', total_norm, global_step)
#
#         # gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#         optimizer.step()  # Update the model parameters
#
#     return prev_updates + len(dataloader)
#
#
# def test(model, dataloader, cur_step, writer=None):
#     """
#     Tests the model on the given data.
#
#     Args:
#         model (nn.Module): The model to test.
#         dataloader (torch.utils.data.DataLoader): The data loader.
#         cur_step (int): The current step.
#         writer: The TensorBoard writer.
#     """
#     model.eval()  # Set the model to evaluation mode
#     test_loss = 0
#     test_recon_loss = 0
#     test_kl_loss = 0
#
#     with torch.no_grad():
#         for data, target in dataloader:
#             data = data.to(device)
#             data = data.view(data.size(0), -1)  # Flatten the data
#
#             output = model(data, compute_loss=True)  # Forward pass
#
#             test_loss += output.loss.item()
#             test_recon_loss += output.loss_recon.item()
#             test_kl_loss += output.loss_kl.item()
#
#     test_loss /= len(dataloader)
#     test_recon_loss /= len(dataloader)
#     test_kl_loss /= len(dataloader)
#     print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
#
#     if writer is not None:
#         writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
#         writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
#         writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)
#
#         # Log reconstructions
#         writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
#         writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)
#
#         # Log random samples from the latent space
#         z = torch.randn(16, latent_dim).to(device)
#         samples = model.decode(z)
#         writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)
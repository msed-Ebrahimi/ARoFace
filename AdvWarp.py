import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def adversarial_img_warping(backbone, img, module_partial_fc, local_labels,eta_scale=0.1,eta_theta=0.1,eta_t=0.1,ratio=0.75,k=1):
    """
        Performs adversarial image warping to create perturbed images for training.

        Args:
            backbone: Neural network backbone for feature extraction.
            img: Input image tensor.
            module_partial_fc: Partial fully connected layer module.
            local_labels: Labels corresponding to the input images.
            eta_scale: Learning rate for the scale parameter.
            eta_theta: Learning rate for the rotation angle.
            eta_t: Learning rate for the translation parameters.
            ratio: Ratio of real to synthetic images in the mini-batch.
            k: Number of iterations for updating adversarial parameters.

        Returns:
            train_img: Concatenated tensor of original and adversarially warped images.
            train_labels: Concatenated labels corresponding to train_img.
        """
    batch_size = img.shape[0]
    eta_scale = np.random.normal(eta_scale, 0.001)
    eta_theta = np.random.normal(eta_theta, 0.001)
    eta_t  = np.random.normal(eta_t, 0.001)

    # Sample random values for scale, rotation angle (theta), and translation (t)
    sampled_scale = Variable(torch.FloatTensor(np.random.normal(1, 0.01, (batch_size, 1))))
    sampled_theta = Variable(torch.FloatTensor(np.random.normal(0, 0.01, (batch_size, 1))))
    sampled_t = Variable(torch.FloatTensor(np.random.normal(0, 0.01, (batch_size, 2))))
    sampled_scale = sampled_scale.cuda(non_blocking=True)
    sampled_theta = sampled_theta.cuda(non_blocking=True)
    sampled_t = sampled_t.cuda(non_blocking=True)

    # Detach to prevent gradient tracking
    scale = sampled_scale.detach()
    theta = sampled_theta.detach()
    t = sampled_t.detach()

    for it in range(k):
        # Enable gradient tracking for adversarial parameters
        scale.requires_grad_()
        theta.requires_grad_()
        t.requires_grad_()

        # Compute transformation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.stack([torch.mul(scale , cos_theta), torch.mul(-scale , sin_theta), torch.mul(scale , sin_theta), torch.mul(scale, cos_theta)],dim=-1).view(batch_size, 2, 2)
        transformation_matrices = torch.cat([rotation_matrix, t.view(batch_size, 2, 1)], dim=-1)
        grid = F.affine_grid(transformation_matrices, img.size())
        updated_img = F.grid_sample(pad_with_value(img), grid, padding_mode='border')

        # Compute gradients and update adversarial parameters
        with torch.enable_grad():
            local_embeddings = backbone(updated_img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            grad_scale, grad_theta, grad_t = torch.autograd.grad(loss, [scale, theta, t])
            scale = scale.detach() + eta_scale * torch.sign(grad_scale.detach())
            theta = theta.detach() + eta_theta * torch.sign(grad_theta.detach())
            t = t.detach() + eta_t * torch.sign(grad_t.detach())

            # Clipping the adversarial parameters
            scale = torch.min(torch.max(scale, sampled_scale-0.5), sampled_scale+1.5)
            theta = torch.min(torch.max(theta, sampled_theta-0.8), sampled_theta+0.8)
            t = torch.min(torch.max(t, sampled_t-0.3), sampled_theta+0.3)

    # Compute final transformation and warp the image
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation_matrix = torch.stack([scale * cos_theta, -scale * sin_theta, scale * sin_theta, scale * cos_theta],dim=-1).view(batch_size, 2, 2)
    transformation_matrices = torch.cat([rotation_matrix, t.view(batch_size, 2, 1)], dim=-1)
    grid = F.affine_grid(transformation_matrices, img.size())
    updated_img = F.grid_sample(pad_with_value(img), grid,padding_mode='border')

    # Create a mini-batch with a mix of real and synthetic images
    batch_size = img.shape[0]
    idx1 = torch.randperm(batch_size)
    idx1 = idx1[:int(batch_size * ratio)]
    idx2 = torch.randperm(batch_size)
    idx2 = idx2[:int(batch_size * (1 - ratio))]

    train_img = torch.cat((img[idx1], updated_img[idx2]), dim=0)
    train_labels = torch.cat((local_labels[idx1], local_labels[idx2]), dim=0)

    return train_img, train_labels

def pad_with_value(img, pad_size=1, pad_value=-1):
    """
    Pad the image on all sides with specified size and value.

    Args:
        img: Input image tensor.
        pad_size: Size of padding to be applied on all sides.
        pad_value: Value to be used for padding.

    Returns:
        padded_img: Padded image tensor.
    """
    pad = (pad_size, pad_size, pad_size, pad_size)
    padded_img = F.pad(img, pad, "constant", pad_value)
    return padded_img
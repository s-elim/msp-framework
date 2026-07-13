import torch

def quat_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternions to rotation matrices.
    
    Mathematical details:
    Maps a quaternion q = (qw, qx, qy, qz) to a 3x3 rotation matrix using
    standard rigid-body kinematic formulations.

    Args:
        quaternions (torch.Tensor): Tensor of shape (..., 4) where the last 
                                    dimension is (w, x, y, z).
    Returns:
        torch.Tensor: Rotation matrices of shape (..., 3, 3).
    """
    # Normalize quaternion to avoid scaling artifacts
    q = quaternions / quaternions.norm(dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.view(quaternions.shape[:-1] + (3, 3))


def generate_random_se3(batch_size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generates random SE(3) transforms represented as 4x4 homogenous matrices.
    
    Used primarily for robust action sampling and augmentation.
    
    Args:
        batch_size (int): Number of transforms to generate.
        device (torch.device): Device to place the tensor on.
        
    Returns:
        torch.Tensor: Shape (batch_size, 4, 4)
    """
    # Random translation in a 1x1x1 meter cube
    translation = torch.rand(batch_size, 3, device=device) - 0.5
    
    # Random quaternion (requires normalization for uniform sampling)
    quat = torch.randn(batch_size, 4, device=device)
    rotation_matrices = quat_to_matrix(quat)
    
    transforms = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    transforms[:, :3, :3] = rotation_matrices
    transforms[:, :3, 3] = translation
    return transforms

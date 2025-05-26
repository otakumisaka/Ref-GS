import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def refract(x: torch.Tensor, n: torch.Tensor, ior: float) -> torch.Tensor:
    # here the ior is set to be n1/n2
    cos_theta = torch.clamp(torch.sum(x * n, dim=-1, keepdim=True), -1.0, 1.0)
    sin2_theta = 1.0 - cos_theta**2
    eta = ior
    k = 1.0 - eta**2 * sin2_theta

    # Check if total internal reflection occurs
    total_reflection = sin2_theta > 1.0

    # Compute refracted direction
    refracted = eta * x - (eta * cos_theta + torch.sqrt(torch.clamp(k, min=0.0))) * n

    # Zero if total internal reflection
    refracted = torch.where(total_reflection, torch.zeros_like(refracted), refracted)

    return refracted, total_reflection

def fresnel_schlick(cos_theta, f0):
    """
    Fresnel reflection coefficient using Schlick's approximation
    """
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)
        
def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def linear2srgb(tensor_0to1):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = torch.clamp(tensor_0to1, min=1e-10, max=1.)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (torch.pow(tensor_0to1, 1/srgb_exponent)) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)
    return tensor_srgb

def srgb2linear(tensor_0to1):
    srgb_linear_thres = 0.04045
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = torch.clamp(tensor_0to1, min=1e-10, max=1.)

    tensor_linear = tensor_0to1 / srgb_linear_coeff
    tensor_nonlinear = torch.pow(tensor_0to1 + (srgb_exponential_coeff - 1) / srgb_exponential_coeff, srgb_exponent)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)
    return tensor_srgb
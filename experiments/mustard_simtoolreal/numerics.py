"""Numerical diagnostics for the deliberately small warm-start action variance."""
import torch


def gaussian_policy_kl(mu0,sigma0,mu1,sigma1,reduce=True):
    """KL(N0 || N1); FP32 prevents AMP cancellation at small standard deviation.

    A fixed epsilon added to variance makes the native diagnostic negative even
    for identical policies at sigma=.034. Clamp only at the FP32 positive floor.
    """
    mu0=mu0.float();mu1=mu1.float()
    floor=torch.finfo(torch.float32).tiny
    sigma0=sigma0.float().clamp_min(floor);sigma1=sigma1.float().clamp_min(floor)
    ratio=sigma0/sigma1
    result=(sigma1.log()-sigma0.log()+.5*(ratio.square()+((mu0-mu1)/sigma1).square()-1)).sum(-1).clamp_min(0)
    return result.mean() if reduce else result

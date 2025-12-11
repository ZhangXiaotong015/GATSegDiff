"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
from torch.autograd import Variable
import enum
import torch.nn.functional as F
# from torchvision.utils import save_image
import torch
import math
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
import torch as th
import torch.nn as nn
from guided_diffusion.train_util import visualize
from guided_diffusion.nn import mean_flat
from guided_diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
# from torchvision import transforms
from guided_diffusion.utils import staple, dice_score, norm
# import torchvision.utils as vutils
from guided_diffusion.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
import string
import random
from tqdm.auto import tqdm
# from guided_diffusion.metrics import SymmetricCrossEntropyLoss #DiceLoss, DC_and_BCE_loss, BoundaryLoss
import nibabel
# from calflops import calculate_flops

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)

def standardize(img):
    mean = th.mean(img)
    std = th.std(img)
    img = (img - mean) / std
    return img


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        dpm_solver,
        uncertainty_control,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.dpm_solver = dpm_solver
        self.uncertainty_control = uncertainty_control

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # parameters = {'gamma':0.2, 'root':"l2"}
        # self.calc_boundary = BoundaryLoss(parameters)
        # self.SCEloss = SymmetricCrossEntropyLoss(alpha=0.5, beta=0.5)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        C=1
        cal = None#0
        assert t.shape == (B,)

        'FLOPs calculation'
        # inputs = {}
        # inputs['x'] = x
        # inputs['timesteps'] = self._scale_timesteps(t)
        # inputs['nodes'] = model_kwargs['nodes']
        # inputs['edges'] = th.stack(model_kwargs['edges'], dim=0)

        # flops, macs, params = calculate_flops(model=model, 
        #                                     # input_shape=(50000,4),
        #                                     kwargs = inputs,
        #                                     output_as_string=True,
        #                                     output_precision=4)
        # raise ValueError("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if isinstance(model_output, tuple):
            model_output, cal = model_output

        x=x[:,-1:,...]  #loss is only calculated on the last channel, not on the input brain MR image

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'cal': cal,
        }


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:

            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, org, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        a, gradient = cond_fn(x, self._scale_timesteps(t),org,  **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return a, new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t,  model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])

        eps = eps.detach() - (1 - alpha_bar).sqrt() *p_mean_var["update"]*0

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x.detach(), t.detach(), eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, eps


    def sample_known(self, img, batch_size = 1):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop_known(model,(batch_size, channels, image_size, image_size), img)

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:, -1:,...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "cal": out["cal"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]


    def inverse_data_transform(self, config, X):
        if hasattr(config.toDict(), "image_mean"):
            X = X + config.image_mean.to(X.device)[None, ...]

        if config.data.logit_transform:
            X = torch.sigmoid(X)
        elif config.data.rescaled:
            X = (X + 1.0) / 2.0

        return torch.clamp(X, 0.0, 1.0)

    def compute_alpha(self, beta, t):
        # beta is the \beta in ddpm paper
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_from_gaussian(self, mean, var):
        samples = mean + (torch.randn(mean.shape).to(mean.device)) * torch.sqrt(var)    
        return samples

    def singlestep_ddpm_sample(self, diffusion,xt,seq,timestep,eps_t):
        #at.sqrt() is the \alpha_t in our paper
        n = xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(xt.device)
        next_t = (torch.ones(n)*seq[timestep-1]).to(xt.device)
        at = self.compute_alpha(torch.from_numpy(diffusion.betas).to(xt.device), t.long())
        at_minus_1 = self.compute_alpha(torch.from_numpy(diffusion.betas).to(xt.device), next_t.long())
        beta_t = 1 - at/at_minus_1
        
        mean = (1/(1-beta_t).sqrt())*(xt - beta_t * eps_t / ( 1 - at ).sqrt())
        
        noise = torch.randn_like(xt)
        logvar = beta_t.log()
        xt_next = mean + torch.exp(logvar * 0.5) * noise
        
        return xt_next

    def ddpm_exp_iteration(self, diffusion,exp_xt,seq,timestep,mc_eps_exp_t):
        # at here is the \bar{\alpha}_t in ddpm paper
        n = exp_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(exp_xt.device)
        next_t = (torch.ones(n)*seq[timestep-1]).to(exp_xt.device)
        at = self.compute_alpha(torch.from_numpy(diffusion.betas).to(exp_xt.device), t.long())
        at_minus_1 = self.compute_alpha(torch.from_numpy(diffusion.betas).to(exp_xt.device), t.long())
        beta_t = 1 - at / at_minus_1
        exp_eps_coefficient = -1 * beta_t / ((1 - beta_t) * (1 - at) ).sqrt()
        exp_xt_next = (1 / (1 - beta_t).sqrt() ) * exp_xt + exp_eps_coefficient * mc_eps_exp_t
        return exp_xt_next

    def ddpm_var_iteration(self, diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep):
        # at is the \bar{\alpha}_t in ddpm paper
        n = var_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
        next_t = (torch.ones(n)*seq[timestep-1]).to(var_xt.device)
        at = self.compute_alpha(torch.from_numpy(diffusion.betas).to(var_xt.device), t.long())
        at_minus_1 = self.compute_alpha(torch.from_numpy(diffusion.betas).to(var_xt.device), next_t.long())
        beta_t = 1 - at/at_minus_1
        cov_coefficient = (-2 * beta_t) / ( (1 - beta_t) * (1 - at).sqrt() )
        var_epst_coefficient = (beta_t ** 2) / ((1 - beta_t) * (1 - at))
        var_xt_next = (1 / (1 - beta_t).sqrt()) * var_xt + cov_coefficient * cov_xt_epst + var_epst_coefficient * var_epst + beta_t
        
        return var_xt_next

    def conditioned_exp_iteration(self, diffusion, exp_xt, seq, timestep, pre_wuq, mc_eps_exp_t=None, acc_eps_t = None):
        if pre_wuq == True:
            return self.ddpm_exp_iteration(diffusion, exp_xt, seq, timestep, mc_eps_exp_t)
        else:
            return self.ddpm_exp_iteration(diffusion, exp_xt, seq, timestep, acc_eps_t)

    def conditioned_var_iteration(self, diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep, pre_wuq):

        if pre_wuq == True:
            return self.ddpm_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep)
        else:
            # at is the \bar{\alpha}_t in ddpm paper
            n = var_xt.size(0)
            t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
            next_t = (torch.ones(n)*seq[timestep-1]).to(var_xt.device)
            at = self.compute_alpha(torch.from_numpy(diffusion.betas).to(var_xt.device), t.long())
            at_minus_1 = self.compute_alpha(torch.from_numpy(diffusion.betas).to(var_xt.device), next_t.long())
            beta_t = 1 - at/at_minus_1
            var_xt_next = (1 / (1 - beta_t).sqrt()) * var_xt + beta_t
            
            return var_xt_next

    def bayes_diff_sample(
        self,
        num_timesteps, 
        timesteps,
        skip_type,
        total_n_samples,
        sample_batch_size,
        cond_class,
        mc_size,
        model_kwargs,
        cond_img,
        fixed_xT,
        custom_model,
        config,
        diffusion,
        ):
        ##########   get t sequence (note that t is different from timestep)  ########## 

        if skip_type == "uniform":
            skip = num_timesteps // timesteps # 1000 // 250
            seq = range(0, num_timesteps, skip)
        elif skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError   
        #########   get skip UQ rules  ##########  
        # if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
        uq_array = [False] * (timesteps)
        for i in range(timesteps-1, 0, -5):
            uq_array[i] = True
        
        sample_x = []
        if total_n_samples % sample_batch_size != 0:
            raise ValueError("Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(total_n_samples, sample_batch_size))
        n_rounds = total_n_samples // sample_batch_size
        # var_sum = torch.zeros((sample_batch_size, n_rounds)).cuda()
        var_sum = torch.zeros((sample_batch_size, n_rounds, config.data.image_size, config.data.image_size)).cuda() # (num_slices,num_ensemble,256,256)

        with torch.no_grad():
            # for loop in tqdm.tqdm(
            #     range(n_rounds), 
            # ):
            for loop in range(n_rounds):
                print('round_'+str(loop))
                if cond_class:
                    if diffusion.args.fixed_class == 10000:
                        classes = torch.randint(low=0, high=diffusion.config.data.num_classes, size=(args.sample_batch_size,)).to(device)
                    else:
                        classes = torch.randint(low=diffusion.args.fixed_class, high=diffusion.args.fixed_class + 1, size=(args.sample_batch_size,)).to(device)
                else:
                    classes = None

                if classes is None:
                    # model_kwargs = {}
                    model_kwargs = model_kwargs
                else:
                    model_kwargs = {"y": classes}
        
                samle_batch_size = sample_batch_size

                # xT = fixed_xT[loop*sample_batch_size:(loop+1)*sample_batch_size, :, :, :]
                xT = fixed_xT.cuda()
                xT_in = torch.cat((cond_img, xT), dim=1)

                timestep, mc_sample_size = timesteps-1, mc_size
                T = (torch.ones(samle_batch_size) * seq[timestep]).to(xT.device)
                if uq_array[timestep] == True:
                    xt_next = xT
                    exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).cuda()
                    assert xT_in.shape[1] == 4, f"Expected xT_in.shape[1] to be 4, but got {xT_in.shape[1]}"
                    eps_mu_t_next, eps_var_t_next = custom_model(xT_in, T, config, **model_kwargs) 
                    cov_xt_next_epst_next = torch.zeros_like(xT).cuda()
                    list_eps_mu_t_next_i = torch.unsqueeze(eps_mu_t_next, dim=0)
                else:
                    xt_next = xT
                    exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).cuda()
                    assert xT.shape[1] == 4, f"Expected xT.shape[1] to be 4, but got {xT.shape[1]}"
                    eps_mu_t_next = custom_model.accurate_forward(xT_in, (torch.ones(samle_batch_size) * seq[timestep]).to(xT.device), config, **model_kwargs)
        
                for timestep in range(timesteps-1, 0, -1):

                    if uq_array[timestep] == True:
                        xt = xt_next
                        exp_xt, var_xt = exp_xt_next, var_xt_next
                        eps_mu_t, eps_var_t, cov_xt_epst = eps_mu_t_next, eps_var_t_next, cov_xt_next_epst_next
                        mc_eps_exp_t = torch.mean(list_eps_mu_t_next_i, dim=0)
                    else: 
                        xt = xt_next
                        exp_xt, var_xt = exp_xt_next, var_xt_next
                        eps_mu_t = eps_mu_t_next

                    if uq_array[timestep] == True:
                        eps_t= self.sample_from_gaussian(eps_mu_t, eps_var_t)
                        xt_next = self.singlestep_ddpm_sample(diffusion, xt, seq, timestep, eps_t)
                        exp_xt_next = self.conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], mc_eps_exp_t=mc_eps_exp_t)
                        var_xt_next = self.conditioned_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst=eps_var_t, seq=seq, timestep=timestep, pre_wuq= uq_array[timestep])
                        if uq_array[timestep-1] == True:
                            list_xt_next_i, list_eps_mu_t_next_i=[], []
                            for _ in range(mc_sample_size):
                                var_xt_next = torch.clamp(var_xt_next, min=0)
                                xt_next_i = self.sample_from_gaussian(exp_xt_next, var_xt_next)
                                list_xt_next_i.append(xt_next_i)
                                xt_next_i_in = torch.cat((cond_img, xt_next_i), dim=1)
                                assert xt_next_i_in.shape[1] == 4, f"Expected xt_next_i_in.shape[1] to be 4, but got {xt_next_i_in.shape[1]}"
                                eps_mu_t_next_i, _ = custom_model(xt_next_i_in, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), config, **model_kwargs)
                                list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                    
                            xt_next_in = torch.cat((cond_img, xt_next), dim=1)
                            assert xt_next_in.shape[1] == 4, f"Expected xt_next_in.shape[1] to be 4, but got {xt_next_in.shape[1]}"
                            eps_mu_t_next, eps_var_t_next = custom_model(xt_next_in, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), config, **model_kwargs)
                            list_xt_next_i = torch.stack(list_xt_next_i, dim=0).cuda()
                            list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).cuda()
                            cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                        else:
                            xt_next_in = torch.cat((cond_img, xt_next), dim=1)
                            assert xt_next_in.shape[1] == 4, f"Expected xt_next_in.shape[1] to be 4, but got {xt_next_in.shape[1]}"
                            eps_mu_t_next = custom_model.accurate_forward(xt_next_in, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), config, **model_kwargs)
                    else:
                        xt_next = self.singlestep_ddpm_sample(diffusion, xt, seq, timestep, eps_mu_t)
                        exp_xt_next = self.conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], acc_eps_t = eps_mu_t)
                        var_xt_next = self.conditioned_var_iteration(diffusion, var_xt, cov_xt_epst= None, var_epst=None, seq= seq, timestep=timestep, pre_wuq= uq_array[timestep])
                        if uq_array[timestep-1] == True:
                            list_xt_next_i, list_eps_mu_t_next_i=[], []
                            for _ in range(mc_sample_size):
                                var_xt_next = torch.clamp(var_xt_next, min=0)
                                xt_next_i = self.sample_from_gaussian(exp_xt_next, var_xt_next)
                                list_xt_next_i.append(xt_next_i)
                                xt_next_i_in = torch.cat((cond_img, xt_next_i), dim=1)
                                assert xt_next_i_in.shape[1] == 4, f"Expected xt_next_i_in.shape[1] to be 4, but got {xt_next_i_in.shape[1]}"
                                eps_mu_t_next_i, _ = custom_model(xt_next_i_in, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), config, **model_kwargs)
                                list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                    
                            xt_next_in = torch.cat((cond_img, xt_next), dim=1)
                            assert xt_next_in.shape[1] == 4, f"Expected xt_next_in.shape[1] to be 4, but got {xt_next_in.shape[1]}"
                            eps_mu_t_next, eps_var_t_next = custom_model(xt_next_in, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), config, **model_kwargs)
                            list_xt_next_i = torch.stack(list_xt_next_i, dim=0).cuda()
                            list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).cuda()
                            cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                        else:
                            xt_next_in = torch.cat((cond_img, xt_next), dim=1)
                            assert xt_next_in.shape[1] == 4, f"Expected xt_next_in.shape[1] to be 4, but got {xt_next_in.shape[1]}"
                            eps_mu_t_next = custom_model.accurate_forward(xt_next_in, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), config, **model_kwargs)

                # var_sum[:, loop] = var_xt_next.sum(dim=(1,2,3))  
                var_sum[:, loop, :, :] = var_xt_next.sum(dim=(1)) # var_sum: (num_batch, n_rounds, 256,256)
                x = self.inverse_data_transform(config, xt_next)
                sample_x.append(x)
                
            # sample_x = torch.concat(sample_x, dim=0) # (num_round*num_batch,1,256,256)
            # var = []       
            # for j in range(n_rounds):
            #     var.append(var_sum[:, j]) # [(num_batch),(num_batch),...], len(var)==num_round
            # var = torch.concat(var, dim=0) # (num_round*num_batch)
            # sorted_var, sorted_indices = torch.sort(var, descending=True)
            # reordered_sample_x = torch.index_select(sample_x, dim=0, index=sorted_indices.int()) # (num_round*num_batch,1,256,256)
            # grid_sample_x = make_grid(reordered_sample_x, nrow=8, padding=2)
            # tvu.save_image(grid_sample_x.cpu().float(), os.path.join(exp_dir, "sorted_sample.png"))

            var_sum_flatten = var_sum.view(sample_batch_size, n_rounds, -1) # (num_batch, n_rounds, 256*256)
            sample_x_flatten = [item.view(sample_batch_size,-1) for item in sample_x] # (num_batch, 256*256)
            reordered_sample_x = torch.zeros(var_sum_flatten.shape).cuda() # (num_batch, n_rounds, 256*256)
            for batch_i in range(sample_batch_size): # CT slice level
                for pix_i in range(config.data.image_size**2): # pixel level on a CT slice
                    var = []       
                    sample_x_pix = []
                    for j in range(n_rounds): # different ensemble for each pixel
                        var.append(var_sum_flatten[batch_i, j, pix_i]) 
                        sample_x_pix.append(sample_x_flatten[j][batch_i, pix_i])
                    var = torch.stack(var, dim=0) # (num_round)
                    sample_x_pix = torch.stack(sample_x_pix, dim=0) # (num_round)
                    sorted_var, sorted_indices = torch.sort(var, descending=True)
                    reordered_sample_x_pix = torch.index_select(sample_x_pix, dim=0, index=sorted_indices.int())
                    reordered_sample_x[batch_i,:,pix_i] = reordered_sample_x_pix

            reordered_sample_x = reordered_sample_x.view(sample_batch_size, n_rounds, config.data.image_size, config.data.image_size)

            return reordered_sample_x

    def p_sample_loop_known(
        self,
        model,
        shape,
        img,
        step=1000,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conditioner = None,
        classifier=None,
        custom_model=None,
        config=None,
        diffusion=None,
        fixed_xT=None,
        num_rounds=None,
        step_control=None
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = img.to(device)
        noise = th.randn_like(img[:, :1, ...]).to(device)
        x_noisy = torch.cat((img[:, :-1,  ...], noise), dim=1)  #add noise as the last channel

        liver_mask = img[:,:-1].clone()
        liver_mask[liver_mask>0] = 1
        ct_img = img[:,:-1].clone()

        if self.dpm_solver:
            print('dpm-solver!!!')
            final = {}
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas= th.from_numpy(self.betas))

            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=model_kwargs,
            )

            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding", img = img[:, :-1,  ...])

            ## Steps in [20, 30] can generate quite good samples.
            sample, cal = dpm_solver.sample(
                noise.to(dtype=th.float),
                steps= step,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            sample[:,-1,:,:] = norm(sample[:,-1,:,:])
            final["sample"] = sample
            final["cal"] = cal

            # cal_out = torch.clamp(final["cal"] + 0.25 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
            cal_out = th.sigmoid(final["cal"]).view(final["cal"].shape[0], 3,32,32)[:,1,] # (B,N)->(B,[D,H,W])
            return final["sample"], x_noisy, img, final["cal"], cal_out
        else:
            if not self.uncertainty_control:
                print('no dpm-solver')
                i = 0
                letters = string.ascii_lowercase
                name = ''.join(random.choice(letters) for i in range(10)) 
                for sample in self.p_sample_loop_progressive(
                    model,
                    shape,
                    time = step,
                    noise=x_noisy,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    org=org,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                ):
                    final = sample

                # if final["cal"] is not None:
                #     if dice_score(final["sample"][:,-1,:,:].unsqueeze(1), final["cal"]) < 0.65:
                #         cal_out = torch.clamp(final["cal"] + 0.25 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
                #     else:
                #         cal_out = torch.clamp(final["cal"] * 0.5 + 0.5 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
                # else:
                #     cal_out = None
                cal_out = th.sigmoid(final["cal"]).view(final["cal"].shape[0], 3,32,32)[:,1,] # (B,N)->(B,[D,H,W])
                return final["sample"], x_noisy, img, final["cal"], cal_out, final['sample_prog'], final['cal_prog']
            elif self.uncertainty_control:
                final_sample = self.bayes_diff_sample(
                                num_timesteps=step, 
                                timesteps=step_control,
                                skip_type="uniform",
                                total_n_samples=img.shape[0]*num_rounds,
                                sample_batch_size=img.shape[0],
                                cond_class=None,
                                mc_size=10,
                                model_kwargs=model_kwargs,
                                cond_img=ct_img,
                                fixed_xT=fixed_xT,
                                custom_model=custom_model,
                                config=config,
                                diffusion=diffusion,
                                )     
                return final_sample

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        org=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(time))[::-1]
        org_c = img.size(1)
        org_MRI = img[:, :-1, ...]      #original brain MR image
        out_progress = {}

        if progress:
            # Lazy import so that we don't depend on tqdm.

            indices = tqdm(indices)

            for i in indices:
                    t = th.tensor([i] * shape[0], device=device)
                    # if i%100==0:
                        # print('sampling step', i)
                        # viz.image(visualize(img.cpu()[0, -1,...]), opts=dict(caption="sample"+ str(i) ))

                    with th.no_grad():
                        # print('img bef size',img.size())
                        if img.size(1) != org_c:
                            img = torch.cat((org_MRI,img), dim=1)       #in every step, make sure to concatenate the original image to the sampled segmentation mask

                        out = self.p_sample(
                            model,
                            img.float(),
                            t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            model_kwargs=model_kwargs,
                        )

                        if (i+1)%100==0 or (i<=99 and (i+1)%10==0) or i==0:
                            if 'sample_prog' not in out_progress.keys():
                                sample_dict = {'sample_prog':{str(i):out["sample"]}}
                                cal_dict = {'cal_prog':{str(i):th.sigmoid(out["cal"]).view(out["cal"].shape[0], 32,32,3)[...,1]}}
                                out_progress.update(sample_dict)
                                out_progress.update(cal_dict)
                            else:
                                out_progress['sample_prog'][str(i)] = out["sample"]
                                out_progress['cal_prog'][str(i)] = th.sigmoid(out["cal"]).view(out["cal"].shape[0], 32,32,3)[...,1]
                            if i==0:
                                out.update(out_progress)
                        yield out
                        img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )


        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x[:, -1:, ...], t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x[:, -1:, ...].shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x[:, -1:, ...].shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x[:, -1:, ...])

        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x[:, -1:, ...].shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}



    def ddim_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(499,500, (b,), device=device).long().to(device)

        img1=torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)

        noise = th.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol=lambdaint*x_noisy1+(1-lambdaint)*x_noisy2

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(99, 100, (b,), device=device).long().to(device)

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):

            final = sample
       # viz.image(visualize(final["sample"].cpu()[0, ...]), opts=dict(caption="sample"+ str(10) ))
        return final["sample"]

    def ddim_sample_loop_known(
            self,
            model,
            shape,
            img,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta = 0.0
    ):
        print('ddim!!!')
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]

        img = img.to(device)

        # t = th.randint(499,500, (b,), device=device).long().to(device)
        # t = th.randint(49,50, (b,), device=device).long().to(device)
        noise = th.randn_like(img[:, :1, ...]).to(device)

        x_noisy = torch.cat((img[:, :-1, ...], noise), dim=1).float()
        img = img.to(device)

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=500,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample

        return final["sample"], x_noisy, img, final['sample_prog']


    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        time=500,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(time-1))[::-1]
        org_c = img.size(1)
        orghigh = img[:, :-1, ...]
        out_progress = {}

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

            for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                with th.no_grad():
                    if img.shape[1] != org_c:
                        img = torch.cat((orghigh,img), dim=1).float()

                    out = self.ddim_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )
                    # print('max: '+str(out['sample'].max()))
                    # print('min: '+str(out['sample'].min()))
                    if (i+1)%10==0 or i==0:
                        if 'sample_prog' not in out_progress.keys():
                            sample_dict = {'sample_prog':{str(i):out["sample"]}}
                            out_progress.update(sample_dict)
                        else:
                            out_progress['sample_prog'][str(i)] = out["sample"]
                        if i==0:
                            out.update(out_progress)
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}



    def training_losses_segmentation(self, model, classifier, x_start, t, logger, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start[:, -1:, ...])


        mask = x_start[:, -1:, ...]
        res = torch.where(mask > 0, 1, 0).float()   #merge all tumor classes into one to get a binary segmentation mask

        res_t = self.q_sample(res, t, noise=noise)     #add noise to the segmentation channel
        x_t=x_start.float()
        x_t[:, -1:, ...]=res_t.float()
        terms = {}


        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:

            model_output, cal = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                C=1
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=res,
                    x_t=res_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=res, x_t=res_t, t=t
                )[0],
                ModelMeanType.START_X: res,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            'Loss for the diffUNet'
            # model_output = (cal > 0.5) * (model_output >0.5) * model_output if 2. * (cal*model_output).sum() / (cal+model_output).sum() < 0.75 else model_output
            terms["mse_diff"] = mean_flat((target - model_output) ** 2 ) 
            # terms["loss_cal"] = mean_flat((res - cal) ** 2)
            
            'Loss for edge enhancement'
            # pred_xstart = self._predict_xstart_from_eps(x_t=res_t, t=t, eps=model_output)
            # bce_kwargs = {}
            # soft_dice_kwargs = {}
            # assert pred_xstart.shape == res.shape
            # assert len(pred_xstart.shape) == len(res.shape) == 4
            # terms["loss_boundary"] = DC_and_BCE_loss(bce_kwargs, soft_dice_kwargs)(pred_xstart, res, t)

            # terms["loss_boundary"], boundary_att = self.calc_boundary(res, t, 1000, model_output, target)
            # for t_ in range(999,0,-1):
            #     t_tensor = torch.tensor(t_).cuda().view(1)
            #     terms["loss_boundary"], boundary_att = self.calc_boundary(res, t_tensor, 1000, model_output, noise)
            #     save_nifti(boundary_att.squeeze(1)[0].detach().cpu().numpy(), '/data1/xzhang2/liver_vessel_segmentation/model/GATSegDiff/outputs/logger/temp/boundary_time_'+str(t_).zfill(3)+'.nii.gz')

            'Loss for the conditional part'
            terms["loss_cal"] = F.binary_cross_entropy_with_logits(cal.view(*model_kwargs['speed'].shape), model_kwargs['speed']) ## current implementation!!!
            # terms["loss_cal"] = self.SCEloss(cal, model_kwargs['speed'].view(cal.shape))

            # terms["loss_cal"] = nn.BCELoss()(cal.type(th.float), res.type(th.float)) + DiceLoss()(cal.type(th.float), res.type(th.float))
            # terms["mse"] = (terms["mse_diff"] + terms["mse_cal"]) / 2.
            if "vb" in terms:
                terms["loss"] = terms["mse_diff"] + terms["vb"] # Although 'vb' has been calculated, it does not involve in the real loss propogation. 
            else:
                terms["loss"] = terms["mse_diff"] 

        else:
            raise NotImplementedError(self.loss_type)

        return (terms, model_output, cal)


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bptimestepsd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

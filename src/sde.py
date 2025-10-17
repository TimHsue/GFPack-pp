import torch
import numpy as np
from torch_geometric.data import Data
import functools

#----- VE SDE -----
#------------------
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=25):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_sde(t, sigma_min=0.01, sigma_max=25):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

def ve_prior(shape, sigma_min=0.01, sigma_max=25):
    return torch.randn(*shape) * sigma_max

#----- VP SDE -----
#------------------
def vp_marginal_prob(x, t, beta_0=0.1, beta_1=20):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def vp_sde(t, beta_0=0.1, beta_1=20):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = torch.sqrt(beta_t)
    return drift_coeff, diffusion_coeff

def vp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

#----- sub-VP SDE -----
#----------------------
def subvp_marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

def subvp_sde(t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    discount = 1. - torch.exp(-2 * beta_0 * t - (beta_1 - beta_0) * t ** 2)
    diffusion_coeff = torch.sqrt(beta_t * discount)
    return drift_coeff, diffusion_coeff

def subvp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

def init_sde(sde_mode):
    # the SDE-related hyperparameters are copied from https://github.com/yang-song/score_sde_pytorch
    if sde_mode == 've':
        sigma_min = 1.0
        sigma_max = 1000.0
        eps = 1e-5
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)
    elif sde_mode == 'vp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
    elif sde_mode == 'subvp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(subvp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(subvp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(subvp_sde, beta_0=beta_0, beta_1=beta_1)
    else:
        raise NotImplementedError
    return prior_fn, marginal_prob_fn, sde_fn, eps


def lossFun(model, state, gnnFeatureData, marginalProbFunc):
    state.to(model.device)
    batchSize = int(state.batch.max().item() + 1)
    randomBatch = torch.rand(batchSize, device=model.device) * (1 - 0.01) + 0.01
    
    actionMask = (state.z.reshape(batchSize, -1) == 0).float().reshape(-1, 1)
    perturbFactor = torch.randn_like(state.x, device=model.device) * actionMask
    perturbedStateX = state.x.clone()
    
    polyIds = (state.y.reshape(-1, 1) * actionMask).long().reshape(-1)

    mu, std = marginalProbFunc(state.x, randomBatch)
    std = std[state.batch].view(-1, 1) + 1e-5
    perturbedStateX = mu + perturbFactor * std

    o1 = model(perturbedStateX, polyIds, state.z, state.batch, randomBatch, gnnFeatureData)
    o1 = o1 * actionMask
    
    delta = (o1) * std + perturbFactor
    lossAll = (delta**2)
    lossAll = lossAll.reshape(batchSize, -1).sum(dim=1) / state.x.shape[1]
    
    weight = state.w.reshape(-1)
    weightSum = weight.sum()
    lossAll = lossAll * weight 

    loss = torch.sum(lossAll) / weightSum
    return loss, delta.detach()


def pc_sampler_state(score_model,  
               sde_coeff,
               polyNumbers,
               polyIds,
               gnnFeatureData,
               paddingMaskData,
               batch_size=512, 
               num_steps=128,
               ):
    # t = torch.ones(1, device=score_model.device).unsqueeze(-1)
    # init_x = torch.zeros(batch_size * polyNumbers, 2, device=device) + 0.5
    init_x = torch.rand(batch_size, polyNumbers, 4, device=score_model.device)

    polyIds = polyIds.unsqueeze(0).repeat(batch_size, 1)
    paddingMaskData = paddingMaskData.unsqueeze(0).repeat(batch_size, 1)
    init_x_batch = torch.tensor([i for i in range(batch_size) for _ in range(polyNumbers)], dtype=torch.int64).cpu()
    state = Data(x=init_x.reshape(-1, 4).clone(), 
                 y=polyIds.reshape(-1).clone(),
                 z=paddingMaskData.reshape(-1).clone(),
                 batch=init_x_batch).to(score_model.device)

    time_steps = torch.linspace(1, 0.01, num_steps, device=score_model.device)
    random_step = time_steps[0] - time_steps[1]
   
    actionsRes = []
    with torch.no_grad():
        feature = score_model.geo_feature(gnnFeatureData)
        actionMask = (state.z.reshape(batch_size, -1) == 0).float().reshape(-1, 1)
        
        for i in range(len(time_steps)):
            time_step = time_steps[i]
            if i + 1 > len(time_steps) - 1:
                step_size = time_steps[i - 1] - time_steps[i]
            else:
                step_size = time_steps[i] - time_steps[i + 1]
            step_size *= 14

            batch_time_step = torch.ones(batch_size, device=score_model.device) * time_step
             
            baseX = state.x.clone() * actionMask

            state.x = state.x + (torch.rand_like(state.x) - 0.5) * 0.001
            state.x = state.x * actionMask
            
            polyIdsInput = (state.y.reshape(-1, 1) * actionMask).long().reshape(-1)
            o1 = score_model(state.x, polyIdsInput, state.z, state.batch, batch_time_step, gnnFeatureData, feature)
            outputAll = o1 * actionMask

            output = outputAll.reshape(batch_size * polyNumbers, 4)
            
            batch_time_step = torch.ones(batch_size, device=score_model.device).unsqueeze(-1) * time_step
            drift, g = sde_coeff(batch_time_step)
            # print(i, g)
            g = g.view(-1, 1).repeat(1, polyNumbers).to(score_model.device).reshape(-1, 1)
            drift = drift.view(-1, 1).to(score_model.device)
            # print(output.shape, drift.shape, g.shape)
            drift = drift + g ** 2 * output
            delta = drift * step_size
            mean_x = baseX + delta + g * torch.sqrt(random_step) * torch.randn_like(baseX)
            state.x = mean_x

            actionsRes.append(mean_x.clone().reshape(batch_size, polyNumbers, 4).unsqueeze(1))

    # The last step does not include any noise
    res = torch.cat(actionsRes, dim=1)
    return res, mean_x.reshape(batch_size, polyNumbers, 4)


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                                                    for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param)) # only update the ema-params

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                                shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']

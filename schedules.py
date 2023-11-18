# -*- coding:utf-8 -*-
# Author: Kedong Xiu
# description: beta和alpha的调度方案

import math
import numpy as np


def betas_for_alpha_bar(num_diffusion_step, alpha_bar_func, max_beta=0.999):
    """
    Create a beta schedule that discretized the given alpha_t_bar function,
    which defines the cumulative product of (1 - beta) over time from t = [0, 1].

    Args:
        num_diffusion_step (int): the number of betas to produce.
        alpha_bar_func (Func): a lambda function that takes an argument t from 0 to 1 and produces the cumulative product of (1 - beta) up to that part of the diffusion process.
        max_beta (float, optional): the maximum beta to use; use values lower than 1 to prevent singularities. Defaults to 0.999.
    """
    betas = []
    for i in range(num_diffusion_step):
        t1 = i / num_diffusion_step
        t2 = (i + 1) / num_diffusion_step
        betas.append(min(1 - alpha_bar_func(t2) / alpha_bar_func(t1), max_beta))
    return np.array(betas)


def beta_for_alpha_bar_trunc_cosine(num_diffusion_step, alpha_bar_func, max_beta=0.999):
    """
    Create a beta schedule that discretized the given alpha_t_bar function,
    which defines the cumulative product of (1 - beta) over time from t = [0, 1].
    Note: this function is a specific variant of betas_for_alpha_bar() that uses a cosine function to smooth the beta schedule.

    Args:
        num_diffusion_step (int): the number of betas to produce.
        alpha_bar_func (Func): a lambda function that takes an argument t from 0 to 1 and produces the cumulative product of (1 - beta) up to that part of the diffusion process.
        max_beta (float, optional): the maximum beta to use; use values lower than 1 to prevent singularities. Defaults to 0.999.
    """
    betas = []
    betas.append(min(1 - alpha_bar_func(0), max_beta))
    for i in range(num_diffusion_step - 1):
        t1 = i / num_diffusion_step
        t2 = (i + 1) / num_diffusion_step
        betas.append(min(1 - alpha_bar_func(t2) / alpha_bar_func(t1), max_beta))
    return np.array(betas)


def reflect_beta_schedule_by_name(schedule_name, num_diffusion_steps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)

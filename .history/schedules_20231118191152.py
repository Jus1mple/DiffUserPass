# -*- coding:utf-8 -*-
# Author: Kedong Xiu
# description: beta和alpha的调度方案

import numpy as np

def betas_for_alpha_bar(num_diffusion_step, alpha_bar_func, max_beta = 0.999):
    
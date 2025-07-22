import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1 / dim * torch.ones_like(w1, device=w1.device, dtype=w1.dtype)  # [batch,N,1]
    K = torch.exp(-cost / reg)
    Kt = K.transpose(2, 1)
    for i in range(max_iter):
        v = w2 / (torch.bmm(Kt, u) + 1e-8)  # [batch,N,1]
        u = w1 / (torch.bmm(K, v) + 1e-8)  # [batch,N,1]

    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow


def ccr_logit_loss(logits_student, logits_teacher, temperature, cost_matrix=None, sinkhorn_lambda=25, sinkhorn_iter=30):
    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)

    cost_matrix = F.relu(cost_matrix) + 1e-8
    cost_matrix = cost_matrix.to(pred_student.device)

    # flow shape [bxnxn]
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return ws_distance


def ccr_logit_loss_with_speration(logits_student, logits_teacher, gt_label, temperature, gamma, cost_matrix=None,
                                  sinkhorn_lambda=0.05, sinkhorn_iter=10):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

    # N*class
    N, c = logits_student.shape
    s_i = F.log_softmax(logits_student, dim=1)
    t_i = F.softmax(logits_teacher, dim=1)
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()
    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_student = logits_student[mask].reshape(N, -1)
    logits_teacher = logits_teacher[mask].reshape(N, -1)

    cost_matrix = cost_matrix.repeat(N, 1, 1)
    gd_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    cost_matrix = cost_matrix[gd_mask].reshape(N, c - 1, c - 1)

    # N*class
    loss_ccr = ccr_logit_loss(logits_student, logits_teacher, temperature, cost_matrix, sinkhorn_lambda, sinkhorn_iter)

    return loss_t + gamma * loss_ccr


def adaptive_avg_std_pool2d(input_tensor, out_size=(1, 1), eps=1e-5):
    def start_index(a, b, c):
        return int(np.floor(a * c / b))

    def end_index(a, b, c):
        return int(np.ceil((a + 1) * c / b))

    b, c, isizeH, isizeW = input_tensor.shape
    if len(out_size) == 2:
        osizeH, osizeW = out_size
    else:
        osizeH = osizeW = out_size

    avg_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # cov_pooled_tensor = torch.zeros((b, c*c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    cov_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # block_list = []
    for oh in range(osizeH):
        istartH = start_index(oh, osizeH, isizeH)
        iendH = end_index(oh, osizeH, isizeH)
        kH = iendH - istartH
        for ow in range(osizeW):
            istartW = start_index(ow, osizeW, isizeW)
            iendW = end_index(ow, osizeW, isizeW)
            kW = iendW - istartW

            # avg pool2d
            input_block = input_tensor[:, :, istartH:iendH, istartW:iendW]
            avg_pooled_tensor[:, :, oh, ow] = input_block.mean(dim=(-1, -2))
            # diagonal cov pool2d
            cov_pooled_tensor[:, :, oh, ow] = torch.sqrt(input_block.var(dim=(-1, -2)) + eps)

    return avg_pooled_tensor, cov_pooled_tensor


def ccr_feature_loss(f_s, f_t, eps=1e-5, grid=1):
    if grid == 1:
        f_s_avg, f_t_avg = f_s.mean(dim=(-1, -2)), f_t.mean(dim=(-1, -2))
        f_s_std, f_t_std = torch.sqrt(f_s.var(dim=(-1, -2)) + eps), torch.sqrt(f_t.var(dim=(-1, -2)) + eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / f_s.size(0)
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / f_s.size(0)
    elif grid > 1:
        f_s_avg, f_s_std = adaptive_avg_std_pool2d(f_s, out_size=(grid, grid), eps=eps)
        f_t_avg, f_t_std = adaptive_avg_std_pool2d(f_t, out_size=(grid, grid), eps=eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / (grid ** 2 * f_s.size(0))
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / (grid ** 2 * f_s.size(0))

    return mean_loss, cov_loss

def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        feat_s, _ = student(data, is_feat=True)
        feat_t, _ = teacher(data, is_feat=True)
    feat_s_shapes = [f.shape for f in feat_s]
    feat_t_shapes = [f.shape for f in feat_t]
    return feat_s_shapes, feat_t_shapes

def nkd_loss_origin(logit_s, logit_t, gt_label, temp, gamma):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

        # N*class
    N, c = logit_s.shape
    s_i = F.log_softmax(logit_s, dim=1)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 1).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = F.log_softmax(logit_s / temp, dim=1)
    T_i = F.softmax(logit_t / temp, dim=1)

    loss_non = (T_i * S_i).sum(dim=1).mean()
    loss_non = - gamma * (temp ** 2) * loss_non

    return loss_t + loss_non

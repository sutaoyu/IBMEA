#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import debugpy

try:
    from layers import *
except:
    from src.layers import *


class GAT(nn.Module):
    def __init__(
        self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag
    ):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(
                    n_heads[i],
                    f_in,
                    n_units[i + 1],
                    attn_dropout,
                    diag,
                    nn.init.ones_,
                    False,
                )
            )

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(x)
        return x


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs"""
    return im.mm(s.t())


def l2norm(X):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(
            torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad
        )

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [
            weight_norm[idx] * F.normalize(embs[idx])
            for idx in range(self.modal_num)
            if embs[idx] is not None
        ]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb


class MultiModalFusionNew_allmodal(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(
            torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad
        )

        self.linear_0 = nn.Linear(100, 600)
        self.linear_1 = nn.Linear(100, 600)
        self.linear_2 = nn.Linear(100, 600)
        self.linear_3 = nn.Linear(300, 600)
        self.linear = nn.Linear(600, 600)
        self.v = nn.Linear(600, 1, bias=False)

        self.LN_pre = nn.LayerNorm(600)
        self.LN_pre = nn.LayerNorm(600)

    def forward(self, embs):
        assert len(embs) == self.modal_num

        emb_list = []

        if embs[0] is not None:
            emb_list.append(self.linear_0(embs[0]).unsqueeze(1))
        if embs[1] is not None:
            emb_list.append(self.linear_1(embs[1]).unsqueeze(1))
        if embs[2] is not None:
            emb_list.append(self.linear_2(embs[2]).unsqueeze(1))
        if embs[3] is not None:
            emb_list.append(self.linear_3(embs[3]).unsqueeze(1))
        new_embs = torch.cat(emb_list, dim=1)  # [n, 4, e]
        new_embs = self.LN_pre(new_embs)
        s = self.v(torch.tanh(self.linear(new_embs)))  # [n, 4, 1]
        a = torch.softmax(s, dim=-1)
        joint_emb_1 = torch.matmul(a.transpose(-1, -2), new_embs).squeeze(1)  # [n, e]
        joint_emb = joint_emb_1
        return joint_emb


class IBMultiModal(nn.Module):
    def __init__(
        self,
        args,
        ent_num,
        img_feature_dim,
        char_feature_dim=None,
        use_project_head=False,
    ):
        super(IBMultiModal, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.rel_fc_mu = nn.Linear(attr_dim, attr_dim)
        self.rel_fc_std = nn.Linear(attr_dim, attr_dim)
        self.rel_fc_d1 = nn.Linear(attr_dim, attr_dim)
        self.rel_fc_d2 = nn.Linear(attr_dim, attr_dim)

        self.att_fc = nn.Linear(1000, attr_dim)
        self.att_fc_mu = nn.Linear(attr_dim, attr_dim)
        self.att_fc_std = nn.Linear(attr_dim, attr_dim)
        self.att_fc_d1 = nn.Linear(attr_dim, attr_dim)
        self.att_fc_d2 = nn.Linear(attr_dim, attr_dim)

        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.img_fc_mu = nn.Linear(img_dim, img_dim)
        self.img_fc_std = nn.Linear(img_dim, img_dim)
        self.img_fc_d1 = nn.Linear(img_dim, img_dim)
        self.img_fc_d2 = nn.Linear(img_dim, img_dim)

        joint_dim = 600
        self.joint_fc = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_mu = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_std = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_d1 = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_d2 = nn.Linear(joint_dim, joint_dim)

        use_graph_vib = self.args.use_graph_vib
        use_attr_vib = self.args.use_attr_vib
        use_img_vib = self.args.use_img_vib
        use_rel_vib = self.args.use_rel_vib

        no_diag = self.args.no_diag

        if no_diag:
            diag = False
        else:
            diag = True

        self.use_graph_vib = use_graph_vib
        self.use_attr_vib = use_attr_vib
        self.use_img_vib = use_img_vib
        self.use_rel_vib = use_rel_vib
        self.use_joint_vib = self.args.use_joint_vib

        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        self.kld_loss = 0
        self.gph_layer_norm_mu = nn.LayerNorm(self.input_dim, elementwise_affine=True)
        self.gph_layer_norm_std = nn.LayerNorm(self.input_dim, elementwise_affine=True)
        if self.args.structure_encoder == "gcn":
            if self.use_graph_vib:
                self.cross_graph_model_mu = GCN(
                    self.n_units[0],
                    self.n_units[1],
                    self.n_units[2],
                    dropout=self.args.dropout,
                )
                self.cross_graph_model_std = GCN(
                    self.n_units[0],
                    self.n_units[1],
                    self.n_units[2],
                    dropout=self.args.dropout,
                )
            else:
                self.cross_graph_model = GCN(
                    self.n_units[0],
                    self.n_units[1],
                    self.n_units[2],
                    dropout=self.args.dropout,
                )
        elif self.args.structure_encoder == "gat":
            if self.use_graph_vib:
                self.cross_graph_model_mu = GAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=args.dropout,
                    attn_dropout=args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=diag,
                )
                self.cross_graph_model_std = GAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=args.dropout,
                    attn_dropout=args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=diag,
                )
            else:
                self.cross_graph_model = GAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=args.dropout,
                    attn_dropout=args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=True,
                )

        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(
                self.n_units[2], self.n_units[2], self.n_units[2], dropout
            )

        if self.args.fusion_id == 1:
            self.fusion = MultiModalFusion(
                modal_num=self.args.inner_view_num, with_weight=self.args.with_weight
            )
        elif self.args.fusion_id == 2:
            self.fusion = MultiModalFusionNew_allmodal(
                modal_num=self.args.inner_view_num, with_weight=self.args.with_weight
            )

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        from torch.distributions.kl import kl_divergence
        from torch.distributions import Normal

        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        mu_1_fixed = mu_1.clone()
        sigma_1_fixed = sigma_1.clone()
        mu_1_fixed[torch.isnan(mu_1_fixed)] = 0
        mu_1_fixed[torch.isinf(mu_1_fixed)] = torch.max(
            mu_1_fixed[~torch.isinf(mu_1_fixed)]
        )
        sigma_1_fixed[torch.isnan(sigma_1_fixed)] = 1
        sigma_1_fixed[torch.isinf(sigma_1_fixed)] = torch.max(
            sigma_1_fixed[~torch.isinf(sigma_1_fixed)]
        )
        sigma_1_fixed[sigma_1_fixed <= 0] = 1
        q_target = Normal(mu_1_fixed, sigma_1_fixed)

        mu_2_fixed = mu_2.clone()
        sigma_2_fixed = sigma_2.clone()
        mu_2_fixed[torch.isnan(mu_2_fixed)] = 0
        mu_2_fixed[torch.isinf(mu_2_fixed)] = torch.max(
            mu_2_fixed[~torch.isinf(mu_2_fixed)]
        )
        sigma_2_fixed[torch.isnan(sigma_2_fixed)] = 1
        sigma_2_fixed[torch.isinf(sigma_2_fixed)] = torch.max(
            sigma_2_fixed[~torch.isinf(sigma_2_fixed)]
        )
        sigma_2_fixed[sigma_2_fixed <= 0] = 1
        q_context = Normal(mu_2_fixed, sigma_2_fixed)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def forward(
        self,
        input_idx,
        adj,
        img_features=None,
        rel_features=None,
        att_features=None,
        name_features=None,
        char_features=None,
    ):

        if self.args.w_gcn:
            if self.use_graph_vib:
                if self.args.structure_encoder == "gat":
                    gph_emb_mu = self.cross_graph_model_mu(
                        self.entity_emb(input_idx), adj
                    )
                    mu = self.gph_layer_norm_mu(gph_emb_mu)

                    gph_emb_std = self.cross_graph_model_std(
                        self.entity_emb(input_idx), adj
                    )
                    std = F.elu(gph_emb_std)
                    eps = torch.randn_like(std)
                    gph_emb = mu + eps * std
                    gph_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                    self.kld_loss = gph_kld_loss
                else:
                    mu = self.cross_graph_model_mu(self.entity_emb(input_idx), adj)
                    logstd = self.cross_graph_model_mu(self.entity_emb(input_idx), adj)
                    eps = torch.randn_like(mu)
                    gph_emb = mu + eps * torch.exp(logstd)
                    gph_kld_loss = self._kld_gauss(
                        mu, logstd, torch.zeros_like(mu), torch.ones_like(logstd)
                    )
                    self.kld_loss = gph_kld_loss
            else:
                gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        if self.args.w_img:
            if self.use_img_vib:
                img_emb = self.img_fc(img_features)
                img_emb_h = F.relu(img_emb)
                mu = self.img_fc_mu(img_emb_h)
                logvar = self.img_fc_std(img_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                img_emb = mu + eps * std
                img_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.img_kld_loss = img_kld_loss
            else:
                img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            if self.use_rel_vib:
                rel_emb = self.rel_fc(rel_features)
                rel_emb_h = F.relu(rel_emb)
                mu = self.rel_fc_mu(rel_emb_h)
                logvar = self.rel_fc_std(rel_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                rel_emb = mu + eps * std
                rel_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.rel_kld_loss = rel_kld_loss
            else:
                rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            if self.use_attr_vib:
                att_emb = self.att_fc(att_features)
                att_emb_h = F.relu(att_emb)
                mu = self.att_fc_mu(att_emb_h)
                logvar = self.att_fc_std(att_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                att_emb = mu + eps * std
                attr_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.attr_kld_loss = attr_kld_loss
            else:
                att_emb = self.att_fc(att_features)
        else:
            att_emb = None

        if self.args.w_name:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        if self.use_project_head:
            gph_emb = self.gph_pro(gph_emb)
            img_emb = self.img_pro(img_emb)
            rel_emb = self.rel_pro(rel_emb)
            att_emb = self.att_pro(att_emb)
            pass

        joint_emb = self.fusion(
            [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb]
        )

        if self.use_joint_vib:
            joint_emb = self.joint_fc(joint_emb)
            joint_emb_h = F.relu(joint_emb)
            mu = self.joint_fc_mu(joint_emb_h)
            logvar = self.joint_fc_std(joint_emb_h)
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            joint_emb = mu + eps * std
            joint_kld_loss = self._kld_gauss(
                mu, std, torch.zeros_like(mu), torch.ones_like(std)
            )
            self.joint_kld_loss = joint_kld_loss

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb

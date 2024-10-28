#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
from pprint import pprint
from transformers import (
    get_cosine_schedule_with_warmup,
)
import torch.optim as optim

try:
    from utils import *
    from models import *
    from Load import *
    from loss import *
except:
    from src.utils import *
    from src.models import *
    from src.Load import *
    from src.loss import *


def load_img_features(ent_num, file_dir, triples, use_mean_img=False):
    if "V1" in file_dir:
        split = "norm"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    elif "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = (
            "data/mmkb-datasets/"
            + filename
            + "/"
            + filename
            + "_id_img_feature_dict.pkl"
        )
    else:
        split = file_dir.split("/")[-1]
        img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"
    if use_mean_img:
        img_features = load_img(ent_num, img_vec_path)
    else:
        img_features = load_img_new(ent_num, img_vec_path, triples)
    return img_features


def load_img_features_dropout(
    ent_num, file_dir, triples, use_mean_img=False, img_dp_ratio=1.0
):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        if abs(1.0 - img_dp_ratio) > 1e-3:
            img_vec_path = (
                "data/mmkb-datasets/"
                + "mmkb_dropout"
                + "/"
                + filename
                + "_id_img_feature_dict_with_dropout{0}.pkl".format(img_dp_ratio)
            )
            print("dropout img_vec_path: ", img_vec_path)
        else:
            img_vec_path = (
                "data/mmkb-datasets/"
                + filename
                + "/"
                + filename
                + "_id_img_feature_dict.pkl"
            )
    else:
        split = file_dir.split("/")[-1]
        if abs(1.0 - img_dp_ratio) > 1e-3:
            img_vec_path = (
                "data/pkls/dbp_dropout/"
                + split
                + "_GA_id_img_feature_dict_with_dropout{0}.pkl".format(img_dp_ratio)
            )
        else:
            img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"
    if use_mean_img:
        img_features = load_img(ent_num, img_vec_path)
    else:
        img_features = load_img_new(ent_num, img_vec_path, triples)
    return img_features


class IBMEA:

    def __init__(self):

        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None

        self.img_features = None
        self.rel_features = None
        self.att_features = None
        self.char_features = None
        self.name_features = None
        self.ent_vec = None
        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.adj = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None
        self.multimodal_encoder = None
        self.weight_raw = None
        self.rel_fc = None
        self.att_fc = None
        self.img_fc = None
        self.char_fc = None
        self.shared_fc = None
        self.gcn_pro = None
        self.rel_pro = None
        self.attr_pro = None
        self.img_pro = None
        self.input_dim = None
        self.entity_emb = None
        self.input_idx = None
        self.n_units = None
        self.n_heads = None
        self.cross_graph_model = None
        self.params = None
        self.optimizer = None
        self.fusion = None
        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)
        self.set_seed(self.args.seed, self.args.cuda)
        self.device = torch.device(
            "cuda" if self.args.cuda and torch.cuda.is_available() else "cpu"
        )
        self.init_data()
        self.init_model()
        self.print_summary()
        self.best_hit_1 = 0.0
        self.best_epoch = 0
        self.best_data_list = []
        self.best_to_write = []

    @staticmethod
    def parse_options(parser):
        parser.add_argument(
            "--file_dir",
            type=str,
            default="data/DBP15K/zh_en",
            required=False,
            help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')",
        )
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")

        parser.add_argument(
            "--cuda",
            action="store_true",
            default=True,
            help="whether to use cuda or not",
        )
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument(
            "--epochs", type=int, default=1000, help="number of epochs to train"
        )
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument(
            "--hidden_units",
            type=str,
            default="128,128,128",
            help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma",
        )
        parser.add_argument(
            "--heads",
            type=str,
            default="2,2",
            help="heads in each gat layer, splitted with comma",
        )
        parser.add_argument(
            "--instance_normalization",
            action="store_true",
            default=False,
            help="enable instance normalization",
        )
        parser.add_argument(
            "--lr", type=float, default=0.005, help="initial learning rate"
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-2,
            help="weight decay (L2 loss on parameters)",
        )
        parser.add_argument(
            "--dropout", type=float, default=0.0, help="dropout rate for layers"
        )
        parser.add_argument(
            "--attn_dropout",
            type=float,
            default=0.0,
            help="dropout rate for gat layers",
        )
        parser.add_argument(
            "--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')"
        )
        parser.add_argument(
            "--csls", action="store_true", default=False, help="use CSLS for inference"
        )
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument(
            "--il", action="store_true", default=False, help="Iterative learning?"
        )
        parser.add_argument(
            "--semi_learn_step",
            type=int,
            default=10,
            help="If IL, what's the update step?",
        )
        parser.add_argument(
            "--il_start", type=int, default=500, help="If Il, when to start?"
        )
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument(
            "--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss"
        )
        parser.add_argument(
            "--with_weight",
            type=int,
            default=1,
            help="Whether to weight the fusion of different " "modal features",
        )
        parser.add_argument(
            "--structure_encoder",
            type=str,
            default="gat",
            help="the encoder of structure view, " "[gcn|gat]",
        )

        parser.add_argument(
            "--projection",
            action="store_true",
            default=False,
            help="add projection for model",
        )

        parser.add_argument(
            "--attr_dim",
            type=int,
            default=100,
            help="the hidden size of attr and rel features",
        )
        parser.add_argument(
            "--img_dim", type=int, default=100, help="the hidden size of img feature"
        )
        parser.add_argument(
            "--name_dim", type=int, default=100, help="the hidden size of name feature"
        )
        parser.add_argument(
            "--char_dim", type=int, default=100, help="the hidden size of char feature"
        )

        parser.add_argument(
            "--w_gcn", action="store_false", default=True, help="with gcn features"
        )
        parser.add_argument(
            "--w_rel", action="store_false", default=True, help="with rel features"
        )
        parser.add_argument(
            "--w_attr", action="store_false", default=True, help="with attr features"
        )
        parser.add_argument(
            "--w_name", action="store_false", default=True, help="with name features"
        )
        parser.add_argument(
            "--w_char", action="store_false", default=True, help="with char features"
        )
        parser.add_argument(
            "--w_img", action="store_false", default=True, help="with img features"
        )

        parser.add_argument(
            "--no_diag", action="store_true", default=False, help="GAT use diag"
        )
        parser.add_argument(
            "--use_joint_vib", action="store_true", default=False, help="use_joint_vib"
        )

        parser.add_argument(
            "--use_graph_vib", action="store_false", default=True, help="use_graph_vib"
        )
        parser.add_argument(
            "--use_attr_vib", action="store_false", default=True, help="use_attr_vib"
        )
        parser.add_argument(
            "--use_img_vib", action="store_false", default=True, help="use_img_vib"
        )
        parser.add_argument(
            "--use_rel_vib", action="store_false", default=True, help="use_rel_vib"
        )
        parser.add_argument(
            "--modify_ms", action="store_true", default=False, help="modify_ms"
        )
        parser.add_argument("--ms_alpha", type=float, default=0.1, help="ms scale_pos")
        parser.add_argument("--ms_beta", type=float, default=40.0, help="ms scale_neg")
        parser.add_argument("--ms_base", type=float, default=0.5, help="ms base")

        parser.add_argument("--Beta_g", type=float, default=0.001, help="graph beta")
        parser.add_argument("--Beta_i", type=float, default=0.001, help="graph beta")
        parser.add_argument("--Beta_r", type=float, default=0.01, help="graph beta")
        parser.add_argument("--Beta_a", type=float, default=0.001, help="graph beta")
        parser.add_argument(
            "--inner_view_num", type=int, default=6, help="the number of inner view"
        )

        parser.add_argument(
            "--word_embedding",
            type=str,
            default="glove",
            help="the type of word embedding, " "[glove|fasttext]",
        )
        parser.add_argument(
            "--use_project_head",
            action="store_true",
            default=False,
            help="use projection head",
        )

        parser.add_argument(
            "--zoom", type=float, default=0.1, help="narrow the range of losses"
        )
        parser.add_argument(
            "--save_path", type=str, default="save_pkl", help="save path"
        )
        parser.add_argument(
            "--pred_name", type=str, default="pred.txt", help="pred name"
        )

        parser.add_argument(
            "--hidden_size", type=int, default=300, help="the hidden size of MEAformer"
        )
        parser.add_argument(
            "--intermediate_size",
            type=int,
            default=400,
            help="the hidden size of MEAformer",
        )
        parser.add_argument(
            "--num_attention_heads",
            type=int,
            default=1,
            help="the number of attention_heads of MEAformer",
        )
        parser.add_argument(
            "--num_hidden_layers",
            type=int,
            default=1,
            help="the number of hidden_layers of MEAformer",
        )
        parser.add_argument("--position_embedding_type", default="absolute", type=str)
        parser.add_argument(
            "--use_intermediate",
            type=int,
            default=0,
            help="whether to use_intermediate",
        )
        parser.add_argument(
            "--replay", type=int, default=0, help="whether to use replay strategy"
        )
        parser.add_argument(
            "--neg_cross_kg",
            type=int,
            default=0,
            help="whether to force the negative samples in the opposite KG",
        )

        parser.add_argument(
            "--tau",
            type=float,
            default=0.1,
            help="the temperature factor of contrastive loss",
        )
        parser.add_argument(
            "--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss"
        )
        parser.add_argument(
            "--use_icl", action="store_true", default=False, help="use_icl"
        )
        parser.add_argument(
            "--use_bce", action="store_true", default=False, help="use_bce"
        )
        parser.add_argument(
            "--use_bce_new", action="store_true", default=False, help="use_bce_new"
        )
        parser.add_argument(
            "--use_ms", action="store_true", default=False, help="use_ms"
        )
        parser.add_argument(
            "--use_nt", action="store_true", default=False, help="use_nt"
        )
        parser.add_argument(
            "--use_nce", action="store_true", default=False, help="use_nce"
        )
        parser.add_argument(
            "--use_nce_new", action="store_true", default=False, help="use_nce_new"
        )

        parser.add_argument(
            "--use_sheduler", action="store_true", default=False, help="use_sheduler"
        )
        parser.add_argument(
            "--sheduler_gamma", type=float, default=0.98, help="sheduler_gamma"
        )

        parser.add_argument("--joint_beta", type=float, default=1.0, help="joint_beta")

        parser.add_argument(
            "--use_sheduler_cos",
            action="store_true",
            default=False,
            help="use_sheduler_cos",
        )
        parser.add_argument(
            "--num_warmup_steps", type=int, default=200, help="num_warmup_steps"
        )
        parser.add_argument(
            "--num_training_steps", type=int, default=1000, help="num_training_steps"
        )

        parser.add_argument(
            "--use_mean_img", action="store_true", default=False, help="use_mean_img"
        )
        parser.add_argument("--awloss", type=int, default=0, help="whether to use awl")
        parser.add_argument("--mtloss", type=int, default=0, help="whether to use awl")

        parser.add_argument(
            "--graph_use_icl", type=int, default=1, help="graph_use_icl"
        )
        parser.add_argument(
            "--graph_use_bce", type=int, default=1, help="graph_use_bce"
        )
        parser.add_argument("--graph_use_ms", type=int, default=1, help="graph_use_ms")

        parser.add_argument("--img_use_icl", type=int, default=1, help="img_use_icl")
        parser.add_argument("--img_use_bce", type=int, default=1, help="img_use_bce")
        parser.add_argument("--img_use_ms", type=int, default=1, help="img_use_ms")

        parser.add_argument("--attr_use_icl", type=int, default=1, help="attr_use_icl")
        parser.add_argument("--attr_use_bce", type=int, default=1, help="attr_use_bce")
        parser.add_argument("--attr_use_ms", type=int, default=1, help="attr_use_ms")

        parser.add_argument("--rel_use_icl", type=int, default=1, help="rel_use_icl")
        parser.add_argument("--rel_use_bce", type=int, default=1, help="rel_use_bce")
        parser.add_argument("--rel_use_ms", type=int, default=1, help="rel_use_ms")

        parser.add_argument(
            "--joint_use_icl", action="store_true", default=False, help="joint_use_icl"
        )
        parser.add_argument(
            "--joint_use_bce", action="store_true", default=False, help="joint_use_bce"
        )
        parser.add_argument(
            "--joint_use_ms", action="store_true", default=False, help="joint_use_ms"
        )

        parser.add_argument(
            "--img_dp_ratio", type=float, default=1.0, help="image dropout ratio"
        )
        parser.add_argument(
            "--fusion_id", type=int, default=1, help="default weight fusion"
        )
        return parser.parse_args()

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def print_summary(self):
        print("-----dataset summary-----")
        print("dataset:\t", self.args.file_dir)
        print("triple num:\t", len(self.triples))
        print("entity num:\t", self.ENT_NUM)
        print("relation num:\t", self.REL_NUM)
        print(
            "train ill num:\t",
            self.train_ill.shape[0],
            "\ttest ill num:\t",
            self.test_ill.shape[0],
        )
        print("-------------------------")

    def init_data(self):
        # Load data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device

        self.ent2id_dict, self.ills, self.triples, self.r_hs, self.r_ts, self.ids = (
            read_raw_data(file_dir, lang_list)
        )
        e1 = os.path.join(file_dir, "ent_ids_1")
        e2 = os.path.join(file_dir, "ent_ids_2")
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)

        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.r_hs)
        print("total ent num: {}, rel num: {}".format(self.ENT_NUM, self.REL_NUM))

        np.random.shuffle(self.ills)

        if abs(1.0 - self.args.img_dp_ratio) > 1e-3:
            self.img_features = load_img_features_dropout(
                self.ENT_NUM,
                file_dir,
                self.triples,
                self.args.use_mean_img,
                self.args.img_dp_ratio,
            )
        else:
            self.img_features = load_img_features(
                self.ENT_NUM, file_dir, self.triples, self.args.use_mean_img
            )
        self.img_features = F.normalize(torch.Tensor(self.img_features).to(device))
        print("image feature shape:", self.img_features.shape)
        data_dir, dataname = os.path.split(file_dir)
        if self.args.word_embedding == "glove":
            word2vec_path = "data/embedding/glove.6B.300d.txt"
        elif self.args.word_embedding == "fasttext":
            pass
        else:
            raise Exception("error word embedding")

        if "DBP15K" in file_dir:
            if self.args.w_name or self.args.w_char:
                print("name feature shape:", self.name_features.shape)
                print("char feature shape:", self.char_features.shape)
                pass

        self.train_ill = np.array(
            self.ills[: int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32
        )
        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate) :]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)

        self.left_non_train = list(
            set(self.left_ents) - set(self.train_ill[:, 0].tolist())
        )
        self.right_non_train = list(
            set(self.right_ents) - set(self.train_ill[:, 1].tolist())
        )

        print(
            "#left entity : %d, #right entity: %d"
            % (len(self.left_ents), len(self.right_ents))
        )
        print(
            "#left entity not in train set: %d, #right entity not in train set: %d"
            % (len(self.left_non_train), len(self.right_non_train))
        )
        self.rel_features = load_relation(self.ENT_NUM, self.triples, 1000)
        self.rel_features = torch.Tensor(self.rel_features).to(device)
        print("relation feature shape:", self.rel_features.shape)
        a1 = os.path.join(file_dir, "training_attrs_1")
        a2 = os.path.join(file_dir, "training_attrs_2")
        self.att_features = load_attr(
            [a1, a2], self.ENT_NUM, self.ent2id_dict, 1000
        ) 
        self.att_features = torch.Tensor(self.att_features).to(device)
        print("attribute feature shape:", self.att_features.shape)

        self.adj = get_adjr(
            self.ENT_NUM, self.triples, norm=True
        ) 
        self.adj = self.adj.to(self.device)

    def init_model(self):
        img_dim = self.img_features.shape[1]
        char_dim = (
            self.char_features.shape[1] if self.char_features is not None else 100
        )

        self.multimodal_encoder = IBMultiModal(
            args=self.args,
            ent_num=self.ENT_NUM,
            img_feature_dim=img_dim,
            char_feature_dim=char_dim,
            use_project_head=self.args.use_project_head,
        ).to(self.device)


        self.params = [{"params": list(self.multimodal_encoder.parameters())}]
        total_params = sum(
            p.numel()
            for p in self.multimodal_encoder.parameters()
            if p.requires_grad
        )

        self.optimizer = optim.AdamW(
            self.params, lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        if self.args.use_sheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=self.args.sheduler_gamma
            )
        elif self.args.use_sheduler_cos:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.num_training_steps,
            )
        ms_alpha = self.args.ms_alpha
        ms_beta = self.args.ms_beta
        ms_base = self.args.ms_base
        self.criterion_ms = MsLoss(
            device=self.device, thresh=ms_base, scale_pos=ms_alpha, scale_neg=ms_beta
        )
        self.criterion_nce = InfoNCE_loss(device=self.device, temperature=self.args.tau)

    def semi_supervised_learning(self):
        with torch.no_grad():
            gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = (
                self.multimodal_encoder(
                    self.input_idx,
                    self.adj,
                    self.img_features,
                    self.rel_features,
                    self.att_features,
                    self.name_features,
                    self.char_features,
                )
            )

            final_emb = F.normalize(joint_emb)
            distance_list = []
            for i in np.arange(0, len(self.left_non_train), 1000):
                d = pairwise_distances(
                    final_emb[self.left_non_train[i : i + 1000]],
                    final_emb[self.right_non_train],
                )
                distance_list.append(d)
            distance = torch.cat(distance_list, dim=0)
            preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
            preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
            del distance_list, distance, final_emb
            del gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb
        return preds_l, preds_r

    def train(self):
        print("model config is:")
        pprint(self.args, indent=2)
        print("[start training...] ")
        t_total = time.time()
        new_links = []
        epoch_KE, epoch_CG = 0, 0

        bsize = self.args.bsize
        device = self.device

        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)

        use_graph_vib = self.args.use_graph_vib
        use_attr_vib = self.args.use_attr_vib
        use_img_vib = self.args.use_img_vib
        use_rel_vib = self.args.use_rel_vib
        use_joint_vib = self.args.use_joint_vib
        use_bce = self.args.use_bce
        use_bce_new = self.args.use_bce_new
        use_icl = self.args.use_icl
        use_ms = self.args.use_ms
        use_nt = self.args.use_nt
        use_nce = self.args.use_nce
        use_nce_new = self.args.use_nce_new

        joint_use_icl = self.args.joint_use_icl
        joint_use_bce = self.args.joint_use_bce
        joint_use_ms = self.args.joint_use_ms

        graph_use_icl = self.args.graph_use_icl
        graph_use_bce = self.args.graph_use_bce
        graph_use_ms = self.args.graph_use_ms

        img_use_icl = self.args.img_use_icl
        img_use_bce = self.args.img_use_bce
        img_use_ms = self.args.img_use_ms

        attr_use_icl = self.args.attr_use_icl
        attr_use_bce = self.args.attr_use_bce
        attr_use_ms = self.args.attr_use_ms

        rel_use_icl = self.args.rel_use_icl
        rel_use_bce = self.args.rel_use_bce
        rel_use_ms = self.args.rel_use_ms

        Beta_g = self.args.Beta_g
        Beta_i = self.args.Beta_i
        Beta_r = self.args.Beta_r
        Beta_a = self.args.Beta_a

        for epoch in range(self.args.epochs):
            t_epoch = time.time()
            self.multimodal_encoder.train()
            self.optimizer.zero_grad()

            gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = (
                self.multimodal_encoder(
                    self.input_idx,
                    self.adj,
                    self.img_features,
                    self.rel_features,
                    self.att_features,
                    self.name_features,
                    self.char_features,
                )
            )
            epoch_CG += 1
            np.random.shuffle(self.train_ill)
            print("train_ill length:", len(self.train_ill))
            for si in np.arange(0, self.train_ill.shape[0], bsize):
                loss_all = 0
                Beta = 0.001
                print("[epoch {:d}] ".format(epoch), end="")
                loss_list = []
                if self.args.w_gcn:
                    if use_graph_vib:

                        gph_bce_loss = self.criterion_nce(
                            gph_emb, self.train_ill[si : si + bsize]
                        )
                        gph_kld_loss = self.multimodal_encoder.kld_loss
                        loss_G = gph_bce_loss + Beta_g * gph_kld_loss
                        print(
                            " gph_bce_loss: {:f}, gph_kld_loss: {:f}, Beta_g: {:f}".format(
                                gph_bce_loss, gph_kld_loss, Beta_g
                            ),
                            end="",
                        )
                    loss_list.append(loss_G)

                if self.args.w_img:
                    if use_img_vib:

                        img_bce_loss = self.criterion_nce(
                            img_emb, self.train_ill[si : si + bsize]
                        )

                        img_kld_loss = self.multimodal_encoder.img_kld_loss
                        loss_I = img_bce_loss + Beta_i * img_kld_loss
                        print(
                            " img_bce_loss: {:f}, img_kld_loss: {:f}, Beta_i: {:f}".format(
                                img_bce_loss, img_kld_loss, Beta_i
                            ),
                            end="",
                        )
                    loss_list.append(loss_I)

                if self.args.w_rel:
                    if use_rel_vib:
                        rel_bce_loss = self.criterion_nce(
                            rel_emb, self.train_ill[si : si + bsize]
                        )
                        rel_kld_loss = self.multimodal_encoder.rel_kld_loss
                        loss_R = rel_bce_loss + Beta_r * rel_kld_loss
                        print(
                            " rel_bce_loss: {:f}, rel_kld_loss: {:f}, Beta_r: {:f}".format(
                                rel_bce_loss, rel_kld_loss, Beta_r
                            ),
                            end="",
                        )
                    loss_list.append(loss_R)

                if self.args.w_attr:
                    if use_attr_vib:
                        attr_bce_loss = self.criterion_nce(
                            att_emb, self.train_ill[si : si + bsize]
                        )
                        attr_kld_loss = self.multimodal_encoder.attr_kld_loss
                        loss_A = attr_bce_loss + Beta_a * attr_kld_loss
                        print(
                            " attr_bce_loss: {:f}, attr_kld_loss: {:f}, Beta_a: {:f}".format(
                                attr_bce_loss, attr_kld_loss, Beta_a
                            ),
                            end="",
                        )
                    loss_list.append(loss_A)

                if use_joint_vib:
                    pass
                else:
                    joint_ms_loss = self.criterion_ms(
                        joint_emb, self.train_ill[si : si + bsize]
                    )
                    loss_J = joint_ms_loss * self.args.joint_beta
                    print(
                        " joint_ms_loss: {:f}, joint_beta: {:f}".format(
                            joint_ms_loss, self.args.joint_beta
                        ),
                        end="",
                    )
                    loss_list.append(loss_J)
                loss_all = sum(loss_list)

                print(" loss_all: {:f},".format(loss_all))
                loss_all.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.multimodal_encoder.parameters(),
                    max_norm=0.1,
                    norm_type=2,
                )

            self.optimizer.step()
            if self.args.use_sheduler and epoch > 400:
                self.scheduler.step()
            if self.args.use_sheduler_cos:
                self.scheduler.step()
            if (
                epoch >= self.args.il_start
                and (epoch + 1) % self.args.semi_learn_step == 0
                and self.args.il
            ):
                preds_l, preds_r = self.semi_supervised_learning()
                if (epoch + 1) % (
                    self.args.semi_learn_step * 10
                ) == self.args.semi_learn_step:
                    new_links = [
                        (self.left_non_train[i], self.right_non_train[p])
                        for i, p in enumerate(preds_l)
                        if preds_r[p] == i
                    ]
                else:
                    new_links = [
                        (self.left_non_train[i], self.right_non_train[p])
                        for i, p in enumerate(preds_l)
                        if (preds_r[p] == i)
                        and (
                            (self.left_non_train[i], self.right_non_train[p])
                            in new_links
                        )
                    ]
                print(
                    "[epoch %d] #links in candidate set: %d" % (epoch, len(new_links))
                )

            if (
                epoch >= self.args.il_start
                and (epoch + 1) % (self.args.semi_learn_step * 10) == 0
                and len(new_links) != 0
                and self.args.il
            ):
                new_links_elect = new_links
                print("\n#new_links_elect:", len(new_links_elect))

                self.train_ill = np.vstack((self.train_ill, np.array(new_links_elect)))
                print("train_ill.shape:", self.train_ill.shape)

                num_true = len([nl for nl in new_links_elect if nl in self.test_ill_])
                print("#true_links: %d" % num_true)
                print(
                    "true link ratio: %.1f%%" % (100 * num_true / len(new_links_elect))
                )
                for nl in new_links_elect:
                    self.left_non_train.remove(nl[0])
                    self.right_non_train.remove(nl[1])
                print(
                    "#entity not in train set: %d (left) %d (right)"
                    % (len(self.left_non_train), len(self.right_non_train))
                )

                new_links = []
            if (epoch + 1) % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                self.test(epoch)
            del joint_emb, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb

        print("[optimization finished!]")
        print(
            "best epoch is {}, hits@1 hits@10 MRR MR is: {}\n".format(
                self.best_epoch, self.best_data_list
            )
        )
        print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

    def test(self, epoch):
        with torch.no_grad():
            t_test = time.time()
            self.multimodal_encoder.eval()

            gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = (
                self.multimodal_encoder(
                    self.input_idx,
                    self.adj,
                    self.img_features,
                    self.rel_features,
                    self.att_features,
                    self.name_features,
                    self.char_features,
                )
            )

            final_emb = F.normalize(joint_emb)
            top_k = [1, 10, 50]
            if "100" in self.args.file_dir:
                Lvec = final_emb[self.test_left].cpu().data.numpy()
                Rvec = final_emb[self.test_right].cpu().data.numpy()
                acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = multi_get_hits(
                    Lvec, Rvec, top_k=top_k, args=self.args
                )
                del final_emb
                gc.collect()
            else:
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = (
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                if self.args.dist == 2:
                    distance = pairwise_distances(
                        final_emb[self.test_left], final_emb[self.test_right]
                    )
                elif self.args.dist == 1:
                    import scipy.spatial as T

                    distance = torch.FloatTensor(
                        T.distance.cdist(
                            final_emb[self.test_left].cpu().data.numpy(),
                            final_emb[self.test_right].cpu().data.numpy(),
                            metric="cityblock",
                        )
                    )
                else:
                    raise NotImplementedError

                if self.args.csls is True:
                    distance = 1 - csls_sim(1 - distance, self.args.csls_k)

                to_write = []
                test_left_np = self.test_left.cpu().numpy()
                test_right_np = self.test_right.cpu().numpy()
                to_write.append(
                    [
                        "idx",
                        "rank",
                        "query_id",
                        "gt_id",
                        "ret1",
                        "ret2",
                        "ret3",
                        "v1",
                        "v2",
                        "v3",
                    ]
                )

                for idx in range(self.test_left.shape[0]):
                    values, indices = torch.sort(distance[idx, :], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_l2r += rank + 1
                    mrr_l2r += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_l2r[i] += 1

                    indices = indices.cpu().numpy()
                    to_write.append(
                        [
                            idx,
                            rank,
                            test_left_np[idx],
                            test_right_np[idx],
                            test_right_np[indices[0]],
                            test_right_np[indices[1]],
                            test_right_np[indices[2]],
                            round(values[0].item(), 4),
                            round(values[1].item(), 4),
                            round(values[2].item(), 4),
                        ]
                    )

                for idx in range(self.test_right.shape[0]):
                    _, indices = torch.sort(distance[:, idx], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_r2l += rank + 1
                    mrr_r2l += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_r2l[i] += 1

                mean_l2r /= self.test_left.size(0)
                mean_r2l /= self.test_right.size(0)
                mrr_l2r /= self.test_left.size(0)
                mrr_r2l /= self.test_right.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
                    acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)
                del (
                    distance,
                    gph_emb,
                    img_emb,
                    rel_emb,
                    att_emb,
                    name_emb,
                    char_emb,
                    joint_emb,
                )
                gc.collect()
            print(
                "l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(
                    top_k, acc_l2r, mean_l2r, mrr_l2r, time.time() - t_test
                )
            )
            print(
                "r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(
                    top_k, acc_r2l, mean_r2l, mrr_r2l, time.time() - t_test
                )
            )
            if acc_l2r[0] > self.best_hit_1:
                self.best_hit_1 = acc_l2r[0]
                self.best_epoch = epoch
                self.best_data_list = [
                    acc_l2r[0],
                    acc_l2r[1],
                    mrr_l2r,
                    mean_l2r,
                ]  
                import copy

                self.best_to_write = copy.deepcopy(to_write)

            if epoch + 1 == self.args.epochs:
                pred_name = self.args.pred_name
                import csv

                save_path = os.path.join(self.args.save_path, pred_name)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                with open(os.path.join(save_path, pred_name + ".txt"), "w") as f:
                    wr = csv.writer(f, dialect="excel")
                    wr.writerows(self.best_to_write)


if __name__ == "__main__":
    model = IBMEA()
    model.train()

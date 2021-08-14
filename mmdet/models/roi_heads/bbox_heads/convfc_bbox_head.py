import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
import numpy as np
from torch.nn import functional as F
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)

        # for self.fusion_modules_e
        n_relations = 16
        appearance_feature_dim = 1024
        d_f = 128
        self.key_feature_dim = int(appearance_feature_dim / n_relations)
        self.geo_feature_dim = int(appearance_feature_dim / n_relations)
        self.appearance_feature_dim = appearance_feature_dim
        self.relation_module = RelationModule(n_relations=n_relations,
                                              appearance_feature_dim=self.appearance_feature_dim,
                                              key_feature_dim=self.key_feature_dim,
                                              geo_feature_dim=self.geo_feature_dim)

        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def SelfAttentionBlock(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 16)

        theta_x = self.theta(x).view(batch_size, out_channels // 16)
        theta_x = theta_x.permute(1, 0)
        phi_x = self.phi(x).view(batch_size, out_channels // 16)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 16)
        W_y = self.W(y)
        z = W_y + x
        return z

    def CrossAttentionBlock(self, x):
        batch_size = x[0].size(0)
        out_channels = x[0].size(1)

        g_x = self.g2(x[0]).view(batch_size, out_channels // 16)

        theta_x = self.theta2(x[1]).view(batch_size, out_channels // 16)
        theta_x = theta_x.permute(1,0)
        phi_x = self.phi2(x[1]).view(batch_size, out_channels // 16)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 16)
        W_y = self.W2(y)
        z = W_y + x[0]
        return z

    def PositionalEmbedding(self, f_g, dim_g=64, wave_len=1000):
        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        delta_x = cx - cx.view(1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(1, -1))
        delta_h = torch.log(h / h.view(1, -1))
        size = delta_h.size()

        delta_x = delta_x.view(size[0], size[1], 1)
        delta_y = delta_y.view(size[0], size[1], 1)
        delta_w = delta_w.view(size[0], size[1], 1)
        delta_h = delta_h.view(size[0], size[1], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

        feat_range = torch.arange(dim_g / 8).to("cuda:3")  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(size[0], size[1], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat.to("cuda:3") * dim_mat
        mul_mat = mul_mat.view(size[0], size[1], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)

        return embedding

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x[0] = conv(x[0])
                x[1] = conv(x[1])

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x[0] = self.avg_pool(x[0])
                x[1] = self.avg_pool(x[1])

            prim_feats = x[0].flatten(1)
            aux_feats = x[1].flatten(1)
            prim_roi = x[2]
            aux_roi = x[3]


            for fc in self.shared_fcs:
                prim_feats = self.relu(fc(prim_feats))
                aux_feats = self.relu(fc(aux_feats))

                ### fusion e
                feats = prim_feats.clone()
                sizea = int(prim_feats.size()[0]/4)
                for i in [0, sizea,int(sizea*2),int(sizea*3)]:

                    position_embedding = self.PositionalEmbedding(torch.cat((prim_roi[i:i+sizea],aux_roi[i:i+sizea]),0)[:,1:], dim_g=self.geo_feature_dim)
                    feats_i = self.relation_module([torch.cat((prim_feats[i:i+sizea],aux_feats[i:i+sizea]),0), position_embedding])
                    feats[i:i+sizea] = feats_i[:int(feats_i.size()[0]/2)]

                prim_feats = feats.clone()

            # separate branches
        x_cls = prim_feats
        x_reg = prim_feats

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)



class RelationModule(nn.Module):
    def __init__(self,n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationModule, self).__init__()
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))
    def forward(self, input_data ):
        f_a, position_embedding = input_data
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat+f_a

class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f_a, position_embedding):
        N,_ = f_a.size()

        position_embedding = position_embedding.view(-1,self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N,1,self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 )
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N,N)
        w_a = scaled_dot.view(N,N)

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N,N,1)
        w_v = w_v.view(N,1,-1)

        output = w_mn*w_v

        output = torch.sum(output,-2)
        return output

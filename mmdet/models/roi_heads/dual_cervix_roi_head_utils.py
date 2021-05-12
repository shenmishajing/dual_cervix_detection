import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from mmcv.cnn import normal_init


def build_proposaloffset(offset_cfg):
    support_offset_type = ("ProposalOffsetXY", "ProposalOffsetXYXY", "ProposalOffsetXYWH")
    offset_type = offset_cfg.pop("type")

    if offset_type in support_offset_type:
        proposal_offset = eval("{}(**offset_cfg)".format(offset_type))
    else:
        raise "offset_type = {} is not support".format(offset_type)

    return proposal_offset


class ProposalOffset(nn.Module, metaclass=ABCMeta):


    def __init__(self, in_channels, out_channels, roi_feat_area, gamma=0.1):
        super(ProposalOffset, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.roi_feat_area = roi_feat_area
        self.gamma = gamma
        self.init_layers()

    @abstractmethod
    def init_layers(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, x):
        # x (roi pooling feature) shape = [B * 512, 256, 7, 7] 
        pass 

    @abstractmethod
    def apply_offset(self, rois_coord, offset):
        """
            rois shape = [B * 512, 1(img_idx) + 4(coord)]
            rois_coord shape = [B * 512, 4]
            offset shape = [B * 512, 2]
        """ 
        pass


class ProposalOffsetXY(ProposalOffset):


    def init_layers(self):
        self.offset = nn.ModuleList([
                nn.Linear(self.roi_feat_area * self.in_channels, self.out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_channels, self.out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_channels, 2)
            ])


    def init_weights(self):
        for m in self.offset:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


    def forward(self, x):
        x = x.view(-1, self.roi_feat_area * self.in_channels)
        for m in self.offset:
            x = m(x)
        
        return x


    def apply_offset(self, rois_coord, offset):
        rois_wh = rois_coord[:, 2:4] - rois_coord[:, 0:2]

        rois_coord = torch.cat([
            rois_coord[:, 0:2] + gamma * offset * rois_wh,
            rois_coord[:, 2:4] + gamma * offset * rois_wh,
        ], dim=-1)

        return rois_coord


class ProposalOffsetXYXY(ProposalOffset):


    def init_layers(self):
        self.offset1 = nn.ModuleList([
            nn.Linear(self.roi_feat_area * self.in_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, 2)
        ])
        self.offset2 = nn.ModuleList([
            nn.Linear(self.roi_feat_area * self.in_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, 2)
        ])


    def init_weights(self):
        for m in self.offset1:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
        for m in self.offset2:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


    def forward(self, x):
        x = x.view(-1, self.roi_feat_area * self.in_channels)
        x1 = x
        x2 = x
        for m in self.offset1:
            x1 = m(x1) 
        for m in self.offset2:
            x2 = m(x2)
        x = torch.cat([x1, x2], dim=-1)

        return x


    def apply_offset(self, rois_coord, offset):
        rois_wh = rois_coord[:, 2:4] - rois_coord[:, 0:2]
        rois_wh = torch.cat([rois_wh, rois_wh], dim=-1)
        rois_coord = rois_coord + gamma * offset * rois_wh

        return rois_coord


class ProposalOffsetXYWH(ProposalOffset):


    def init_layers(self):
        self.offset1 = nn.ModuleList([
            nn.Linear(self.roi_feat_area * self.in_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, 2)
        ])
        self.offset2 = nn.ModuleList([
            nn.Linear(self.roi_feat_area * self.in_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, 2)
        ])


    def init_weights(self):
        for m in self.offset1:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
        for m in self.offset2:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


    def forward(self, x):
        x = x.view(-1, self.roi_feat_area * self.in_channels)
        x1 = x
        x2 = x
        for m in self.offset1:
            x1 = m(x1) 
        for m in self.offset2:
            x2 = m(x2)
        x = torch.cat([x1, x2], dim=-1)

        return x


    def apply_offset(self, rois_coord, offset):
        offset_xy = offset[:, 0:2]
        offset_wh = offset[:, 2:4]
        rois_xy = rois_coord[:, 0:2] + gamma * offset_xy * rois_wh
        rois_wh = rois_wh * torch.exp(offset_wh)
        rois_coord = torch.cat([rois_xy, rois_xy + rois_wh], dim=-1)

        return rois_coord

#-----------------------------------------------------------------------

class PrimAuxAttention(nn.Module):
    #TODO augment feature mutually

    def __init__(self, in_channels, out_channels, num_levels=5, shared=False):
        #TODO add support when in_channels is a list 
        super(PrimAuxAttention, self).__init__()
        assert isinstance(in_channels, int), "type of in_channels must be int"
        assert isinstance(out_channels, int), "type of out_channels must be int"

        self.shared = shared
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_layers()


    def init_layers(self):
        if self.shared:
            self.att = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.att_list = nn.ModuleList([
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
                for _ in range(self.num_levels)
            ]) 


    def init_weights(self):
        if self.shared:
            normal_init(self.att, std=0.01)
        else:
            for i in range(self.num_levels):
                normal_init(self.att_list[i], std=0.01)


    def forward(self, prim_feats, aux_feats):
        """using one feature to augment another feature

        Args:
            prim_feats (list[tensor]): features in different levels 
            aux_feats (list[tensor]): features in different levels

        Returns:
            list[tensor]: features augmented
        """
        if self.shared:
            aug_feats = [
                (self.att(aux_feats[i]).sigmoid() + 1) * prim_feats[i]
                for i in range(self.num_levels)
            ]
        else:
            aug_feats = [
                (self.att_list[i](aux_feats[i]).sigmoid() + 1) * prim_feats[i]
                for i in range(self.num_levels)
            ]
        
        return aug_feats


#-----------------------------------------------------------------------

def build_fpnfeaturefuser(fpn_fuser_cfg):
    support_type = ("FPNFeatureFuser", "DualFPNFeatureFuser")
    fpn_fuser_type = fpn_fuser_cfg.pop("type")

    if fpn_fuser_type in support_type:
        fpn_fuser = eval("{}(**fpn_fuser_cfg)".format(fpn_fuser_type))
    else:
        raise "fpn_fuser_type = {} is not support.".format(fpn_fuser_type)
    
    return fpn_fuser


class FPNFeatureFuser(nn.Module):


    def __init__(self, roi_feat_size, num_levels, in_channels=None, out_channels=None, with_conv=False, fuse_type=None, naive_fuse=True, finest_scale=56):
        super(FPNFeatureFuser, self).__init__()
        #! None is sum
        assert fuse_type in (None, "cat"), "fuse_type is not in (None, 'cat)"
        if fuse_type == "cat":
            assert (in_channels is not None and out_channels is not None), "when fuse_type = cat, in_channels and out_channels must be not None"

        self.num_levels = num_levels
        self.output_size = (roi_feat_size, roi_feat_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_conv = with_conv
        self.fuse_type = fuse_type
        self.naive_fuse_flag = naive_fuse
        self.finest_scale = finest_scale
        self.init_layers()

    
    def init_layers(self):
        self.pool_list = nn.ModuleList([
            nn.FractionalMaxPool2d(3, output_size=self.output_size)
            for _ in range(self.num_levels)
        ])

        if self.fuse_type == "cat" and self.with_conv:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1)
            

    def init_weights(self):
        if self.fuse_type == "cat" and self.with_conv:
            normal_init(self.conv, std=0.01)


    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls


    def naive_fuse(self, prim_bbox_feats, aux_global_feats):
        tmp = self.pool_list[0](aux_global_feats[0])
        for i in range(1, self.num_levels):
            tmp += self.pool_list[i](aux_global_feats[i])
        tmp /= self.num_levels
        
        # prim_bbox_feats 按图片的顺序放置 [B * 512, 256, 7, 7]
        # [512(img_1), 512(img_2), ..., 512(img_B)]
        n1 = prim_bbox_feats.shape[0]
        n2 = tmp.shape[0]
        aux_global_feats_repeated = torch.repeat_interleave(tmp, n1 // n2, dim=0)
        if self.fuse_type is None:
            out = prim_bbox_feats + aux_global_feats_repeated
        elif self.fuse_type == "cat":
            out = torch.cat([prim_bbox_feats, aux_global_feats_repeated], dim=1)
            if self.with_conv:
                out = self.conv(out)
            else:
                #! 没有经过conv， 输出的特征数就是512了
                pass
        
        return out


    def align_fuse(self, prim_bbox_feats, aux_global_feats, prim_rois):
        aux_feats_list = []
        for i in range(self.num_levels):
            aux_feats_list.append(
                self.pool_list[i](aux_global_feats[i])
            )
        target_lvls = self.map_roi_levels(prim_rois, self.num_levels)
        # prim_bbox_feats 按图片的顺序放置 [B * 512, 256, 7, 7]
        # [512(img_1), 512(img_2), ..., 512(img_B)]

        #! [lvl1_feats, lvl2_feats, ...] - > shape = [num_level, B, 256, 7, 7]
        aux_feats_stack = torch.stack(aux_feats_list, dim=0)
        num_imgs = aux_global_feats[0].shape[0]
        num_proposals_all_imgs = prim_bbox_feats.shape[0]
        repeat_times = num_proposals_all_imgs // num_imgs
        gather_aux_feats = []
        for i in range(num_imgs):
            idx = prim_rois[i * repeat_times: (i + 1) * repeat_times, 0].long()
            aux_feats_gathered = aux_feats_stack[idx, i]
            gather_aux_feats.append(aux_feats_gathered)
        aux_feats = torch.cat(gather_aux_feats, dim=0)

        if self.fuse_type is None:
            out = prim_bbox_feats + aux_feats
        elif self.fuse_type == "cat":
            out = torch.cat([prim_bbox_feats, aux_feats], dim=1)
            if self.with_conv:
                out = self.conv(out)
            else:
                #! 没有经过conv， 输出的特征数就是512了
                pass  

        return out


    def forward(self, prim_bbox_feats, prim_global_feats, aux_global_feats, prim_rois):
        if self.naive_fuse_flag:
            out = self.naive_fuse(prim_bbox_feats, aux_global_feats)
        else:
            out = self.align_fuse(prim_bbox_feats, aux_global_feats, prim_rois)

        return out


class DualFPNFeatureFuser(nn.Module):


    def __init__(self, roi_feat_size, num_levels, in_channels=None, out_channels=None, with_conv=False, fuse_type=None, naive_fuse=True, finest_scale=56):
        super(DualFPNFeatureFuser, self).__init__()
        assert naive_fuse , "align fuse is not support, only support naive fuse in DualFPNFeatureFuser"
        #! None is sum
        assert fuse_type in (None, "cat"), "fuse_type is not in (None, 'cat)"
        if fuse_type == "cat":
            assert (in_channels is not None and out_channels is not None), "when fuse_type = cat, in_channels and out_channels must be not None"

        self.num_levels = num_levels
        self.output_size = (roi_feat_size, roi_feat_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_conv = with_conv
        self.fuse_type = fuse_type
        self.naive_fuse_flag = naive_fuse
        self.finest_scale = finest_scale
        self.init_layers()

    
    def init_layers(self):
        self.prim_pool_list = nn.ModuleList([
            nn.FractionalMaxPool2d(3, output_size=self.output_size)
            for _ in range(self.num_levels)
        ])

        self.aux_pool_list = nn.ModuleList([
            nn.FractionalMaxPool2d(3, output_size=self.output_size)
            for _ in range(self.num_levels)
        ])

        if self.fuse_type == "cat" and self.with_conv:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1)
            

    def init_weights(self):
        if self.fuse_type == "cat" and self.with_conv:
            normal_init(self.conv, std=0.01)


    def naive_fuse(self, prim_bbox_feats, prim_global_feats, aux_global_feats):
        aux_tmp = self.aux_pool_list[0](aux_global_feats[0])
        for i in range(1, self.num_levels):
            aux_tmp += self.aux_pool_list[i](aux_global_feats[i])
        aux_tmp /= self.num_levels
        
        prim_tmp = self.prim_pool_list[0](prim_global_feats[0])
        for i in range(1, self.num_levels):
            prim_tmp += self.prim_pool_list[i](prim_global_feats[i])
        prim_tmp /= self.num_levels

        # prim_bbox_feats 按图片的顺序放置 [B * 512, 256, 7, 7]
        # [512(img_1), 512(img_2), ..., 512(img_B)]
        n1 = prim_bbox_feats.shape[0]
        n2 = aux_tmp.shape[0]
        aux_global_feats_repeated = torch.repeat_interleave(aux_tmp, n1 // n2, dim=0)
        prim_global_feats_repeated = torch.repeat_interleave(prim_tmp, n1 // n2, dim=0)
        if self.fuse_type is None:
            out = prim_bbox_feats + aux_global_feats_repeated + prim_global_feats_repeated
        elif self.fuse_type == "cat":
            out = torch.cat([prim_bbox_feats, aux_global_feats_repeated, prim_global_feats_repeated], dim=1)
            if self.with_conv:
                out = self.conv(out)
            else:
                #! 没有经过conv， 输出的特征数就是756了
                pass
        
        return out


    def forward(self, prim_bbox_feats, prim_global_feats, aux_global_feats, prim_rois):
        if self.naive_fuse_flag:
            out = self.naive_fuse(prim_bbox_feats, prim_global_feats, aux_global_feats)
        else:
            raise "align fuse is not support"

        return out

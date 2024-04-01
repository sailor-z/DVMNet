import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, random_rotations
from utils import patchify, unpatchify, log_optimal_transport
from einops import rearrange
from functools import partial
import math

torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
sys.path.append("./croco/")

from models.croco import CroCoNet
from models.blocks import DecoderBlock, DecoderBlock_Monocular
from models.pos_embed import RoPE2D

class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

class DVMNet(nn.Module):
    def __init__(self, transport=None, mask="both"):
        super().__init__()
        self.spatial_size = 14
        self.patch_size = 16
        self.in_channels = 768
        self.embed_channels = 896
        self.decode_channels = 910

        self.n_heads = 8
        self.cross_depth = 3
        self.de_depth = 3
        self.volume_channels = self.decode_channels // self.spatial_size - 1

        ckpt = torch.load('./croco/CroCo_V2_ViTBase_BaseDecoder.pth', 'cpu')

        self.backbone = CroCoNet(**ckpt.get('croco_kwargs',{}))
        self.backbone.load_state_dict(ckpt['model'], strict=True)

        self.position_getter = PositionGetter()

        self.cross_att_blocks_src = nn.ModuleList([
            DecoderBlock(self.in_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True, \
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for i in range(self.cross_depth)])

        self.cross_att_blocks_tgt = nn.ModuleList([
            DecoderBlock(self.in_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True, \
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for i in range(self.cross_depth)])

        self.encoder_embed = nn.Linear(self.in_channels, self.decode_channels)

        self.dec_blocks = nn.ModuleList([
            DecoderBlock_Monocular(self.embed_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True, \
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for i in range(self.de_depth)])

        # final norm layer
        self.prediction_head = nn.Sequential(
             nn.LayerNorm(self.embed_channels),
             nn.Linear(self.embed_channels, self.patch_size**2 *4, bias=True)
        )

        self.svd_head = SVDHead_Mask(transport, mask)

        self.xyz_src = (self.point_coordinates(14, 14, 14) - 6.5) / 6.5
        self.xyz_tgt = (self.point_coordinates(14, 14, 14) - 6.5) / 6.5

    def positional_encoding(self, x, embedding):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        embed = (x[..., None] * embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


    def point_coordinates(self, D, H, W):
        grid_z, grid_y, grid_x = torch.meshgrid([torch.arange(D), torch.arange(H), torch.arange(W)])

        coordinates_3d = torch.stack((grid_x, grid_y, grid_z), dim=0)
        coordinates_3d = coordinates_3d.view(3, -1).float()

        return coordinates_3d

    def encoding(self, img_src, img_tgt):
        bs = img_src.shape[0]

        feat_src = self.backbone._encode_image(img_src, do_mask=False, return_all_blocks=False)[0]
        feat_tgt = self.backbone._encode_image(img_tgt, do_mask=False, return_all_blocks=False)[0]

        pos_src = self.position_getter(bs, self.spatial_size, self.spatial_size, feat_src.device)
        pos_tgt = self.position_getter(bs, self.spatial_size, self.spatial_size, feat_tgt.device)

        for idx, blk in enumerate(self.cross_att_blocks_src):
            if idx == 0:
                feat_src_, feat_tgt = blk(feat_src, feat_tgt, pos_src, pos_tgt)
            else:
                feat_src_, feat_tgt = blk(feat_src_, feat_tgt, pos_src, pos_tgt)

        for idx, blk in enumerate(self.cross_att_blocks_tgt):
            if idx == 0:
                feat_tgt_, feat_src = blk(feat_tgt, feat_src, pos_tgt, pos_src)
            else:
                feat_tgt_, feat_src = blk(feat_tgt_, feat_src, pos_tgt, pos_src)

        feat_src = self.encoder_embed(feat_src_).transpose(1, 2)
        feat_tgt = self.encoder_embed(feat_tgt_).transpose(1, 2)

        volume_src = feat_src.reshape(feat_src.shape[0], self.volume_channels+1, self.spatial_size, self.spatial_size, self.spatial_size) ##64x14x14x14
        volume_tgt = feat_tgt.reshape(feat_tgt.shape[0], self.volume_channels+1, self.spatial_size, self.spatial_size, self.spatial_size) ##64x14x14x14

        weight_src, weight_tgt = volume_src[:, self.volume_channels:], volume_tgt[:, self.volume_channels:]
        volume_src, volume_tgt = volume_src[:, :self.volume_channels], volume_tgt[:, :self.volume_channels]

        return feat_src, feat_tgt, weight_src, weight_tgt, volume_src, volume_tgt

    def decoding(self, view_volume):
        b, c, d, h, w = view_volume.shape
        feat = rearrange(view_volume, 'b c d h w -> b (c d) (h w)')
        feat = feat.transpose(1, 2)

        pos = self.position_getter(b, h, w, view_volume.device)

        for blk in self.dec_blocks:
            feat = blk(feat, pos)

        pred_img = self.prediction_head(feat)

        return feat, pred_img

    def forward(self, img_src, img_tgt):
        feat_src, feat_tgt, occupancy_src, occupancy_tgt, volume_src, volume_tgt = self.encoding(img_src, img_tgt)

        occupancy_src, occupancy_tgt = occupancy_src.flatten(2), occupancy_tgt.flatten(2)
        embed_src, embed_tgt = volume_src.flatten(2), volume_tgt.flatten(2)

        b, c, d, h, w = volume_src.shape

        xyz_src = self.xyz_src[None].expand(b, -1, -1).to(embed_src.device)
        xyz_tgt = self.xyz_tgt[None].expand(b, -1, -1).to(embed_tgt.device)

        feat_src, pred_src = self.decoding(volume_src)
        feat_tgt, pred_tgt = self.decoding(volume_tgt)

        img_mask_src = pred_src.reshape(b, h, w, -1, c)[..., -1].mean(dim=-1)
        img_mask_tgt = pred_tgt.reshape(b, h, w, -1, c)[..., -1].mean(dim=-1)

        img_mask_src = img_mask_src[:, None].expand(-1, d, -1, -1).reshape(b, 1, -1)
        img_mask_tgt = img_mask_tgt[:, None].expand(-1, d, -1, -1).reshape(b, 1, -1)

        pred_delta_R, _ = self.svd_head(embed_src, embed_tgt, xyz_src, xyz_tgt, occupancy_src, occupancy_tgt, img_mask_src, img_mask_tgt)

        return pred_src, pred_tgt, occupancy_src, occupancy_tgt, pred_delta_R, _

class DVMNet_6D(nn.Module):
    def __init__(self, transport=None, mask="both"):
        super().__init__()
        self.spatial_size = 14
        self.patch_size = 16
        self.in_channels = 768
        self.embed_channels = 896
        self.decode_channels = 910

        self.n_heads = 8
        self.cross_depth = 3
        self.de_depth = 3
        self.volume_channels = self.decode_channels // self.spatial_size - 1

        ckpt = torch.load('./croco/CroCo_V2_ViTBase_BaseDecoder.pth', 'cpu')

        self.backbone = CroCoNet(**ckpt.get('croco_kwargs',{}))
        self.backbone.load_state_dict(ckpt['model'], strict=True)

        self.position_getter = PositionGetter()

        self.cross_att_blocks_src = nn.ModuleList([
            DecoderBlock(self.in_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True, \
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for i in range(self.cross_depth)])

        self.cross_att_blocks_tgt = nn.ModuleList([
            DecoderBlock(self.in_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True, \
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for i in range(self.cross_depth)])

        self.encoder_embed = nn.Linear(self.in_channels, self.decode_channels)

        self.dec_blocks = nn.ModuleList([
            DecoderBlock_Monocular(self.embed_channels, num_heads=self.n_heads, mlp_ratio=4., qkv_bias=True, \
                norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=self.backbone.rope) for i in range(self.de_depth)])

        # final norm layer
        self.prediction_head = nn.Sequential(
             nn.LayerNorm(self.embed_channels),
             nn.Linear(self.embed_channels, self.patch_size**2 *4, bias=True)
        )

        self.svd_head = SVDHead_Mask(transport, mask)

        self.xyz_src = (self.point_coordinates(14, 14, 14) - 6.5) / 6.5
        self.xyz_tgt = (self.point_coordinates(14, 14, 14) - 6.5) / 6.5

        ## crop_params
        self.num_pe_bases = 8
        self.register_buffer("embed_crop_src", (2 ** torch.arange(self.num_pe_bases)).reshape(1, 1, -1))
        self.register_buffer("embed_crop_tgt", (2 ** torch.arange(self.num_pe_bases)).reshape(1, 1, -1))

        self.t_block1 = nn.Sequential(
            nn.Linear(2 * (self.embed_channels + 48), 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
        )
        self.t_block2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )
        self.t_regressor = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(512, 6),
        )

    def positional_encoding(self, x, embedding):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        embed = (x[..., None] * embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


    def point_coordinates(self, D, H, W):
        grid_z, grid_y, grid_x = torch.meshgrid([torch.arange(D), torch.arange(H), torch.arange(W)])

        coordinates_3d = torch.stack((grid_x, grid_y, grid_z), dim=0)
        coordinates_3d = coordinates_3d.view(3, -1).float()

        return coordinates_3d

    def encoding(self, img_src, img_tgt):
        bs = img_src.shape[0]

        feat_src = self.backbone._encode_image(img_src, do_mask=False, return_all_blocks=False)[0]
        feat_tgt = self.backbone._encode_image(img_tgt, do_mask=False, return_all_blocks=False)[0]

        pos_src = self.position_getter(bs, self.spatial_size, self.spatial_size, feat_src.device)
        pos_tgt = self.position_getter(bs, self.spatial_size, self.spatial_size, feat_tgt.device)

        for idx, blk in enumerate(self.cross_att_blocks_src):
            if idx == 0:
                feat_src_, feat_tgt = blk(feat_src, feat_tgt, pos_src, pos_tgt)
            else:
                feat_src_, feat_tgt = blk(feat_src_, feat_tgt, pos_src, pos_tgt)

        for idx, blk in enumerate(self.cross_att_blocks_tgt):
            if idx == 0:
                feat_tgt_, feat_src = blk(feat_tgt, feat_src, pos_tgt, pos_src)
            else:
                feat_tgt_, feat_src = blk(feat_tgt_, feat_src, pos_tgt, pos_src)

        feat_src = self.encoder_embed(feat_src_).transpose(1, 2)
        feat_tgt = self.encoder_embed(feat_tgt_).transpose(1, 2)

        volume_src = feat_src.reshape(feat_src.shape[0], self.volume_channels+1, self.spatial_size, self.spatial_size, self.spatial_size) ##64x14x14x14
        volume_tgt = feat_tgt.reshape(feat_tgt.shape[0], self.volume_channels+1, self.spatial_size, self.spatial_size, self.spatial_size) ##64x14x14x14

        weight_src, weight_tgt = volume_src[:, self.volume_channels:], volume_tgt[:, self.volume_channels:]
        volume_src, volume_tgt = volume_src[:, :self.volume_channels], volume_tgt[:, :self.volume_channels]

        return feat_src, feat_tgt, weight_src, weight_tgt, volume_src, volume_tgt

    def decoding(self, view_volume):
        b, c, d, h, w = view_volume.shape
        feat = rearrange(view_volume, 'b c d h w -> b (c d) (h w)')
        feat = feat.transpose(1, 2)

        pos = self.position_getter(b, h, w, view_volume.device)

        for blk in self.dec_blocks:
            feat = blk(feat, pos)

        pred_img = self.prediction_head(feat)

        return feat, pred_img

    def forward(self, img_src, img_tgt, crop_params_src, crop_params_tgt):
        feat_src, feat_tgt, occupancy_src, occupancy_tgt, volume_src, volume_tgt = self.encoding(img_src, img_tgt)

        occupancy_src, occupancy_tgt = occupancy_src.flatten(2), occupancy_tgt.flatten(2)
        embed_src, embed_tgt = volume_src.flatten(2), volume_tgt.flatten(2)

        b, c, d, h, w = volume_src.shape

        xyz_src = self.xyz_src[None].expand(b, -1, -1).to(embed_src.device)
        xyz_tgt = self.xyz_tgt[None].expand(b, -1, -1).to(embed_tgt.device)

        feat_src, pred_src = self.decoding(volume_src)
        feat_tgt, pred_tgt = self.decoding(volume_tgt)

        img_mask_src = pred_src.reshape(b, h, w, -1, c)[..., -1].mean(dim=-1)
        img_mask_tgt = pred_tgt.reshape(b, h, w, -1, c)[..., -1].mean(dim=-1)

        img_mask_src = img_mask_src[:, None].expand(-1, d, -1, -1).reshape(b, 1, -1)
        img_mask_tgt = img_mask_tgt[:, None].expand(-1, d, -1, -1).reshape(b, 1, -1)

        pred_delta_R, _ = self.svd_head(embed_src, embed_tgt, xyz_src, xyz_tgt, occupancy_src, occupancy_tgt, img_mask_src, img_mask_tgt)

        ## translation estimation
        crop_pe_src = self.positional_encoding(crop_params_src, self.embed_crop_src)
        crop_pe_tgt = self.positional_encoding(crop_params_tgt, self.embed_crop_tgt)

        feat_src = feat_src.mean(dim=1)
        feat_tgt = feat_tgt.mean(dim=1)

        feat_T = torch.cat([feat_src, crop_pe_src, feat_tgt, crop_pe_tgt], dim=-1)

        feat_T = self.t_block1(feat_T)
        feat_T = feat_T + self.t_block2(feat_T)

        pred_Ts = self.t_regressor(feat_T).reshape(-1, 6)

        bias = torch.FloatTensor([[0, 0, 1, 0, 0, 1]]).to(pred_Ts.device)
        pred_Ts += bias

        return pred_src, pred_tgt, occupancy_src, occupancy_tgt, pred_delta_R, pred_Ts

class SVDHead_Mask(nn.Module):
    def __init__(self, transport=None, mask="both"):
        super().__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.transport = transport
        self.temp_score = 0.1
        self.temp_mask = 1.0
        self.bin_score = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.skh_iters = 3
        self.log_optimal_transport = log_optimal_transport
        self.mask = mask

    def forward(self, *input):
        src_embedding = input[0]  ## BxCxL
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        src_depth_mask = input[4] #### bx1xL
        tgt_depth_mask = input[5]
        src_img_mask = input[6] #### bx1xL
        tgt_img_mask = input[7]

        batch_size = src.size(0)

        if self.transport == "dual_softmax":
            src_embedding, tgt_embedding = map(lambda feat: feat / feat.shape[1]**.5,
                                   [src_embedding, tgt_embedding])

            scores = torch.matmul(src_embedding.transpose(2, 1), tgt_embedding) / self.temp_score
            scores = F.softmax(scores, 1) * F.softmax(scores, 2)

        if self.transport == "sinkhorn":
            src_embedding, tgt_embedding = map(lambda feat: feat / feat.shape[1]**.5,
                                   [src_embedding, tgt_embedding])

            scores = torch.matmul(src_embedding.transpose(2, 1), tgt_embedding)

            log_scores = self.log_optimal_transport(scores, self.bin_score, self.skh_iters)
            scores = log_scores.exp()
            scores = scores[:, :-1, :-1]

        if self.transport == "cosine":
            src_embedding = F.normalize(src_embedding, p=2, dim=1)
            tgt_embedding = F.normalize(tgt_embedding, p=2, dim=1)

            scores = torch.matmul(src_embedding.transpose(2, 1), tgt_embedding) / self.temp_score
            scores = torch.softmax(scores, dim=2)

        else:
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1))

        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        tgt_depth_mask = torch.matmul(tgt_depth_mask, scores.transpose(2, 1))
        tgt_img_mask = torch.matmul(tgt_img_mask, scores.transpose(2, 1))

        img_mask = 0.5 * (src_img_mask + tgt_img_mask).squeeze(1)
        depth_mask = 0.5 * (src_depth_mask + tgt_depth_mask).squeeze(1)

        if self.mask == "occupancy":
            mask = torch.sigmoid(depth_mask / self.temp_mask)
            mask_min = mask.min(dim=-1, keepdim=True)[0]
            mask_max = mask.max(dim=-1, keepdim=True)[0]
            mask = (mask - mask_min) / (mask_max - mask_min).clamp(min=1e-6)
        elif self.mask == "image":
            mask = torch.sigmoid(img_mask / self.temp_mask)
            mask_min = mask.min(dim=-1, keepdim=True)[0]
            mask_max = mask.max(dim=-1, keepdim=True)[0]
            mask = (mask - mask_min) / (mask_max - mask_min).clamp(min=1e-6)
        elif self.mask == "both":
            mask = torch.sigmoid(depth_mask / self.temp_mask) * torch.sigmoid(img_mask / self.temp_mask)
            mask_min = mask.min(dim=-1, keepdim=True)[0]
            mask_max = mask.max(dim=-1, keepdim=True)[0]
            mask = (mask - mask_min) / (mask_max - mask_min).clamp(min=1e-6)
        elif self.mask == "none":
            mask = img_mask.new_ones(img_mask.shape)
        else:
            raise RuntimeError("Unsupported masking process")

        mask_diag = torch.stack([torch.diag(mask[i]) for i in range(batch_size)], dim=0)

        H = torch.matmul(torch.matmul(src_centered, mask_diag), src_corr_centered.transpose(2, 1))

        R = []

        u, s, v = torch.svd(H)
        r = torch.matmul(v, u.transpose(2, 1).contiguous())
        r_det = torch.det(r)
        for i in range(src.size(0)):
            if r_det[i] < 0:
                R.append(torch.matmul(torch.matmul(v[i], self.reflect), u[i].transpose(1, 0).contiguous()))
            else:
                R.append(r[i])

        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)

class MaskedMSE(torch.nn.Module):

    def __init__(self, norm_pix_loss=False, masked=True, patch_size=16):
        """
            norm_pix_loss: normalize each patch by their pixel mean and variance
            masked: compute loss over the masked patches only
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked
        self.patch_size = patch_size

    def forward(self, pred, target, mask):
        pred = unpatchify(pred, patch_size=self.patch_size, channels=4)
        pred_img = pred[:, :3]
        pred_mask = pred[:, 3]

        if self.norm_pix_loss:
            with torch.no_grad():
                target = patchify(target, patch_size=self.patch_size)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5
                target = unpatchify(target, patch_size=self.patch_size, channels=3)

        loss_img = (pred_img - target.detach()) ** 2
        loss_img = loss_img.sum(dim=1).flatten(1)

        mask = mask.squeeze(1).flatten(1)

        if self.masked:
            loss_img = (loss_img * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)  # mean loss on foregound pixels
        else:
            loss_img = loss_img.mean(dim=-1)  ### B

        loss_mask = F.binary_cross_entropy_with_logits(pred_mask.flatten(1), mask.detach(), reduction='none').mean(dim=-1)

        return loss_img + loss_mask

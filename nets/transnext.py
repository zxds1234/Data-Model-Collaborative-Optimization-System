import os
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from timm.models.registry import register_model
# from mmcv.runner.checkpoint import load_checkpoint
from timm.models.layers import DropPath, to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
import numpy as np



class PatchEmbed(nn.Module):


    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=dict(type='BN2d'),
                 act_cfg=None,):
        super().__init__()
        self.proj = ConvModule(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            norm_cfg=norm_layer,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.proj(x)

class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio+3,
                           stride=sr_ratio,
                           padding=(sr_ratio+3)//2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None,),)
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C//self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], 
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module): ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim, 
                       dim//reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'),),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):


        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            bias = None

        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias)
        
        return x.reshape(B, C, H, W)

class HybridTokenMixer(nn.Module):
    def __init__(self, 
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)
        
        inner_dim = max(16, dim//reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x ## STE
        return x

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i]//2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)
            
    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class Mlp(nn.Module):  ### MS-FFN


    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, 
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class Block(nn.Module):


    def __init__(self,
                 dim=64,
                 kernel_size=3,
                 sr_ratio=1,
                 num_groups=2,
                 num_heads=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 drop_path=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):

        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = HybridTokenMixer(dim,
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg,
                       drop=drop,)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.layer_scale_1(
                self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x

def basic_blocks(dim,
                 index,
                 layers,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=0,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):

    blocks = nn.ModuleList()
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            Block(
                dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop=drop_rate,
                drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=grad_checkpoint,
            ))
    return blocks

class TransXNet(nn.Module):


    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [3, 3, 9, 3],
                         'embed_dims': [48, 96, 224, 448],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 2, 2],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 4, 8],
                         'mlp_ratios': [4, 4, 4, 4],
                         'layer_scale_init_value': 1e-5,}),

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [4, 4, 12, 4],
                         'embed_dims': [64, 128, 320, 512],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 3, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratios': [6, 6, 4, 4],
                         'layer_scale_init_value': 1e-5,}),

        **dict.fromkeys(['b', 'base', 'B'],
                        {'layers': [4, 4, 21, 4],
                         'embed_dims': [76, 152, 336, 672],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 4, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mlp_ratios': [8, 8, 4, 4],
                         'layer_scale_init_value': 1e-5,}),}

    def __init__(self,
                 image_size=224,
                 arch='tiny',
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=3,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,
                 checkpoint_stage=[0] * 4,
                 num_classes=1000,
                 fork_feat=True,
                 start_level=0,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.channel = [48,96,224,448]
        self.fork_feat = fork_feat
        self.grad_checkpoint = grad_checkpoint
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        mlp_ratios = arch['mlp_ratios'] if 'mlp_ratios' in arch else [4, 4, 4, 4]
        layer_scale_init_value = arch['layer_scale_init_value'] if 'layer_scale_init_value' in arch else 1e-5

        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        self.relative_pos_enc = []
        self.pos_enc_record = []
        image_size = to_2tuple(image_size)
        image_size = [math.ceil(image_size[0]/in_stride),
                      math.ceil(image_size[1]/in_stride)]
        for i in range(4):
            num_patches = image_size[0]*image_size[1]
            sr_patches = math.ceil(
                image_size[0]/sr_ratio[i])*math.ceil(image_size[1]/sr_ratio[i])
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches), requires_grad=True))
            self.pos_enc_record.append([image_size[0], image_size[1], 
                                        math.ceil(image_size[0]/sr_ratio[i]), 
                                        math.ceil(image_size[1]/sr_ratio[i]),])
            image_size = [math.ceil(image_size[0]/2),
                          math.ceil(image_size[1]/2)]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                kernel_size=kernel_size[i],
                num_groups=num_groups[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i],)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1]))
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb < start_level:
                    layer = nn.Identity()
                else:
                    layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier
            self.classifier = nn.Sequential(
                build_norm_layer(norm_cfg, embed_dims[-1])[1],
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
            ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_model_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            # self.init_weights()
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
            self.train()

    # init for image classification
    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier[-1].out_channels = num_classes
        else:
            self.classifier = nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        pos_idx = 0
        for idx in range(len(self.network)):
            if idx in [0, 2, 4, 6]:
                for blk in self.network[idx]:
                    x = blk(x, self.relative_pos_enc[pos_idx])
                pos_idx += 1       
            else:
                x = self.network[idx](x)
            if self.fork_feat and (idx in self.out_indices):
                x_out = getattr(self, f'norm{idx}')(x)
                outs.append(x_out)

        if self.fork_feat:

            return outs

        return x

    def forward(self, x):

        x = self.forward_embeddings(x)

        x = self.forward_tokens(x)
        if self.fork_feat:

            return x
        else:

            x = self.classifier(x).flatten(1)
            return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.95,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }


def transxnet_t(pretrained=False, pretrained_cfg=None, **kwargs):
    model = TransXNet(arch='t', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        model_dict      = model.state_dict()

        pretrained_dict = torch.load('transx-s.pth.tar', map_location = 'cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    return model


def transxnet_s(pretrained=True, pretrained_cfg=None, **kwargs):
    model = TransXNet(arch='s', **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)

    if pretrained:
        model_dict      = model.state_dict()

        pretrained_dict = torch.load('model_data/transx-s.pth.tar', map_location = 'cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    return model









if __name__ == '__main__':
	model = transxnet_t()
	model.eval()
	x = torch.randn(1,3,640,640) 
	y = model(x)


	# torch.save(model.state_dict(), 'cat.pth')
	print(y[0].size())
	print(y[1].size())
	print(y[2].size())
	print(y[3].size())
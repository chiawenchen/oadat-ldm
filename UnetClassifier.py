import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torch.optim.lr_scheduler import LinearLR

def get_time_embedding(
    time_steps: torch.Tensor,
    t_emb_dim: int
) -> torch.Tensor:
    
    """ 
    Transform a scalar time-step into a vector representation of size t_emb_dim.
    
    :param time_steps: 1D tensor of size -> (Batch,)
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)
    
    :return tensor of size -> (B, t_emb_dim)
    """
    
    assert t_emb_dim%2 == 0, "time embedding must be divisible by 2."
    
    factor = 2 * torch.arange(start = 0, 
                              end = t_emb_dim//2, 
                              dtype=torch.float32, 
                              device=time_steps.device
                             ) / (t_emb_dim)
    
    factor = 10000**factor

    t_emb = time_steps[:,None] # B -> (B, 1) 
    t_emb = t_emb/factor # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1) # (B , t_emb_dim)
    
    return t_emb

class NormActConv(nn.Module):
    """
    Perform GroupNorm, Activation, and Convolution operations.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 num_groups:int = 8, 
                 kernel_size: int = 3, 
                 norm:bool = True,
                 act:bool = True
                ):
        super(NormActConv, self).__init__()
        
        # GroupNorm
        self.g_norm = nn.GroupNorm(
            num_groups,
            in_channels
        ) if norm is True else nn.Identity()
        
        # Activation
        self.act = nn.SiLU() if act is True else nn.Identity()
        
        # Convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size, 
            padding=(kernel_size - 1)//2
        )
        
    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
    
    # Making this class subscriptable for later use.
    def __getitem__(self, index):
        if index == 0:
            return self.g_norm
        elif index == 1:
            return nn.Sequential(self.act, 
                                 self.conv
                                )
        else:
            raise IndexError('Index out of range. Valid indices are 0, 1')
    
#-----------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """
    Maps the Time Embedding to the Required output Dimension.
    """
    def __init__(self, 
                 n_out:int, # Output Dimension
                 t_emb_dim:int = 128 # Time Embedding Dimension
                ):
        super(TimeEmbedding, self).__init__()
        
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, n_out)
        )
        
    def forward(self, x):
        return self.te_block(x)
    
#---------------------------------------------------------------

# class SelfAttentionBlock(nn.Module):
#     """
#     Perform GroupNorm and Multiheaded Self Attention operation.    
#     """
#     def __init__(self, 
#                  num_channels:int,
#                  num_groups:int = 8, 
#                  num_heads:int = 1,
#                  norm:bool = True
#                 ):
#         super(SelfAttentionBlock, self).__init__()
        
#         # GroupNorm
#         self.g_norm = nn.GroupNorm(
#             num_groups,
#             num_channels
#         ) if norm is True else nn.Identity()
        
#         # Self-Attention
#         self.attn = nn.MultiheadAttention(
#             num_channels,
#             num_heads, 
#             batch_first=True
#         )
        
#     def forward(self, x):
#         batch_size, channels, h, w = x.shape
#         x = x.reshape(batch_size, channels, h*w)
#         x = self.g_norm(x)
#         x = x.transpose(1, 2)
#         x, _ = self.attn(x, x, x)
#         x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
#         return x
    
# #----------------------------------------------------------------

class Downsample(nn.Module):
    """
    Perform Downsampling by the factor of k across Height and Width.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 k:int = 2, # Downsampling factor
                 use_conv:bool = True, # If Downsampling using conv-block
                 use_mpool:bool = True # If Downsampling using max-pool
                ):
        super(Downsample, self).__init__()
        
        self.use_conv = use_conv
        self.use_mpool = use_mpool
        
        # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_mpool else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            )
        ) if use_conv else nn.Identity()
        
        # Downsampling using Maxpool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_mpool else nn.Identity()
        
    def forward(self, x):
        
        if not self.use_conv:
            return self.mpool(x)
        
        if not self.use_mpool:
            return self.cv(x)
            
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class DownC(nn.Module):
    """
    Perform Down-convolution on the input using following approach.
    1. Conv + TimeEmbedding
    2. Conv
    3. Skip-connection from input x.
    4. Self-Attention
    5. Skip-Connection from 3.
    6. Downsampling
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 t_emb_dim:int = 128, # Time Embedding Dimension
                 num_layers:int=2,
                 down_sample:bool = True, # True for Downsampling
                 use_scale_shift_norm:bool = True # Use scale shift norm
                ):
        super(DownC, self).__init__()
        
        self.num_layers = num_layers
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                        out_channels
                       ) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(2 * out_channels if use_scale_shift_norm else out_channels, 
                          t_emb_dim
                         ) for _ in range(num_layers)
        ])
        
        # self.attn_block = nn.ModuleList([
        #     SelfAttentionBlock(out_channels) for _ in range(num_layers)
        # ])
        
        self.down_block =Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
        
    def forward(self, x, t_emb):
        
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            emb_out = self.te_block[i](t_emb)[:, :, None, None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.conv2[i][0], self.conv2[i][1]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                out = out_norm(out) * (1 + scale) + shift
                out = out_rest(out)
            else:
                out = out + emb_out
                out = self.conv2[i](out)
                
            out = out + self.res_block[i](resnet_input)

            # # Self Attention
            # out_attn = self.attn_block[i](out)
            # out = out + out_attn

        # Downsampling
        out = self.down_block(out)
        
        return out

class MidC(nn.Module):
    """
    Refine the features obtained from the DownC block.
    It refines the features using following operations:
    
    1. Resnet Block with Time Embedding
    2. A Series of Self-Attention + Resnet Block with Time-Embedding 
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 t_emb_dim:int = 128,
                 num_layers:int = 2,
                 use_scale_shift_norm:bool = True # Use scale shift norm
                ):
        super(MidC, self).__init__()
        
        self.num_layers = num_layers
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                        out_channels
                       ) for i in range(num_layers + 1)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                        out_channels
                       ) for _ in range(num_layers + 1)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(2 * out_channels if use_scale_shift_norm else out_channels, 
                          t_emb_dim
                         ) for _ in range(num_layers + 1)
        ])
        
        # self.attn_block = nn.ModuleList([
        #     SelfAttentionBlock(out_channels) for _ in range(num_layers)
        # ])
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers + 1)
        ])
        
    def forward(self, x, t_emb):
        out = x
        
        # First-Resnet Block
        resnet_input = out
        
        # Resnet Block
        out = self.conv1[0](out)
        emb_out = self.te_block[0](t_emb)[:, :, None, None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.conv2[0][0], self.conv2[0][1]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out = out_norm(out) * (1 + scale) + shift
            out = out_rest(out)
        else:
            out = out + emb_out
            out = self.conv2[0](out)

        out = out + self.res_block[0](resnet_input)

        
        # Sequence of Self-Attention + Resnet Blocks
        for i in range(self.num_layers):
            
            # # Self Attention
            # out_attn = self.attn_block[i](out)
            # out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.conv1[i+1](out)
            emb_out = self.te_block[i+1](t_emb)[:, :, None, None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.conv2[i+1][0], self.conv2[i+1][1]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                out = out_norm(out) * (1 + scale) + shift
                out = out_rest(out)
            else:
                out = out + emb_out
                out = self.conv2[i+1](out)
                
            out = out + self.res_block[i+1](resnet_input)
            
        return out

class UnetClassifier(LightningModule):
    def __init__(self,
                 im_channels: int = 1,  # GRAY
                 down_ch: list = [32, 64, 128, 256],
                 mid_ch: list = [256, 256, 128],
                 down_sample: list[bool] = [True, True, False],
                 t_emb_dim: int = 128,
                 num_downc_layers: int = 2,
                 num_midc_layers: int = 2,
                 use_scale_shift_norm: bool = True,
                 num_classes: int = 2,  # Binary classification
                 learning_rate: float = 1e-4,
                 num_timesteps: int = 1000,
                #  class_weights: torch.Tensor = None
                 ):
        super(UnetClassifier, self).__init__()

        self.im_channels = im_channels
        self.down_ch = down_ch
        self.mid_ch = mid_ch
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        # self.class_weights = class_weights

        # Initial Convolution
        self.cv1 = nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=3, padding=1)

        # Initial Time Embedding Projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # DownC Blocks
        self.downs = nn.ModuleList([
            DownC(
                self.down_ch[i],
                self.down_ch[i + 1],
                self.t_emb_dim,
                self.num_downc_layers,
                self.down_sample[i],
                self.use_scale_shift_norm
            ) for i in range(len(self.down_ch) - 1)
        ])

        # MidC Block
        self.mids = nn.ModuleList([
            MidC(
                self.mid_ch[i],
                self.mid_ch[i + 1],
                self.t_emb_dim,
                self.num_midc_layers,
                self.use_scale_shift_norm
            ) for i in range(len(self.mid_ch) - 1)
        ])

        # Output layer for classification
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.mid_ch[-1], self.num_classes)
        )

        # Loss function and metrics
        # if self.class_weights != None:
        #     self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # else:
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

    def forward(self, x, timesteps):
        # Initial convolution
        out = self.cv1(x)

        # Time projection
        t_emb = get_time_embedding(timesteps, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        # Pass through DownC blocks
        for down in self.downs:
            out = down(out, t_emb)

        # Pass through MidC blocks
        for mid in self.mids:
            out = mid(out, t_emb)

        # Final head for classification
        return self.head(out)

    def training_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Log metrics
        self.train_accuracy(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar=False)
        self.log("train_precision", self.train_precision, prog_bar=True)
        self.log("train_recall", self.train_recall, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Log metrics
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, prog_bar=False)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = LinearLR(optimizer, total_iters=1, last_epoch=-1)
        return [optimizer], [lr_scheduler]


    # def on_training_epoch_end(self, outputs):
    #     # Clear memory after each training epoch
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()

    # def on_validation_epoch_end(self, outputs):
    #     # Clear memory after each validation epoch
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()

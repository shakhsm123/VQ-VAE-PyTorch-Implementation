import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.K                 = config["codebook_size"]
        self.D                 = config["codebook_dim"]
        self.commitment_beta   = config["commitment_beta"]
        self.ema_decay         = config["ema_decay"]
        self.dead_code_interval = config.get("dead_code_interval", 50)
        self.steps             = 0

        self.codebook = nn.Embedding(self.K, self.D)
        nn.init.normal_(self.codebook.weight)

        self.register_buffer("ema_cluster_size", torch.zeros(self.K))
        self.register_buffer("ema_weight_sum",   torch.zeros(self.K, self.D))
        self.register_buffer("usage_count",      torch.zeros(self.K))

    def forward(self, x):
        B, C, H, W = x.shape                        # C == self.D
        x_permuted  = x.permute(0, 2, 3, 1)         # (B, H, W, C)
        x_flat      = x_permuted.flatten(0, 2)       # (B*H*W, C)

        # Squared L2 distance to each codebook entry
        x_sq  = (x_flat ** 2).sum(dim=1, keepdim=True)   # (N, 1)
        cb_sq = (self.codebook.weight ** 2).sum(dim=1)    # (K,)
        cross = x_flat @ self.codebook.weight.T            # (N, K)
        distances = x_sq - 2 * cross + cb_sq              # (N, K)

        indices = torch.argmin(distances, dim=1)           # (N,)
        lookup  = self.codebook(indices)                   # (N, D)

        if self.training:
            with torch.no_grad():
                one_hot       = F.one_hot(indices, num_classes=self.K).float()
                new_count     = one_hot.sum(dim=0)
                new_weight_sum = one_hot.T @ x_flat

                # EMA codebook update
                self.ema_cluster_size.mul_(self.ema_decay).add_(new_count     * (1 - self.ema_decay))
                self.ema_weight_sum  .mul_(self.ema_decay).add_(new_weight_sum * (1 - self.ema_decay))
                self.codebook.weight.data = self.ema_weight_sum / (self.ema_cluster_size.unsqueeze(1) + 1e-6)

                # Dead code reset
                self.usage_count.add_(new_count)
                self.steps += 1
                if self.steps % self.dead_code_interval == 0:
                    dead = (self.usage_count < 1.0).nonzero(as_tuple=True)[0]
                    if len(dead) > 0:
                        rand_idx = torch.randint(0, x_flat.shape[0], (len(dead),))
                        self.codebook.weight.data[dead] = x_flat[rand_idx].detach()
                        self.usage_count[dead] = 1.0

        x_back         = lookup.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        straight_through = x + (x_back - x).detach()
        loss             = self.commitment_beta * F.mse_loss(x, x_back.detach())

        return straight_through, loss, indices.reshape(B, H, W)

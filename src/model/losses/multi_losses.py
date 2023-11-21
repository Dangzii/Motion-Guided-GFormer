import torch
import torch.nn as nn


class Multi_losses(nn.Module):
    def __init__(self,alpha,beta,patch_size,grid_size,norm_pix_loss=False):
        super(Multi_losses,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.norm_pix_loss = norm_pix_loss
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # h = w = imgs.shape[2] // p

        p = self.patch_size
        h, w = imgs.shape[2] // p[0], imgs.shape[3] // p[1]

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p[0] * p[1] * 3))
        return x

    def restoration_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def path_loss(self, idx1, idx2, mask1, mask2):
        """
        idx:[J,],
        mask:[N,L],0 is keep, 1 is remove 
        """        
        p = self.patch_size
        idx1_unshuffle, idx2_unshuffle = torch.argsort(idx1), torch.argsort(idx2)
        mask1 = mask1.reshape(-1,self.grid_size[0],self.grid_size[1]).repeat_interleave(p[0],dim=1).repeat_interleave(p[1],dim=2)
        path1 = 1 - mask1[:,idx1_unshuffle,:]
        mask2 = mask2.reshape(-1,self.grid_size[0],self.grid_size[1]).repeat_interleave(p[0],dim=1).repeat_interleave(p[1],dim=2)
        path2 = 1 - mask2[:,idx2_unshuffle,:]
        loss = torch.sum((path1 - path2) ** 2) / (p[0] * p[1] * (mask1.shape[1] * mask1.shape[2])) 
        
        return loss
    
    
    def forward(self,imgs,pred1,pred2,idx1,idx2,mask1,mask2):
        if self.training:
            restoration_loss1 = self.restoration_loss(imgs, pred1, mask1)
            restoration_loss2 = self.restoration_loss(imgs, pred2, mask2)
            path_loss = self.path_loss(idx1, idx2, mask1, mask2)
            loss = self.alpha * (restoration_loss1 + restoration_loss2) + self.beta * path_loss
            return path_loss

class Multi_decoder_losses(nn.Module):
    def __init__(self,alpha,beta,patch_size,grid_size,norm_pix_loss=False):
        super(Multi_decoder_losses,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # h = w = imgs.shape[2] // p

        p = self.patch_size
        h, w = imgs.shape[2] // p[0], imgs.shape[3] // p[1]

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p[0] * p[1] * 3))
        return x

    def forward(self, imgs, pred1, pred2, idx1, idx2, mask1, mask2):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        idx:[J]
        """
        target1 = self.patchify(imgs[:,:,idx1,:])
        target2 = self.patchify(imgs[:,:,idx2,:])
        if self.norm_pix_loss:
            mean1,mean2 = target1.mean(dim=-1, keepdim=True),target2.mean(dim=-1, keepdim=True)
            var1,var2 = target1.var(dim=-1, keepdim=True),target2.var(dim=-1, keepdim=True)
            target1,target2 = (target1 - mean1) / (var1 + 1.e-6)**.5, (target2 - mean2) / (var2 + 1.e-6)**.5

        # loss_restoration1 = (pred1 - target1) ** 2 
        # loss_restoration1 = loss_restoration1.mean(dim=-1)  # [N, L], mean loss per patch
        # loss_restoration2 = (pred2 - target2) ** 2 
        # loss_restoration2 = loss_restoration2.mean(dim=-1)  # [N, L], mean loss per patch

        # loss_restoration1 = (loss_restoration1 * mask1).sum() / mask1.sum()  # mean loss on removed patches
        # loss_restoration2 = (loss_restoration2 * mask2).sum() / mask2.sum()  # mean loss on removed patches
        # loss_restoration = loss_restoration1 + loss_restoration2

        #idx1_unshuffle, idx2_unshuffle = torch.argsort(idx1), torch.argsort(idx2)
        #mask1 = mask1.reshape(-1,self.grid_size[0],self.grid_size[1]).repeat_interleave(self.patch_size[0],dim=1).repeat_interleave(self.patch_size[1],dim=2)
        mask1 = mask1.unsqueeze(-1).repeat(1,1,pred1.shape[-1])
        pred1 = torch.mul(pred1,(1-mask1))
        #pred1 = pred1[:,idx1_unshuffle,:]
        #mask2 = mask2.reshape(-1,self.grid_size[0],self.grid_size[1]).repeat_interleave(p[0],dim=1).repeat_interleave(p[1],dim=2)
        mask2 = mask2.unsqueeze(-1).repeat(1,1,pred1.shape[-1])
        pred2 = torch.mul(pred2,(1-mask2))
        #pred2 = pred2[:,idx2_unshuffle,:]
        loss_path = (pred1 - pred2) ** 2
        loss_path = loss_path.mean(dim=1).sum()
        # loss = loss_restoration + loss_path

        return loss_path


class Multi_decoder_losses(nn.Module):
    def __init__(self,alpha,beta,patch_size,grid_size,norm_pix_loss=False):
        super(Multi_decoder_losses,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # h = w = imgs.shape[2] // p

        p = self.patch_size
        h, w = imgs.shape[2] // p[0], imgs.shape[3] // p[1]

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p[0], w, p[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p[0] * p[1] * 3))
        return x

    def forward(self, imgs, pred1, pred2, idx1, idx2, mask1, mask2):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        idx:[J]
        """
        target1 = self.patchify(imgs[:,:,idx1,:])
        target2 = self.patchify(imgs[:,:,idx2,:])
        if self.norm_pix_loss:
            mean1,mean2 = target1.mean(dim=-1, keepdim=True),target2.mean(dim=-1, keepdim=True)
            var1,var2 = target1.var(dim=-1, keepdim=True),target2.var(dim=-1, keepdim=True)
            target1,target2 = (target1 - mean1) / (var1 + 1.e-6)**.5, (target2 - mean2) / (var2 + 1.e-6)**.5
        
        
        # loss_restoration1 = (pred1 - target1) ** 2 
        # loss_restoration1 = loss_restoration1.mean(dim=-1)  # [N, L], mean loss per patch
        # loss_restoration2 = (pred2 - target2) ** 2 
        # loss_restoration2 = loss_restoration2.mean(dim=-1)  # [N, L], mean loss per patch

        # loss_restoration1 = (loss_restoration1 * mask1).sum() / mask1.sum()  # mean loss on removed patches
        # loss_restoration2 = (loss_restoration2 * mask2).sum() / mask2.sum()  # mean loss on removed patches
        # loss_restoration = loss_restoration1 + loss_restoration2

        #idx1_unshuffle, idx2_unshuffle = torch.argsort(idx1), torch.argsort(idx2)
        #mask1 = mask1.reshape(-1,self.grid_size[0],self.grid_size[1]).repeat_interleave(self.patch_size[0],dim=1).repeat_interleave(self.patch_size[1],dim=2)
        mask1 = mask1.unsqueeze(-1).repeat(1,1,pred1.shape[-1])
        pred1 = torch.mul(pred1,(1-mask1))
        #pred1 = pred1[:,idx1_unshuffle,:]
        #mask2 = mask2.reshape(-1,self.grid_size[0],self.grid_size[1]).repeat_interleave(p[0],dim=1).repeat_interleave(p[1],dim=2)
        mask2 = mask2.unsqueeze(-1).repeat(1,1,pred1.shape[-1])
        pred2 = torch.mul(pred2,(1-mask2))
        #pred2 = pred2[:,idx2_unshuffle,:]
        loss_path = (pred1 - pred2) ** 2
        loss_path = loss_path.mean(dim=1).sum()
        # loss = loss_restoration + loss_path

        return loss_path

class Multi_losses1(nn.Module):
    def __init__(self,alpha,beta):
        super(Multi_losses1,self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self,target,pred,mask1,mask2):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        pred_loss = (pred - target) ** 2
        pred_loss = pred_loss.mean(dim=-1)  # [N, L], mean loss per patch
        pred_loss = (pred_loss * mask1).sum() / mask1.sum()  # mean loss on removed patches

        mask_loss = (mask1 - mask2) **2
        mask_loss = mask_loss.mean()
        # path_loss = target * (mask1 - mask2).unsqueeze(-1).repeat(1,1,pred.shape[-1])
        # path_loss = path_loss.mean(dim=-1) 

        loss = self.alpha * pred_loss + self.beta * mask_loss

        return loss

class Multi_losses2(nn.Module):

    def __init__(self,alpha,beta):
        super(Multi_losses2,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.LARGE_NUM = 1e9
    
    def _contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        """
        hidden1: (batch_size, dim)
        hidden2: (batch_size, dim)
        """
        batch_size, hidden_dim = hidden1.shape
        
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss


    def forward(self,target,pred,latent1,latent2,mask):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        assert mask.sum() / mask.shape[1] > 0.5
        pred_loss = (pred - target) ** 2
        pred_loss = pred_loss.mean(dim=-1)  # [N, L], mean loss per patch
        pred_loss = (pred_loss * mask).sum() / mask.sum()  # mean loss on removed patches

        contrastive_loss = self._contrastive_loss_forward(latent1,latent2)
        # path_loss = target * (mask1 - mask2).unsqueeze(-1).repeat(1,1,pred.shape[-1])
        # path_loss = path_loss.mean(dim=-1) 

        loss = self.alpha * pred_loss + self.beta * contrastive_loss

        return loss
    


class Multi_losses3(nn.Module):
    def __init__(self,alpha,in_feat=512,num_classes=3097,dropout_rate=0.5,weight=1.0,fc=False):
        super(Multi_losses3,self).__init__()
        if fc:
            self.linear = nn.Linear(in_feat,num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
        self.fc = fc
        
        self.alpha = alpha

    def forward(self,loss,predicts,targets=None):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.fc:
            predicts = self.linear(self.dropout(predicts))
        else:
            predicts = self.dropout(predicts)
        if self.training:
            return self.criterion(predicts,targets) + self.alpha * loss
        else:
            return self.criterion(predicts,targets) + self.alpha * loss


class MaskandCE(nn.Module):
    def __init__(self,weight=1.0,ratio=0.8):
        super(MaskandCE,self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight    
        self.ratio = ratio

    def forward(self,mask,predicts,targets=None):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.training:
            mask_ratio = mask.mean()
            return self.criterion(predicts,targets) + self.weight * ((self.ratio - mask_ratio) ** 2)
        else:
            return predicts, self.criterion(predicts,targets)
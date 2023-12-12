import torch
import torch.nn.functional as F
import torch.nn as nn


class PLL_loss(nn.Module):
    def __init__(self, train_givenY, mu=0.1):
        super().__init__()
        self.mu = mu
        print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = train_givenY.float()/train_givenY.sum(dim=1, keepdim=True)
        self.distribution = self.confidence.sum(0)/self.confidence.sum()

    def forward(self, logits, index, targets=None):
        log_p = F.log_softmax(logits, dim=1)
        if targets is None:
            # using confidence
            final_outputs = log_p * self.confidence[index, :].detach()
        else:
            # using given tagets
            final_outputs = log_p * targets.detach()
        loss_vec = -final_outputs.sum(dim=1)
        average_loss = loss_vec.mean()
        return average_loss, loss_vec

    @torch.no_grad()
    def get_distribution(self):
        self.update_distribution()
        return self.distribution

    @torch.no_grad()
    def update_distribution(self):
        self.distribution = self.confidence.sum(0) / self.confidence.sum()

    @torch.no_grad()
    def confidence_move_update(self, temp_un_conf, batch_index, ratio=None):
        if ratio:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - ratio) + temp_un_conf * ratio
        else:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - self.mu) + temp_un_conf * self.mu
        return None



import torch.nn.functional as F
import torch
from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss."""

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        loss_cls = F.cross_entropy(cls_score, label, **kwargs) # combines log_softmax and nll_loss
        return loss_cls

@LOSSES.register_module()
class AdversarialLoss(BaseWeightedLoss):
    """Cross Entropy Loss."""

    def _forward(self, cls_score, label, margin=0.05, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        # cls_score #[N,C]
        # label N
        softmax = F.softmax(cls_score)
        N, C = list(softmax.size())
        one_hot_label = F.one_hot(label, C)
        label_prob = torch.masked_select(softmax, one_hot_label.bool())
        max_non_label_prob = torch.max(softmax-one_hot_label, dim=1).values
        loss_margin = margin
        # print(softmax)
        # print(label)
        print(label_prob)
        print(max_non_label_prob)

        l_1 = torch.tensor([0.0]*N).to(softmax.device)
        l_2 = (label_prob-(max_non_label_prob-loss_margin))**2 / loss_margin
        l_3 = label_prob-(max_non_label_prob-loss_margin)
        adv_loss = torch.mean(torch.max(l_1, torch.min(l_2, l_3)))
        return adv_loss

        #loss_cls = F.cross_entropy(cls_score, label, **kwargs) # combines log_softmax and nll_loss
        #return loss_cls

@LOSSES.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits."""

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


if __name__ == "__main__":
    adv_loss = AdversarialLoss()
    cls_score = torch.randn([4,3])
    label = torch.tensor([0,1,2,0])
    print(adv_loss(cls_score, label))
import torch
import torch.nn as nn

from .base import BaseHead
from .bnn import BayesianPredictor, get_uncertainty
from ..builder import build_loss, HEADS


@HEADS.register_module()
class I3DBNNHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='BayesianNNLoss'),
                 spatial_type='avg',
                 dropout_ratio=0,
                 init_std=0.01,
                 compute_uncertainty=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.bnn_loss = self.loss_cls

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.compute_uncertainty = compute_uncertainty
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.bnn_cls = BayesianPredictor(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # normal_init(self.bnn_cls, std=self.init_std)
        pass  # BNN do not need to explicity initialized


    def forward(self, x, npass=2, testing=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        outputs, log_priors, log_variational_posteriors = self.bnn_cls(x, npass=npass, testing=testing)

        # gather output dictionary
        output = outputs.mean(0)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        output_dict = {'pred_mean': output,
                       'log_prior': log_prior,
                       'log_posterior': log_variational_posterior}
        if self.compute_uncertainty:
            uncertain_alea, uncertain_epis = get_uncertainty(outputs)
            output_dict.update({'aleatoric': uncertain_alea,
                                'epistemic': uncertain_epis})
        return output_dict

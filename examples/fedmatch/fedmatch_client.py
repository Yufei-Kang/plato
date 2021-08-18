"""
A federated semi-supervised learning client using FedMatch, and data samples on devices are mostly unlabeled.

Reference:
Jeong et al., "Federated Semi-supervised learning with inter-client consistency & disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf 
"""
import os
from dataclasses import dataclass
from plato.algorithms import fedavg
from plato.clients import simple


@dataclass
class Report(base.Report):
    """A client report containing the means and variances."""
    mean: float
    variance: float


class Client(simple.Client):
    """A fedmatch federated learning client who sends weight updates
    and the number of local epochs."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.helpers = None
        #self.pattern = None
        #self.max_local_iter = None

    async def train(self):
        """ Fedmatch clients use different number of local epochs. """

        report, weights = await super().train(
        )  # obtain update from local trainer,
        # loss in the trainer should be changed due to semi-supervised learning property

        # compute for mean, and variance that should be sent to server for further clustering
        #mean =
        #variance =

        # send them back to server
        return Report(report.num_samples, report.accuracy, mean,
                      variance), weights

    def load_payload(self, server_payload):
        """ Load model weights and helpers from server payload onto this client. """

        if isinstance(server_payload, list):
            fedavg.load_weights(server_payload[0])
            self.helpers = server_payload[1:]  # download helpers from server
        else:
            fedavg.load_weights(server_payload)
            self.helpers = None

"""
A federated learning training session using FedMatch.

Reference:
Jeong et al., "Federated Semi-supervised learning with inter-client consistency & disjoint learning", in the Proceedings of ICLR 2021.

https://arxiv.org/pdf/2006.12097.pdf 
"""
import os

os.environ['config_file'] = 'fedmatch_MNIST_lenet5.yml'

import fedmatch_client
import fedmatch_server
import fedmatch_trainer


def main():
    """ A Plato federated learning training session using the SCAFFOLD algorithm. """
    trainer = fedmatch_trainer.Trainer()
    client = fedmatch_client.Client(trainer=trainer)
    server = fedmatch_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
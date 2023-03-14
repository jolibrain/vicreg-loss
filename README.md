# A simple VICReg implementation

This project is an implementation on the [VICReg paper](https://arxiv.org/abs/2105.04906).
This paper introduces a new contrastive loss that is decomposed in three parts:

1. *The invariance loss*, to force the model to produce the same hidden representation
between two embeddings of the same object.
2. *The variance loss*, to force the model to diverse its representations between
the embeddings of different objects.
3. *The covariance loss*, to force the model to encode more information inside its
hidden representations by decorrelating the representation dimensions.

The goal of this repository is to provide a simple and hackable implementation
of this loss.
That way it is easy to copy and paste the code into your own project.

All credits go to the original paper and the official implementation:

* Paper: <https://arxiv.org/abs/2105.04906>
* Official implementation: <https://github.com/facebookresearch/vicreg>

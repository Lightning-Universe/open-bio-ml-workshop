# Open Bio ML Workshop

Slides and Code for the [OpenBioML](https://openbioml.org/) workshop 2023 with [Lightning AI](https://lightning.ai).

The first workshop, on Thursday, February 23, 2023 9:00 PM will cover:
- General introduction
- From raw PyTorch to Fabric
- Organize code with LightningModule
- Fabric with LightningModule
- Trainer with LightningModule + Callback
- Data loading
- Running on SLURM

The second workshop, on Thursday, March 2, 2023 9:00 PM, will cover:
- Intro to distributed (data, model, tensor parallelism + distributed sampling) training strategies
- PyTorch 2.0
- Performance optimization
- Benchmarking
- Apps and recipes


## Getting Started

Set up a new Python virtual environment. Example using conda:

```
conda create -n openbioml python=3.10
conda activate openbioml
```

Install Lightning:

```
pip install --pre lightning
```

This will also install PyTorch!


## Resources

[Lightning AI](https://lightning.ai)
[Lightning Fabric Documentation](https://pytorch-lightning.readthedocs.io/en/latest/fabric/fabric.html)
[PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/en/latest/)

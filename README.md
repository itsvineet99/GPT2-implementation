this repository contains code to pre-train "GPT-2" 124 million parameter model from scratch using pytorch. this project mainly follows a book "[Build a Large Language Model (From Scratch)](https://sebastianraschka.com/llms-from-scratch/)" by Sebastian Raschka. 

along with all the implementation given in a book here we add few extra optimizations and pre-train our model on bigger dataset called [simple-wikipedia](https://huggingface.co/datasets/rahular/simple-wikipedia).

we perform all the pre-training stages from scratch, from tokenization, embedding layers to attention mechanism and transformer blocks.

for optimization we use methods like gradient accumulation, mixed-precision training, `torch.compile()`, fused optimizer, selective weight decay, gradient clipping etc.

loss graph and some visuals from training:

![loss.png](https://i.ibb.co/cSzJD7kF/Screenshot-2026-03-02-at-11-14-10-PM.png)

epoch 1:
![loss_2.png](https://i.ibb.co/tT6vR39C/Screenshot-2026-03-02-at-11-12-45-PM.png)

epoch 2:
![epoch_1.png](https://i.ibb.co/9krtsK3K/image.png)

# Graph-based protein design from scratch 

Here we combine some of the best methods for graph based protein design currently available and see if we can develop some excellent evals for assessing how and why the models perform differently, with an eye towards improving them. 
The main sources of inspiration are: 

- [Generative Models for Graph-Based Protein Design](https://papers.nips.cc/paper/9711-generative-models-for-graph-based-protein-design) by John Ingraham, Vikas Garg, Regina Barzilay and Tommi Jaakkola, NeurIPS 2019.
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) by Justas Dauparas, 2021 
- [ESM-IF](https://github.com/facebookresearch/esm) by Alex Rives and coworkers, 2021  

From the orginal implementation and idea by John Ingraham: "Our approach 'designs' protein sequences for target 3D structures via a graph-conditioned, autoregressive language model". 


## Goals for this repository 

- [ ] Present a simple and understandable implementation of state of the art algorithms for graph-based protein design
- [ ] Implement featurization scheme from ProteinMPNN so that it can be directly compared 
- [ ] Perform analysis of the model attention mechanism 
- [ ] Devise evals that probe the ability of models under different conditions 


## Experiments 

To begin, we can train a base model using the default settings from the argument parser, using the `full` feature set 

```shell 
python train.py --features full
```


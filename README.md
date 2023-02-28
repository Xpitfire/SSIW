# Tutorial: Semantic Segmentation

**This guide compares a fine-tuned [SegFormer](https://arxiv.org/abs/2105.15203) with a fine-tuned variant of the 'The devil is in the labels' paper from [Yin et al.](https://arxiv.org/abs/2202.02002) on a semantic segmentation task. The dataset used is the [CMP Facade Database](https://cmp.felk.cvut.cz/~tylecr1/facade/).**

The main framework we will utilize is Huggingface [`🤗 transformers`](https://huggingface.co/transformers), an open-source library that offers easy-to-use implementations of state-of-the-art models. Baseline models are made publicly available on the HuggingFace hub, the largest open-source catalog of models and datasets, which includes the used Segformer baseline.

**Semantic segmentation** is the task of classifying each pixel in an image to a corresponding category. It has a wide range of use cases in fields such as medical imaging, autonomous driving, robotics, etc. For the facade dataset we are interested to classify the front-view of buildings based on 12 distinct classes. The classes are as follows: facade, molding, cornice, pillar, window, door, sill, blind, balcony, shop, deco, and background. 

In 2014, [Long et al.](https://arxiv.org/abs/1411.4038) published a fundamental paper that used convolutional neural networks for semantic segmentation. More recent successes in computer vision include the usage of Transformers, therefore also the usage in image classification tasks (i.e. [ViT](https://huggingface.co/blog/fine-tune-vit)). In this turn, Transformers have also been used for semantic segmentation, demonstrating excellent performance on several widely used datasets. We will specifically look at the SegFormer, which is a state-of-the-art model architecture for semantic segmentation introduced in 2021. It has a hierarchical Transformer encoder that does not rely on positional encodings and a simple multi-layer perceptron decoder.  In this case, we will use SegFormer to classify street view images of buildings.

More, recently, Yin et al. introduced a novel approach to semantic segmentation that achieves state-of-the-art performance in a zero-shot setting. This means that the model is able to achieve results equivalent to supervised methods on various semantic segmentation datasets, without being trained on these datasets. The approach involves replacing class labels with vector-valued embeddings of short paragraphs that describe the class. This allows for the merging of multiple datasets with different class labels and semantics, resulting in a merged dataset of over 2 million images. The resulting model achieves performance equal to state-of-the-art supervised methods on 7 benchmark datasets, and even outperforms methods when fine-tuned on standard semantic segmentation datasets. 


## Installing requirements

Create a conda environment:

```bash
conda env create -f environment.yml
```

Load environment:

```bash
conda activate cmp

```

## Fine-Tune a Model

Simply call the `train.sh` file:

```bash
./train.sh
```


## References


This repository contains the source code based on the repository of [Yin et al.](https://github.com/irfanICMLL/SSIW/) with their publication
[Wei Yin, Yifan Liu, Chunhua Shen, Anton van den Hengel, Baichuan Sun, The devil is in the labels: Semantic segmentation from sentences](https://arxiv.org/abs/2202.02002) and the [HuggingFace SegFormer fine-tuning tutorial](https://huggingface.co/blog/fine-tune-segformer).

To obtain the weights go to:

Embedding: https://cloudstor.aarnet.edu.au/plus/s/gXaGsZyvoUwu97t

CKPT: https://cloudstor.aarnet.edu.au/plus/s/AtYYaVSVVAlEwve


To see the full tutorial open the `Fine_tune_SS_Model.ipynb`. 

Have fun! :) 

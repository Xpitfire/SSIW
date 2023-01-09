# Assignment

**This guide compares a fine-tuned [SegFormer](https://arxiv.org/abs/2105.15203) with the 'The devil is in the labels' from [Yin et al.](https://arxiv.org/abs/2202.02002). The dataset used is the [CMP Facade Database](https://cmp.felk.cvut.cz/~tylecr1/facade/). The used framework is Huggingface [`🤗 transformers`](https://huggingface.co/transformers), an open-source library that offers easy-to-use implementations of state-of-the-art models. The models are made available on the HuggingFace hub, the largest open-source catalog of models and datasets.**

Semantic segmentation is the task of classifying each pixel in an image to a corresponding category. It has a wide range of use cases in fields such as medical imaging, autonomous driving, robotics, etc. For the facade dataset we are interested to identify and classify the following classes: facade, molding, cornice, pillar, window, door, sill, blind, balcony, shop, deco, and background.

Semantic segmentation is a type of image classification that involves dividing an image into different regions and assigning each region a class label. In 2014, [Long et al.](https://arxiv.org/abs/1411.4038) published a fundamental paper that used convolutional neural networks for semantic segmentation. More recently, the use of Transformers for image classification has become popular (such as [ViT](https://huggingface.co/blog/fine-tune-vit)). Transformers are also being used for semantic segmentation, leading to improved performance in this task. SegFormer is a state-of-the-art model for semantic segmentation introduced in 2021. It has a hierarchical Transformer encoder that does not rely on positional encodings and a simple multi-layer perceptron decoder. SegFormer has demonstrated excellent performance on several widely used datasets. In this case, we will use SegFormer to classify sidewalk images for use in a pizza delivery robot.

Recently, Yin et al. introduced a novel approach to semantic segmentation that achieves state-of-the-art performance in a zero-shot setting. This means that the model is able to achieve results equivalent to supervised methods on various semantic segmentation datasets, without being trained on these datasets. The approach involves replacing class labels with vector-valued embeddings of short paragraphs that describe the class. This allows for the merging of multiple datasets with different class labels and semantics, resulting in a merged dataset of over 2 million images. The resulting model achieves performance equal to state-of-the-art supervised methods on 7 benchmark datasets, and even outperforms methods when fine-tuned on standard semantic segmentation datasets. Therefore, we will use as a backbone the pre-trained encoder from Yin et al. and fine-tune a classification head for the CMP facade dataset with very few labels.


## Installing requirements

Create a conda environment:

```bash
conda env create -f environment.yml
```

Load environment:

```bash
conda activate cmp
```



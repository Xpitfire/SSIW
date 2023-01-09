import torch
import hydra
import torch.distributed as dist
import numpy as np
from omegaconf import DictConfig
from torch import nn
from utils.segformer import get_configured_segformer
from dataset.load_cmp import CMPDataset
from datasets import load_dataset
from torchvision.transforms import ColorJitter
from transformers import SegformerFeatureExtractor
from typing import Optional, Union, Tuple
from torch.nn import CrossEntropyLoss
from transformers import SegformerPreTrainedModel, SegformerConfig, SegformerDecodeHead
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import TrainingArguments
from datasets import load_metric
from transformers import Trainer
from transformers import SegformerForSemanticSegmentation
from box import Box


class CustomTrainer(Trainer):
    """
    A custom class for training a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    inputs : Dict[str, Any]
        A dictionary of input tensors.
    return_outputs : bool, optional
        If True, return a tuple of (loss, outputs). Defaults to False.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


class CustomSegformerForSemanticSegmentation(SegformerPreTrainedModel):
    """
    A custom class for the Segformer model that can be used for semantic segmentation tasks.

    Parameters
    ----------
    config : SegformerConfig
        Configuration object for the Segformer model.
    seg_model : nn.Module
        The base Segformer model.
    """
    def __init__(self, config: SegformerConfig, seg_model: nn.Module):
        super().__init__(config)
        self.segformer = seg_model
        self.decode_head = SegformerDecodeHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
        )
        logits = self.decode_head(outputs)

        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def load_datasets(config: DictConfig) -> None:
    """
    Load the training and evaluation datasets using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    # used for getting the Dataset properties
    id2label = CMPDataset.id2label
    label2id = {v: k for k, v in id2label.items()}
    num_labels = CMPDataset.num_labels

    ds = load_dataset('Xpitfire/cmp_facade')

    train_ds = ds['train']
    eval_ds = ds['eval']
    test_ds = ds['test']
    
    # assign values
    config.train_ds = train_ds
    config.eval_ds = eval_ds
    config.test_ds = test_ds
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = num_labels


def prepare_dataset_transforms(config: DictConfig) -> None:
    """
    Prepare the dataset transforms for training and validation using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    feature_extractor = SegformerFeatureExtractor()
    # create color jittering data augmentation
    jitter = ColorJitter(brightness=config.colorjitter.brightness, 
                         contrast=config.colorjitter.contrast, 
                         saturation=config.colorjitter.saturation, 
                         hue=config.colorjitter.hue) 

    def train_transforms(example_batch):
        images = [jitter(x) for x in example_batch['pixel_values']]
        labels = [x for x in example_batch['label']]
        inputs = feature_extractor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch['pixel_values']]
        labels = [x for x in example_batch['label']]
        inputs = feature_extractor(images, labels)
        return inputs

    # Set transforms
    config.train_ds.set_transform(train_transforms)
    config.eval_ds.set_transform(val_transforms)
    
    
def init_baseline_model(config: DictConfig) -> None:
    """
    Initialize a baseline model using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    pretrained_model_name = config.baseline_model
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id
    )
    model = model.cuda()    
    if config.baseline_ckpt_path and len(config.baseline_ckpt_path) > 0:
        # set the path to the checkpoint file
        ckpt_path = config.baseline_ckpt_path
        # load the checkpoint file and save it to the 'checkpoint' variable
        checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
        model.load_state_dict(checkpoint, strict=True)
    config.model = model
    
    
def init_segmentation_backbone(config: DictConfig) -> None:
    """
    Initialize a segmentation backbone model using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    seg_model = get_configured_segformer(config.num_labels, criterion=None, load_imagenet_model=False)

    # set the model to eval mode
    seg_model.eval()
    # wrap the model in a DataParallel wrapper
    seg_model = torch.nn.DataParallel(seg_model)

    # set the path to the checkpoint file
    ckpt_path = config.ckpt_path
    # load the checkpoint file and save it to the 'checkpoint' variable
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    # create a dictionary with only the items from the checkpoint file whose keys do not contain 'criterion.0.criterion.weight'
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    # create a dictionary with only the items from the previous dictionary whose keys do not contain 'module.segmodel.head'
    ckpt_filter = {k: v for k, v in ckpt_filter.items() if 'module.segmodel.head' not in k}
    # create a dictionary with only the items from the previous dictionary whose keys do not contain 'module.segmodel.auxi_net'
    ckpt_filter = {k: v for k, v in ckpt_filter.items() if 'module.segmodel.auxi_net' not in k}
    # load the filtered checkpoint into the model
    seg_model.load_state_dict(ckpt_filter, strict=False)

    
    
def init_segmentation_model(config: DictConfig) -> None:
    """
    Initialize a segmentation model using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    # adapt hyperparameters for the SSIW B5 model
    configuration = SegformerConfig(subnorm_type='batch',
                                semantic_loss_ignore_index=0, # ignore first index of dataset id
                                hidden_sizes=[64, 128, 320, 512],
                                decoder_hidden_size=768,
                                num_labels=config.num_labels)
    seg_model = config.seg_model.module.segmodel.encoder
    model = CustomSegformerForSemanticSegmentation(configuration, 
                                                   seg_model)
    model = model.cuda()
    config.model = model
    

def init_metric(config: DictConfig) -> None:
    """
    Initialize a metric function using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    metric = load_metric(config.metric)
    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(predictions=pred_labels, references=labels, 
                                    num_labels=config.num_labels, 
                                    ignore_index=0, # ignore first index of dataset id
                                    reduce_labels=False)
            for key, value in metrics.items():
                if type(value) is np.ndarray:
                    metrics[key] = value.tolist()[1:]
            return metrics
    config.metric = compute_metrics
    
    
def init_training(config: DictConfig) -> None:
    """
    Initialize a training process using the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    training_args = TrainingArguments(
        config.ouptup_dir,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_strategy="end",
        optim=config.optimizer,
        remove_unused_columns=True
    )
    
    trainer = CustomTrainer(
        model=config.model,
        args=training_args,
        train_dataset=config.train_ds,
        eval_dataset=config.eval_ds,
        compute_metrics=config.metric,
    )
    
    config.trainer = trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Initialize the segmentation model backbone.
    
    Parameters
    ----------
    config : DictConfig
        Dictionary-like object containing configuration parameters.

    Returns
    -------
    None
    """
    config = Box(config)
    
    # prepare the dataset
    load_datasets(config)
    prepare_dataset_transforms(config)
    
    # prepare the model
    if config.model_id == 'SegFormer_Baseline':
        init_baseline_model(config)
    elif config.model_id == 'SegFormer_SSIW':
        init_segmentation_backbone(config)
        init_segmentation_model(config)
    else:
        raise NotImplementedError("Specified model is currently not supported!")
    
    # prepare training
    init_metric(config)
    init_training(config)
    
    # execute training run
    config.trainer.train()


if __name__ == "__main__":
    # init for dataparallel
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:23456", rank=0, world_size=1)
    main()

import segmentation_models_pytorch as smp
import torch.nn as nn

def get_resnet_unet_model(
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
):
    """
    Creates a U-Net model with a ResNet encoder using segmentation_models_pytorch.

    Parameters:
    - encoder (str): Name of ResNet encoder, e.g., 'resnet18', 'resnet34', 'resnet50', etc.
    - encoder_weights (str or None): Pretrained weights. Options: 'imagenet', None
    - in_channels (int): Number of input image channels (3 for RGB)
    - classes (int): Number of output classes (1 for binary segmentation)
    - activation (str or None): Activation function at the final layer (e.g., 'sigmoid', 'softmax', or None)

    Returns:
    - model (nn.Module): A PyTorch model instance
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    return model

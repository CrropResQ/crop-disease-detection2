import torch
import torch.nn as nn
import timm
def get_cropresq_model(model_name, num_classes):
 """
 Returns the specified pretrained model customized for the dataset.
 Supported: vit, dinov2, coatnet, swin, maxvit, deit, siglip, resnet50
 """
 model_name = model_name.lower()
 if model_name == "vit":
 model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
 elif model_name == "dinov2":
 model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True,
num_classes=num_classes)
 elif model_name == "coatnet":
 model = timm.create_model('coatnet_0_rw_224', pretrained=True, num_classes=num_classes)
 elif model_name == "swin":
 model = timm.create_model('swin_base_patch4_window7_224', pretrained=True,
num_classes=num_classes)
 elif model_name == "maxvit":
 model = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=num_classes)
 elif model_name == "deit":
 model = timm.create_model('deit_base_patch16_224', pretrained=True,
num_classes=num_classes)
 elif model_name == "siglip":
 # Using SigLIP vision tower pretrained weights
 model = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=True,
num_classes=num_classes)
 elif model_name == "resnet50":
 # Included as requested from the ResNet50 baseline directory
 model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
 else:
 raise ValueError(f"Model {model_name} not recognized.")
 return model
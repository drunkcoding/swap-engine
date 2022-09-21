
from transformers import AutoFeatureExtractor
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path, split="train"):
        transform = []

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path
        )

        if feature_extractor.do_resize:
            transform.append(
                RandomResizedCrop(feature_extractor.size) if split == "train" else Resize(feature_extractor.size)
            )

        transform.append(RandomHorizontalFlip() if split == "train" else CenterCrop(feature_extractor.size))
        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x.convert("RGB"))

def vit_collate_fn(batch):
    # print("==batchsize====", len(batch))
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = torch.tensor(transposed_data[1])
    return {"pixel_values": inp, 'labels': tgt}
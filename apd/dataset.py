from glob import glob
import os
from PIL import Image

from torchvision.transforms import Compose
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class CocoDataset(Dataset):
    "COCO Caption-Image Pair Dataset"

    def __init__(
        self,
        caption_file: str,
        image_dir: str,
        preprocess: Compose,
    ):
        super().__init__()
        self.preprocess = preprocess

        coco_dict = COCO(caption_file).anns
        self.image_dir = image_dir
        self.captions = [cap_dict["caption"] for cap_dict in coco_dict.values()]
        self.image_ids = [cap_dict["image_id"] for cap_dict in coco_dict.values()]

    def __getitem__(self, index, device="cuda"):
        img_id = self.image_ids[index]
        img_path = glob(os.path.join(self.image_dir, str(img_id).zfill(12) + ".jpg"))[-1]
        image = self.preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

        return self.captions[index], image

    def __len__(self):
        return len(self.captions)

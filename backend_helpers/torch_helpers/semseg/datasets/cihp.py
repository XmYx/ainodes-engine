import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


def replace_with_dict(ar, k, v, sidx):
    ks = k[sidx]
    vs = v[sidx]
    return vs[torch.searchsorted(ks, ar)]


class CIHP(Dataset):
    """This has Best Human Parsing Labels
    num_classes: 19+background
    28280 train images
    5000 val images
    """
    OLD_CLASSES = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 'coat', 'socks', 'pants',
                   'jumpsuits', 'scarf', 'skirt', 'face', 'left-arm', 'right-arm', 'left-leg', 'right-leg', 'left-shoe', 'right-shoe']
    OLD2NEW = {
        'background': ['background'],
        'head': ['hat', 'hair', 'sunglasses', 'face'],
        'upperclothes': ['upperclothes', 'coat'],
        'dress': ['dress', 'jumpsuits'],
        'lowerclothes': ['pants', 'skirt'],
        'scarf': ['scarf'],
        'arms': ['left-arm', 'right-arm', 'glove'],
        'legs': ['left-leg', 'right-leg', 'socks'],
        'shoes': ['left-shoe', 'right-shoe']
    }
    OLD2NEW_IDX = {}
    for i, v in enumerate(OLD2NEW.values()):
        for ok in v:
            OLD2NEW_IDX[OLD_CLASSES.index(ok)] = i
    CLASSES = list(OLD2NEW.keys())
    PALETTE = torch.tensor([[120, 120, 120], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [
                           84, 50, 0], [51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84], [254, 254, 0], [254, 169, 0]])
    PALETTE = PALETTE[:len(CLASSES)]

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'Training' if split == 'train' else 'Validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        img_path = Path(root) / 'instance-level_human_parsing' / split / 'Images'
        self.files = list(img_path.glob('*'))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

        self.k_dic = torch.tensor(list(self.OLD2NEW_IDX.keys()))
        self.v_dic = torch.tensor(list(self.OLD2NEW_IDX.values()))
        self.sidx = self.k_dic.argsort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('Images', 'Category_ids').replace('.jpg', '.png')

        image = io.read_image(img_path, mode=io.ImageReadMode.RGB)
        label = io.read_image(lbl_path)

        label = replace_with_dict(label, self.k_dic, self.v_dic, self.sidx)

        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


class CCIHP(CIHP):
    CLASSES = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'facemask', 'coat', 'socks', 'pants', 'torso-skin',
               'scarf', 'skirt', 'face', 'left-arm', 'right-arm', 'left-leg', 'right-leg', 'left-shoe', 'right-shoe', 'bag', 'others']
    PALETTE = torch.tensor([[120, 120, 120], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0], [
                           51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84], [254, 254, 0], [254, 169, 0], [102, 254, 0], [182, 255, 0]])


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(CIHP, '/home/shivam/')

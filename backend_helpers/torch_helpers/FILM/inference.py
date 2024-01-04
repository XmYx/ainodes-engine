import bisect
import os.path

from tqdm import tqdm
import torch
import numpy as np

from .film_util import load_image
from .. import poorman_wget


class FilmModel():

    def __init__(self):
        super().__init__()



        self.model_path = "models/other/film_net_fp16.pt"
        url = 'https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/film_net_fp16.pt'
        if not os.path.isfile(self.model_path):
            poorman_wget(url, self.model_path)

        self.model = torch.jit.load(self.model_path, map_location='cpu')
        self.model.eval()
        self.model = self.model.half()
        self.apply_torch_options()
    def apply_torch_options(self):
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    def inference(self, img1, img2, inter_frames):
        self.model.cuda()
        img_batch_1, crop_region_1 = load_image(img1)
        img_batch_2, crop_region_2 = load_image(img2)

        img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
        img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

        results = [
            img_batch_1,
            img_batch_2
        ]

        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in tqdm(range(len(remains)), 'Generating in-between frames'):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            x0 = x0.half()
            x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = self.model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]

        y1, x1, y2, x2 = crop_region_1
        frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

        self.model.cpu()
        return frames
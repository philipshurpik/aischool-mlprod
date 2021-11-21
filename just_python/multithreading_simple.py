import time
from PIL import Image
import numpy as np
import torch
torch.set_num_threads(1)

import torchvision

model = torchvision.models.mobilenet_v3_large(pretrained=True, progress=True)
model = model.eval().requires_grad_(False).to(torch.float32).to(device)


def resize(image, shape):
    return np.array(Image.fromarray(image).resize(shape))


def processing(im_frames):
    print('processing', im_frames.shape)
    resized = [resize(frame, (256, 256)) for frame in im_frames]
    results = [model(torch.tensor(frame).permute(2, 0, 1).div(255).unsqueeze(0).to(device)) for frame in resized]
    [resize(frame, (3840, 2160)) for frame in resized]
    max_ = [x.cpu().numpy().argmax() for x in results]
    print('mean result value',np.array(max_).mean())
    return max_


def run_single_thread(frames):
    start = time.time()
    processing(frames)
    end = time.time()
    print('Elapsed time single thread:', end - start)


if __name__ == '__main__':
    frames = np.random.randint(255, size=(32, 3840, 2160, 3), dtype=np.uint8)
    run_single_thread(frames)

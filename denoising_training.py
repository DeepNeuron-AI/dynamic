import cv2
from torch.utils.data import Dataset
import torch 
import numpy as np
from torchvision import transforms
import os 
import echonet 
from pathlib import Path


def to_tensor(array: np.ndarray):
    return torch.from_numpy(array)
        
def to_float(tensor: torch.Tensor):
    return tensor.float()
    

class DenoisedDataset(Dataset):

    def __init__(self, directory: str, noise_factor: float = 0.1):
        self.directory = Path(directory)
        self.video_directory = self.directory / "Videos"
        self.noise_factor = noise_factor
        
        # Compute mean and std
        mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=str(self.directory), split="train"))
        self.mean = mean 
        self.std = std 

        # Get filepaths for each video
        self.filepaths = list(self.video_directory.iterdir())


        self.transform = transforms.Compose([
            Transpose((1, 2, 3, 0)), # (frame, height, width, color)
            Normalize(mean, std),
            Transpose((0, 3, 1, 2)), # (frames, color, height, width)
            to_tensor
            # transpose_video,
            # to_tensor,
            # to_float, 
            # normalize,
            # transforms.Normalize(mean, std),
        ])
        self.noise = AddGaussianNoise(0., noise_factor)

        print(self.mean)
        print(self.std)

    def __getitem__(self, index):
        video = echonet.utils.loadvideo(str(self.filepaths[index]))
        video = self.transform(video)
        noisy_video = self.noise(video)
        return noisy_video, video

    def __len__(self):
        return len(self.filepaths) 

class AddGaussianNoise(object):
    '''
    from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745 
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Transpose:
    def __init__(self, dims) -> None:
        self.dims = dims

    def __call__(self, array: np.ndarray):
        return array.transpose(self.dims)      


class Normalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, array: np.ndarray):
        return (array - self.mean) / self.std



if __name__ == "__main__":
    directory = r"C:\Users\Allis\Documents\MDN\Ultrasound2023\dynamic\a4c-video-dir"
    dataset = DenoisedDataset(directory, noise_factor=1.)
    noisy_video, video = dataset[100]
    noisy_video = noisy_video.numpy()
    video = video.numpy()

    mean, std = dataset.mean, dataset.std
    video = video.transpose((0, 2, 3, 1)) # (frame, height, width, color)
    print(((video * std + mean) < 0).sum())
    video = (video * std + mean).astype(np.uint8)
    # video = video.transpose((0, 3, 1, 2)) 

    noisy_video = noisy_video.transpose((0, 2, 3, 1)) # (frame, height, width, color)
    print(((noisy_video * std + mean) < 0).sum())
    noisy_video = noisy_video * std + mean
    noisy_video = np.clip(noisy_video, 0, 255)
    noisy_video = noisy_video.astype(np.uint8)
    
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Noisy", cv2.WINDOW_NORMAL)

    i = 0
    is_playing = True
    while True:
        if i >= len(video):
            i = 0

        cv2.imshow("Original", video[i])
        cv2.imshow("Noisy", noisy_video[i])

        keypress = cv2.waitKey(50) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord(' '):
            is_playing = not is_playing
        else:
            if is_playing:
                i += 1

    """
    mean = 30, std= 50
    Starts: (0, 0, 0)
    Normalise: (-0.6, -0.6, -0.6)
    Normalise: (-0.05, 0, 0) -> (255, 0, 0)
    """
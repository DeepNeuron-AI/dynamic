#%%
import numpy as np
import torch
import torchvision
import tqdm
from dotenv import dotenv_values

import echonet
from weak_labels.utils import contains_RV, cutoff_from_LV_box, crop_ultrasound_borders, get_largest_contour, get_LV_RV_area_correlation, replace_tiny_RV_frames, remove_septum, RVDisappeared

from pathlib import Path

config = dotenv_values(".env")
ECHONET_VIDEO_DIR = Path(config["ECHONET_VIDEO_DIR"])
NUM_WORKERS = 4
CHECKPOINT_FP = Path("output/segmentation/all-patients/best.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10


def run_inference(dataloader, model):
    with torch.no_grad():
        for (x, (filenames), length) in tqdm.tqdm(dataloader):
            segmentation_masks = []
            
            # Required since their dataloader collate function doesn't handle 
            # filename target very well, so we're just unmangling the filenames here!
            filenames = ["".join(fn) for fn in zip(*filenames)]

            # Run segmentation model on blocks of frames one-by-one
            # The whole concatenated video may be too long to run together
            y = np.concatenate([model(x[i:(i + BATCH_SIZE), :, :, :].to(DEVICE))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], BATCH_SIZE)])

            start = 0
            x = x.numpy()
            for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                video = x[start:(start + offset), ...]
                logit = y[start:(start + offset), 0, :, :]

                # Get frames, channels, height, and width
                f, c, h, w = video.shape  # pylint: disable=W0612 (e.g. video.shape = (248, 3, 112, 112))
                assert c == 3

                segmentation_mask = logit > 0
                segmentation_masks.append(segmentation_mask)

                # Move to next video
                start += offset

            yield segmentation_masks, filenames

        raise StopIteration


## Set up datasets and dataloaders
# Get mean and std of data
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=None, external_test_location=ECHONET_VIDEO_DIR, split="EXTERNAL_TEST"))

# Saving videos with segmentations
dataset = echonet.datasets.Echo(root=None, split="EXTERNAL_TEST",
                                external_test_location=ECHONET_VIDEO_DIR,
                                target_type=["Filename"],  # Need filename for saving, and human-selected frames to annotate
                                mean=mean, std=std,  # Normalization
                                length=None, max_length=None, period=1  # Take all frames
                                )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=False, collate_fn=echonet.utils.segmentation._video_collate_fn)

# Make flipped dataloader now
flipped_dataset = echonet.datasets.Echo(root=None, split="EXTERNAL_TEST",
                                external_test_location=ECHONET_VIDEO_DIR,
                                target_type=["Filename"],  # Need filename for saving, and human-selected frames to annotate
                                mean=mean, std=std,  # Normalization
                                flip_video=True,
                                length=None, max_length=None, period=1  # Take all frames
                                )
flipped_dataloader = torch.utils.data.DataLoader(flipped_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=False, collate_fn=echonet.utils.segmentation._video_collate_fn)

# Set up model
model = torchvision.models.segmentation.__dict__["deeplabv3_resnet50"](weights=None, aux_loss=False)

model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
if DEVICE.type == "cuda":
    model = torch.nn.DataParallel(model)
model.to(DEVICE)

checkpoint = torch.load(CHECKPOINT_FP)
model.load_state_dict(checkpoint['state_dict'])

# Save videos with segmentation
# Only run if missing videos

model.eval()



success_counter = 0
for (LV_masks_list, filenames), (RV_masks_list, filenames) in zip(run_inference(dataloader, model), run_inference(flipped_dataloader, model)):
    for LV_masks, RV_masks, filename in zip(LV_masks_list, RV_masks_list, filenames):
        RV_masks = np.flip(RV_masks, axis=-1)

        if not contains_RV(LV_masks, RV_masks):
            print("Video does not appear to contain RV, skipping")
            continue
        if any(np.sum(RV_masks, axis=(1, 2)) == 0):
            print("Initial RV segmentation completely disappeared in at least one frame, skipping")
            continue


        # Refine the RV masks
        RV_masks = cutoff_from_LV_box(LV_masks, RV_masks)
        if any(np.sum(RV_masks, axis=(1, 2)) == 0):
            print("RV segmentation (after LV cutoffs) completely disappeared in at least one frame, skipping")
            continue

        RV_masks = crop_ultrasound_borders(RV_masks)
        if any(np.sum(RV_masks, axis=(1, 2)) == 0):
            print("RV segmentation (after border crop) completely disappeared in at least one frame, skipping")
            continue
        try:
            RV_masks = get_largest_contour(RV_masks)
            if any(np.sum(RV_masks, axis=(1, 2)) == 0):
                print("RV segmentation (largest contour) completely disappeared in at least one frame, skipping")
                continue
        except RVDisappeared as e:
            print("RV disappeared afer cropping LV box and ultrasound borders, skipping")
            continue
            
        RV_masks = remove_septum(LV_masks, RV_masks)
        if any(np.sum(RV_masks, axis=(1, 2)) == 0):
            print("RV segmentation (after septum removal) completely disappeared in at least one frame, skipping")
            continue

        RV_masks = replace_tiny_RV_frames(RV_masks)

        # If we made it this far, apply some metrics to guess if this is a "good" 
        # quality RV segmentation or not. We only want the best of the best here
        if get_LV_RV_area_correlation(LV_masks, RV_masks) < 0.5:
            print("LV and RV area did not correlate well")
        else:
            success_counter += 1
            print(f"Actually worked! ({success_counter})")
            output_fp = Path(f"output/segmentation/best-RV/{filename}").with_suffix(".npy")
            np.save(output_fp, RV_masks)

# %%
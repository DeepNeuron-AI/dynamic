import echonet

from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()
    echonet.utils.segmentation.run(data_dir="C:/Users/mini_/Documents/dynamic/a4c-video-dir", output="C:/Users/mini_/Documents/dynamic/output/seg-test", weights="C:/Users/mini_/Documents/dynamic2/output/segmentation/deeplabv3_resnet50_random/best.pt", run_test=True, save_video=False, num_epochs=0,num_train_patients=0)

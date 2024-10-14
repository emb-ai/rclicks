from .grabcut import GrabCutDataset


class BerkeleyDataset(GrabCutDataset):
    def __init__(self, dataset_path, args, **kwargs):
        super().__init__(dataset_path, args, images_dir_name='images', masks_dir_name='masks', **kwargs)

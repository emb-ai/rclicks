import cv2
import numpy as np

from isegm.data.sample import DSample
from isegm.data.datasets import GrabCutDataset


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()


def limit_longest_size(image, max_side_size, target_size=None, interpolation=cv2.INTER_LINEAR):
    if target_size is None:
        if image.shape[0] > image.shape[1]:
            target_h = min(image.shape[0], max_side_size)
            target_w = int(image.shape[1] / image.shape[0] * target_h + 0.5)
        else:
            target_w = min(image.shape[1], max_side_size)
            target_h = int(image.shape[0] / image.shape[1] * target_w + 0.5)
    else:
        target_h, target_w = target_size
    dtype = image.dtype
    image = cv2.resize(image.astype(np.uint8), (target_w, target_h), interpolation=interpolation)
    image = image.astype(dtype)
    return image, (target_h, target_w)


class TETRISDataset(GrabCutDataset):
    def __init__(self, dataset_path, args, max_side_size=2048, images_dir_name: str = 'images', masks_dir_name: str = 'masks', **kwargs):
        super(TETRISDataset, self).__init__(dataset_path, args, images_dir_name=images_dir_name, masks_dir_name=masks_dir_name, **kwargs)
        self.max_side_size = max_side_size

    def aligned_resize(self, mask, new_height, new_width):
        mask_resized = np.zeros((new_height, new_width), dtype=mask.dtype)
        uniques = np.nonzero(np.bincount(mask.ravel()))[0]
        for val in uniques:
            if val == 0:
                continue
            resized_mask = cv2.resize((mask == val).astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask_resized = np.where(resized_mask, val, mask_resized)
        return mask_resized.astype(mask.dtype)


    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = self._images_path / image_name
        mask_path = (self._insts_path / image_name).with_suffix('.png')

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(str(mask_path))
        instances_mask = instances_mask.astype(np.int32)
        instances_mask = instances_mask[:, :, 0] * 65536 + instances_mask[:, :, 1] * 256 + instances_mask[:, :, 2]

        if self.max_side_size > 0:
            image, new_shape = limit_longest_size(image, self.max_side_size)
            instances_mask = self.aligned_resize(instances_mask, new_shape[0], new_shape[1])

        object_ids, _ = get_labels_with_sizes(instances_mask)

        return DSample(image, instances_mask, objects_ids=object_ids, sample_id=index, imname=image_name)

from pathlib import Path
import math
import typing as t
import rclicks
import rclicks.clickmap
import rclicks.paths
from rclicks.utils import padding
from tqdm import tqdm
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
from rclicks.nets import SegNeXtSaliency
from torch.utils.data import Dataset, DataLoader
import albumentations as alb
from albumentations import *
from torch.utils.tensorboard import SummaryWriter
import cv2
import click



def kld(y_pred, y_true, eps=1e-9):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.
    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.
    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_true = torch.sum(y_true, dim=(1, 2, 3), keepdim=True)
    y_true = y_true / (eps + sum_true)
    
    sum_pred = torch.sum(y_pred, dim=(1, 2, 3), keepdim=True)
    y_pred = y_pred / (eps + sum_pred)
    
    loss = y_true * torch.log(eps + y_true / (eps + y_pred))
    loss = torch.mean(torch.sum(loss, dim=(1, 2, 3)))
    return loss

def similarity(s_map, gt):
    s_map = s_map.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return np.sum(np.minimum(s_map, gt))

def cc_f(s_map, gt):
    s_map = s_map.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r

def normalize_map(s_map):
    norm_s_map = (s_map - (s_map).min())/(((s_map).max() - (s_map).min())*1.0)
    return norm_s_map

def calculate_metrics(pred_clickmap, gt_clickmap, ):
    """Calculates average CC, SIM, NSS on the passed data
    """
    
    sim_score = []
    nss_score = []
    cc_score = []
    
    for pr, gt in zip(pred_clickmap, gt_clickmap):
        gt_120_sm = normalize_map(gt[0])
        pred_sm = normalize_map(pr[0])
        sim_score.append(similarity(pred_sm, gt_120_sm))
        cc_vlues = cc_f(pred_sm, gt_120_sm)
        cc_score.append(cc_vlues)

    return  np.mean(cc_score), np.mean(sim_score)


class ClickMapsDataset(Dataset):

    def __init__(self,
                 clickmaps: t.Union[rclicks.clickmap.CachedClickMaps, rclicks.clickmap.ClickMaps],
                 image_normalization: alb.Compose=None,
                 transforms: alb.Compose=None,
        ):

        self.image_normalization = image_normalization
        self.transforms = transforms
        self.clickmaps = clickmaps

    def __len__(self):
        return len(self.clickmaps)
    
    def __getitem__(self, idx):
        sample = self.clickmaps.sample(idx)
        image = sample.image
        mask = sample.mask
        error_mask = sample.error_mask
        click_map = sample.click_map

        image = self.image_normalization(image=image)['image']
        if self.transforms is not None:
            augmented = self.transforms(
                image=image,
                mask=mask,
                click_map=click_map,
                error_mask=error_mask)
            mask = augmented['mask']
            image = augmented['image']
            error_mask = augmented['error_mask']
            click_map = augmented['click_map']
        mask = padding(mask)
        error_mask = padding(error_mask)
        image = padding(image)
        click_map = padding(click_map)
        return (np.transpose(
            image, (2, 0, 1)),
            mask[None, :, :],
            error_mask[None, :, :],
            click_map[None, :, :]
        )


normalization = alb.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)
denormalization = alb.transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def create_train_dataset(cls=rclicks.clickmap.CachedClickMaps,
                  **kwargs):
    img_normalization = alb.Compose(
        [
            alb.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        ])
    transforms = alb.Compose(
        [HorizontalFlip()],
        additional_targets={
            'click_map':'image',
            'image':'image',
            'mask':'mask',
            'error_mask':'mask'}
        )
    
    clickmaps = cls.load(
        dataset_name='TETRIS',
        files=rclicks.paths.TETRIS_TRAIN,
        **kwargs
    )

    return ClickMapsDataset(
        clickmaps,
        image_normalization=img_normalization,
        transforms=transforms,
    )


def create_val_dataset(cls=rclicks.clickmap.CachedClickMaps,
                  **kwargs):
    img_normalization = alb.Compose(
        [
            alb.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
        ])
    transforms = None
    
    clickmaps = cls.load(
        dataset_name='TETRIS',
        files=rclicks.paths.TETRIS_VAL,
        **kwargs
    )

    return ClickMapsDataset(
        clickmaps,
        image_normalization=img_normalization,
        transforms=transforms,
    )


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            img, mask, error_mask, gt_clickmap = next(self.loader)
            self.next_img = img
            self.next_coord_feature = torch.cat((mask, error_mask), 1)
            self.next_gt_clickmap = gt_clickmap
        except StopIteration:
            self.next_img = None
            self.next_coord_feature = None
            self.next_gt_clickmap = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.cuda(non_blocking=True)
            self.next_coord_feature = self.next_coord_feature.cuda(non_blocking=True)
            self.next_gt_clickmap = self.next_gt_clickmap.cuda(non_blocking=True)

    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.next_img
        coord_feature = self.next_coord_feature
        gt_clickmap = self.next_gt_clickmap
        if img is None and coord_feature is None and gt_clickmap is None:
            raise StopIteration
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
        if coord_feature is not None:
            coord_feature.record_stream(torch.cuda.current_stream())
        if gt_clickmap is not None:
            gt_clickmap.record_stream(torch.cuda.current_stream())
        self.preload()
        return img, coord_feature, gt_clickmap


DEVICE = 'cuda'
NUM_EPOCHS = 20
LOGS_DIR = rclicks.paths.BENCHMARK_DIR / Path('logs')
OUT_DIR = rclicks.paths.BENCHMARK_DIR / Path('experiments/train')


@click.command('Train clickability model on the click maps')
@click.option('-s', '--sigma', type=float, help='Sigma radius')
@click.option('-c', '--cache_dir', type=Path,
              default=None,
              help='Directory with precalculated ground truth click maps, see scripts/generate_click_maps.py')
@click.option('-l', '--logs_dir', type=Path,
              default=None,
              help='Directory to save TB logs')
@click.option('-o', '--out_dir', type=Path,
              default=None,
              help='Directory to save models')
@click.option('--num_epochs', type=int,
              default=NUM_EPOCHS,
              help='Number of epochs')
def main(
    sigma,
    cache_dir,
    logs_dir,
    out_dir,
    num_epochs):

    if cache_dir is not None:
        train_dataset = create_train_dataset(cache_dir=cache_dir)
        val_dataset = create_val_dataset(cache_dir=cache_dir)
    else:
        train_dataset = create_train_dataset(
            cls=rclicks.clickmap.ClickMaps,
            sigma=sigma)
        val_dataset = create_val_dataset(
            cls=rclicks.clickmap.ClickMaps,
            sigma=sigma)
    
    model = SegNeXtSaliency(coord_feature_ch=2).to(DEVICE)

    cv2.setNumThreads(2)

    loader_kwargs = dict(
        batch_size=16,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    valid_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    EPOCH_LEN = len(train_loader)
    NUM_STEPS = num_epochs * EPOCH_LEN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = CosineLRScheduler(optimizer, t_initial=NUM_STEPS,
                    warmup_t=int(0.1 * NUM_STEPS), warmup_prefix=True)
    torch.cuda.empty_cache()

    if logs_dir is None:
        logs_dir = LOGS_DIR / f'sigma={sigma}'

    if out_dir is None:
        out_dir = OUT_DIR / f'sigma={sigma}'
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(logs_dir)
    
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000
    best_sim = 0
    counter = 0

    for epoch in range(num_epochs):
        train_losses = []
        valid_losses = []
        model.train()
        train_prefetcher = data_prefetcher(train_loader)
        loop = tqdm(train_prefetcher)
        running_loss = 0
        print(f'Starting training epoch {epoch}/{num_epochs}')
        for i_batch, (img, coord_feature, gt_clickmap) in enumerate(loop):
            optimizer.zero_grad()
            pred_clickmap = model(img, coord_feature)['instances']
            loss = kld(pred_clickmap, gt_clickmap)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            running_loss += loss_value
            loop.set_postfix(loss=loss_value)
            train_losses.append(loss_value)
            scheduler.step(counter)
            counter += 1

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i_batch)

        writer.add_scalar('Loss/train_epoch', running_loss, epoch)
        print(f'Finished training epoch {epoch}/{num_epochs}')

        torch.save(model.state_dict(), out_dir / f'epoch_{epoch}.pth')

        cc_list = []
        sim_list = []
        epoch_loss = 0
        print(f'Starting evaluating epoch {epoch}/{num_epochs}')
        with torch.no_grad():
            model.eval()
            valid_prefetcher = data_prefetcher(valid_loader)
            loop = tqdm(valid_prefetcher)
            running_loss = 0
            for i_batch, (img, coord_feature, gt_clickmap) in enumerate(loop):
                pred_clickmap = model(img, coord_feature)['instances']
                loss = kld(pred_clickmap, gt_clickmap)
                running_loss += loss.item()
                cc, sim = calculate_metrics(pred_clickmap, gt_clickmap,)
                cc_list.append(cc)
                sim_list.append(sim)
                valid_losses.append(loss.item())
                epoch_loss += loss.item()
                writer.add_scalar('Loss/val', loss.item(), epoch * len(valid_loader) + i_batch)
                
            print(
                "| Epoch: ", epoch, 
                "| Val Loss: ", np.mean(valid_losses), 
                "| Train Loss: ", np.mean(train_losses),
                "| CC: ", np.mean(cc_list),
                "| SIM: ", np.mean(sim_list)
            )

            writer.add_scalar('Loss/val_epoch', running_loss, epoch)
            writer.add_scalar('ValMetrics/CC', np.mean(cc_list), epoch)
            writer.add_scalar('ValMetrics/SIM', np.mean(sim_list), epoch)

        print(f'Finished evaluating epoch {epoch}/{num_epochs}')
        
        if best_sim < np.mean(sim_list):
            print(epoch)
            best_sim = np.mean(sim_list)
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), out_dir / 'best.pth')
        
    writer.close()
        
if __name__=='__main__':
    main()
import rclicks
import rclicks.clickmap
from tqdm import tqdm
from joblib import Parallel, delayed
import click
from pathlib import Path
import numpy as np

@click.command('Generate clicks maps for training clickability model')
@click.option('-s', '--sigma', type=float, help='Sigma radius')
@click.option('-c', '--cache_dir', type=Path,
              help='Directory to save results')
def main(sigma: float, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    clickmaps = rclicks.clickmap.ClickMaps.load('TETRIS', sigma=sigma)
    n = len(clickmaps)
    def cache_clickmaps(i):
        st = clickmaps.full_stem(i)
        outpath = cache_dir / f'{st}.npy'
        if outpath.is_file():
            return
        cm = clickmaps.click_map(i)
        np.save(outpath, cm)
    Parallel(n_jobs=15)(delayed(cache_clickmaps)(i) for i in tqdm(range(n)))
    
if __name__=='__main__':
    main()
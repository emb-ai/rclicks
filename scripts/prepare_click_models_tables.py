import pandas as pd
import numpy as np
from pathlib import Path
import click

scripts_path = Path(__file__).parent
EXPDIR = scripts_path.parent / 'experiments/eval_cm'
OUTDIR = EXPDIR

@click.command(help='Prepare clickability models evalution and ablation paper tables')
@click.option('-e', '--expdir', type=Path, 
              help='Path to directory with results of `eval_click_models.sh`')
@click.option('-o', '--outdir', type=Path, 
              help='Path to save processed tables')
def main(expdir, outdir):
    expdir = EXPDIR
    outdir = OUTDIR

    csv_files = [
        file
        for file in expdir.glob('*_per_image.csv')
    ]

    dfs = {csv.name.split('_')[0]: pd.read_csv(csv, index_col=False) for csv in csv_files}

    for dataset, df in dfs.items():
        df['dataset'] = dataset


    df = pd.concat(dfs.values())

    metrics_names = [
            'ks2d',
            'l1',
            'wasserstien',
            'nss',
            'pde',
    ]

    metrics_rename = {
        'ks2d': r'KS\,$\uparrow$',
        'l1': r'PL$_1$\,$\downarrow$',
        'wasserstien': r'WD\,$\downarrow$',
        'nss': r'NSS\,$\uparrow$',
        'pde': r'PDE\,$\uparrow$',
    }


    df_mean = df.groupby(['dataset', 'model_name', ], group_keys=False)[metrics_names].mean().reset_index()


    def ablation_table():
        cm_list = [
            'cm_001',
            'cm_025',
            'cm_05',
            'cm_1',
            'cm_2',
            'cm',
            'cm_10',
            'cm_20',
            'cm_30',
            'cm_60',
            'cm_90',
            'cm_120',
        ]

        cm_rename = {
            'cm_001': r'CM$_{0.01}$',
            'cm_025': r'CM$_{0.25}$',
            'cm_05': r'CM$_{0.5}$',
            'cm_1': r'CM$_{1}$',
            'cm_2': r'CM$_{2}$',
            'cm': r'CM$_{5}$',
            'cm_10': r'CM$_{10}$',
            'cm_20': r'CM$_{20}$',
            'cm_30': r'CM$_{30}$',
            'cm_60': r'CM$_{60}$',
            'cm_90': r'CM$_{90}$',
            'cm_120': r'CM$_{120}$',
        }

        df_tetris = df_mean[df_mean.dataset == 'TETRIS']
        df_ablation = df_tetris[df_tetris.model_name.isin(cm_list)].drop(columns=['dataset'])
        df_ablation = df_ablation.set_index('model_name').reindex(cm_list)

        df_ablation.ks2d = df_ablation.ks2d.apply(lambda x: f'{x:.2f}')
        df_ablation.l1 = df_ablation.l1.apply(lambda x: f'{x:.3f}')
        df_ablation.wasserstien = df_ablation.wasserstien.apply(lambda x: f'{x:.2f}')
        df_ablation.nss = df_ablation.nss.apply(lambda x: f'{x:.2f}')
        df_ablation.pde = df_ablation.pde.apply(lambda x: f'{x:.2E}')

        df_ablation = df_ablation.rename(columns=metrics_rename, index=cm_rename).reset_index()
        df_ablation = df_ablation.rename(columns={'model_name': 'Train data'})
        df_ablation.to_csv(outdir / 'ablation_cm_sigma_tetris.csv', index=False)

    ablation_table()

    def main_paper_table_3():
        models_names = ['ud', 'dt', 'sm', 'cm',]
        models_rename = {
            'ud': 'UD',
            'dt': 'DT',
            'sm': 'SM',
            'cm': 'Ours',
        }

        df_tetris = df_mean[df_mean.dataset == 'TETRIS']
        df_main = df_tetris[df_tetris.model_name.isin(models_names)].drop(columns=['dataset'])
        df_main = df_main.set_index('model_name').reindex(models_names)

        df_main.ks2d = df_main.ks2d.apply(lambda x: f'{x:.2f}')
        df_main.l1 = df_main.l1.apply(lambda x: f'{x:.2f}')
        df_main.wasserstien = df_main.wasserstien.apply(lambda x: f'{x:.2f}')
        df_main.nss = df_main.nss.apply(lambda x: f'{x:.2f}')
        df_main.pde = df_main.pde.apply(lambda x: f'{x:.2E}')

        df_main = df_main.rename(columns=metrics_rename, index=models_rename).reset_index()
        df_main = df_main.rename(columns={'model_name': 'Model'})
        df_main.to_csv(outdir / 'eval_cm_tetris.csv', index=False)
    
    main_paper_table_3()

    # Metrics for other datasets
    models_names = ['ud', 'dt', 'si', 'sm', 'cm',]
    models_rename = {
        'ud': 'UD',
        'dt': 'DT',
        'si': 'SI',
        'sm': 'SM',
        'cm': 'Ours',
    }

    dataset_names = [
        'GrabCut',
        'Berkeley',
        'DAVIS',
        'COCO',
        'TETRIS',
    ]

    dataset_rename = {
        'GrabCut': 'GrabCut',
        'Berkeley': 'Berkeley',
        'DAVIS': 'DAVIS',
        'COCO': 'COCO-MVal',
        'TETRIS': 'TETRIS (Val)',
    }

    # df_tetris = df_mean[df_mean.dataset == 'TETRIS']
    df_main = df_mean[df_mean.model_name.isin(models_names)]
    df_main = df_main.set_index(['dataset', 'model_name'])
    df_main = df_main.reindex(models_names, level=1)
    df_main = df_main.reindex(dataset_names, level=0)
    df_main = df_main.reset_index()

    df_main.ks2d = df_main.ks2d.apply(lambda x: f'{x:.2f}')
    df_main.l1 = df_main.l1.apply(lambda x: f'{x:.2f}')
    df_main.wasserstien = df_main.wasserstien.apply(lambda x: f'{x:.2f}')
    df_main.nss = df_main.nss.apply(lambda x: f'{x:.2f}')
    df_main.pde = df_main.pde.apply(lambda x: f'{x:.2E}')

    df_main = df_main.rename(columns=metrics_rename)
    df_main = df_main.rename(columns={'model_name': 'Model', 'dataset': 'Dataset'})
    df_main = df_main.set_index(['Dataset', 'Model'])
    df_main = df_main.rename(index=models_rename, level=1)
    df_main = df_main.rename(index=dataset_rename, level=0)
    df_main = df_main.reset_index()
    df_main.to_csv(outdir / 'eval_cm_all.csv', index=False)

main()
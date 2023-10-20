import scanpy as sc
import pandas as pd

import sys
sys.path.append('/opt/data/private/nips_single_cell_2023-master/model')
import cpa

sc.settings.set_figure_params(dpi=100)
adata = sc.read('/opt/data/private/nips_single_cell_2023-master/data/processed_data/adata.h5ad')

cpa.CPA.setup_anndata(adata,
                      perturbation_key='sm_name',
                      control_group=None,
                      batch_key=None,
                      is_count_data=True,
                      categorical_covariate_keys=['cell_type'],
                      max_comb_len=1,
                     )

ae_hparams = {
    "n_latent": 128,
    "recon_loss": "gauss",
    "doser_type": "logsigm",
    "n_hidden_encoder": 512,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.1,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 0,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 3,
    "mixup_alpha": 0.1,
    "adv_steps": 2,
    "n_hidden_adv": 64,
    "n_layers_adv": 2,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 20.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 45,
}

model = cpa.CPA(adata=adata,
                split_key='split',
                train_split='train',
                valid_split='valid',
                test_split='ood',
                **ae_hparams,
               )

model.train(max_epochs=2000,
            use_gpu=True,
            batch_size=512,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path='/opt/data/private/nips_single_cell_2023-master/cpa/save',
           )

id_map = pd.read_csv('/opt/data/private/nips_single_cell_2023-master/data/id_map.csv')
sample_submission = pd.read_csv('/opt/data/private/nips_single_cell_2023-master/data/sample_submission.csv', index_col='id')

adata_test = sc.AnnData(X=sample_submission.values) 
adata_test.obs['cell_type'] = id_map['cell_type'].tolist()
adata_test.obs['sm_name'] = id_map['sm_name'].tolist()
adata_test.var['gene'] = adata.var['gene'].tolist().copy()

cpa.CPA.setup_anndata(adata_test,
                      perturbation_key='sm_name',
                      control_group=None,
                      batch_key=None,
                      is_count_data=True,
                      categorical_covariate_keys=['cell_type'],
                      max_comb_len=1,
                     )

model = cpa.CPA.load(dir_path='/opt/data/private/nips_single_cell_2023-master/cpa_save',
                     adata=adata, use_gpu=True)

model.predict(adata_test, batch_size=1024)

submission = pd.DataFrame(adata_test.obsm['CPA_pred'], columns=adata.var_names)
submission.index.name = 'id'
submission.to_csv('/opt/data/private/nips_single_cell_2023-master/submission/submission.csv')

# cpa.pl.plot_history(model)

print('cpa end')
import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

wandb.require("core")


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(cfg: DictConfig):
    ckpt_callback = ModelCheckpoint(save_top_k=0, save_last=True)
    logger = WandbLogger(
        log_model=False, project=cfg.project, mode=cfg.wandb_mode, config=OmegaConf.to_container(cfg, resolve=True)
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_callback],
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        log_every_n_steps=cfg.log_every_n_steps,
        overfit_batches=cfg.overfit_batches,
    )

    module = instantiate(cfg.module)
    dm = instantiate(cfg.data)
    trainer.fit(module, dm)

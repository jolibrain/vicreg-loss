import hydra
from omegaconf import DictConfig, OmegaConf

from src.model import VICRegModel
from src.trainer import VICRegTrainer
from src.vicreg import VICRegLoss


@hydra.main(config_path="configs", config_name="normal", version_base="1.3")
def main(config: DictConfig):
    model = VICRegModel(
        in_channels=config.model.in_channels,
        n_layers=config.model.n_layers,
        hidden_size=config.model.hidden_size,
        representation_size=config.model.representation_size,
    )
    loss = VICRegLoss(
        inv_coeff=config.loss.inv_coeff,
        var_coeff=config.loss.var_coeff,
        cov_coeff=config.loss.cov_coeff,
        gamma=config.loss.gamma,
    )
    trainer = VICRegTrainer(
        model,
        loss,
        learning_rate=config.trainer.learning_rate,
        batch_size=config.trainer.batch_size,
    )
    trainer.launch_training(
        config.trainer.n_epochs,
        config.trainer.device,
        OmegaConf.to_container(config),
    )


if __name__ == "__main__":
    # Launch training using hydra.
    main()

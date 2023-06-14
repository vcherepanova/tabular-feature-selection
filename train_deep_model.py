import json
import logging
import os
from collections import OrderedDict
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import deep_tabular as dt

@hydra.main(config_path="config", config_name="train_model")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_deep_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.hyp.save_period < 0:
        cfg.hyp.save_period = 1e8
    torch.manual_seed(cfg.hyp.seed)
    torch.cuda.manual_seed_all(cfg.hyp.seed)
    torch.backends.cudnn.deterministic = True

    loaders, unique_categories, n_numerical, n_classes = dt.utils.get_dataloaders(cfg)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.model,
                                                                                 n_numerical,
                                                                                 unique_categories,
                                                                                 n_classes,
                                                                                 device)


    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(f"This {cfg.model.name} has {pytorch_total_params / 1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")

    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer_for_single_net(cfg.hyp,
                                                                                      net,
                                                                                      optimizer_state_dict)
    criterion = dt.utils.get_criterion(cfg.dataset.task)
    train_setup = dt.TrainingSetup(criterions=criterion,
                                   optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler)

    ####################################################
    #  Train
    log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -np.inf
    done = False
    epoch = start_epoch
    best_epoch = epoch

    while not done and epoch < cfg.hyp.epochs:

        loss, grad_avg = dt.default_training_loop(net, loaders["train"], train_setup, device, cfg.hyp)
        log.info(f"Training loss at epoch {epoch}: {loss}")

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.hyp.val_period == 0 or epoch + 1 == cfg.hyp.epochs:
            test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                                   [loaders["test"], loaders["val"], loaders["train"]],
                                                                   cfg.dataset.task,
                                                                   device)
            log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
            log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
            log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

        if cfg.hyp.use_patience:
            val_stats, test_stats = dt.evaluate_model(net,
                                                      [loaders["val"], loaders["test"]],
                                                      cfg.dataset.task,
                                                      device)

            if val_stats["score"] > highest_val_acc_so_far:
                best_epoch = epoch
                highest_val_acc_so_far = val_stats["score"]
                log.info(f"New best epoch, val score: {val_stats['score']}")
                # save current model
                state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
                out_str = "model_best.pth"
                log.info(f"Saving model to: {out_str}")
                torch.save(state, out_str)

            if epoch - best_epoch > cfg.hyp.patience:
                done = True
        epoch += 1

    log.info("Running Final Evaluation...")
    checkpoint_path = "model_best.pth"
    net.load_state_dict(torch.load(checkpoint_path)["net"])
    test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                           [loaders["test"], loaders["val"], loaders["train"]],
                                                           cfg.dataset.task,
                                                           device)

    log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
    log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
    log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

    stats = OrderedDict([("dataset", cfg.dataset.name),
                         ("model_name", cfg.model.name),
                         ("best_epoch", best_epoch),
                         ("routine", "from_scratch"),
                         ("test_stats", test_stats),
                         ("train_stats", train_stats),
                         ("val_stats", val_stats)])
    if cfg.mode=='downstream':
        with open(os.path.join("stats.json"), "w") as fp:
            json.dump(stats, fp, indent=4)
        log.info(json.dumps(stats, indent=4))


    if cfg.mode=='feature_selection':
        if cfg.hyp.regularization=='deep_lasso':
            from deep_tabular.utils.testing import get_feat_importance_deeplasso
            importances = get_feat_importance_deeplasso(net, loaders["val"], criterion, device)
            torch.save(importances, f'./feature_importances.pt')
        elif cfg.hyp.regularization=='lasso':
            from deep_tabular.utils.testing import get_feat_importance_lasso
            importances = get_feat_importance_lasso(net)
            torch.save(importances, f'./feature_importances.pt')
        elif cfg.hyp.regularization=='first_lasso':
            from deep_tabular.utils.testing import get_feat_importance_firstlasso
            importances = get_feat_importance_firstlasso(net)
            torch.save(importances, f'./feature_importances.pt')
        elif cfg.model.name == 'ft_transformer_attention_map':
            from deep_tabular.utils.testing import get_feat_importance_attention
            average_cls_attention_map, importances, feature_ranks, \
            feature_indices_sorted_by_importance = get_feat_importance_attention(net.module, loaders["val"], device)
            torch.save(importances, f'./feature_importances.pt')
        else:
            raise NotImplementedError
    if cfg.mode == 'feature_selection':
        return stats, importances
    else:
        return stats

if __name__ == "__main__":
    main()

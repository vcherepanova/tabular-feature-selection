from dataclasses import dataclass
from typing import Any
import torch
import torch.autograd as autograd

def add_dimension_glasso(var, dim=0):
    return var.pow(2).sum(dim=dim).add(1e-8).pow(1/2.).sum()

@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    criterions: Any
    optimizer: Any
    scheduler: Any
    warmup: Any
    num_datasets_in_batch: Any = None


def default_training_loop(net, trainloader, train_setup, device, hyp):
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    criterion = train_setup.criterions

    train_loss = 0
    total = 0

    for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(trainloader):
        inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
        inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                 inputs_cat if inputs_cat.nelement() != 0 else None
        if batch_idx==0:
            grad_avg = torch.zeros([inputs_num.shape[1]])
        inputs_num.requires_grad_()
        optimizer.zero_grad()
        outputs = net(inputs_num, inputs_cat)
        loss = criterion(outputs, targets)

        # add regularization
        if hyp.regularization=='deep_lasso':
            grad_params = autograd.grad(loss, inputs_num, create_graph=True, allow_unused=True)
            reg = add_dimension_glasso(grad_params[0], dim=0)
            loss = hyp.reg_weight*reg + (1-hyp.reg_weight)*loss
        elif hyp.regularization=='lasso':
            reg = add_dimension_glasso(net.module.head.weight)
            loss = hyp.reg_weight*reg + (1-hyp.reg_weight)*loss
        elif hyp.regularization=='first_lasso':
            reg = add_dimension_glasso(net.module.layers[0].weight)
            loss = hyp.reg_weight * reg + (1 - hyp.reg_weight) * loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)

        if hyp.regularization == 'deep_lasso':
            grad_avg += grad_params[0].detach().cpu().abs().mean(0)
            del grad_params

    train_loss = train_loss / (batch_idx + 1)
    lr_scheduler.step()
    warmup_scheduler.dampen()
    return train_loss, grad_avg


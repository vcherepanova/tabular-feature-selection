import torch
import torch.autograd as autograd


def add_dimension_glasso(var, dim=0):
    return var.pow(2).sum(dim=dim).add(1e-8).pow(1/2.).sum()


def deep_lasso_regularizer(loss, inputs):
    grad_params = autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
    regval = add_dimension_glasso(grad_params[0], dim=0)
    return regval


def get_feat_importance_deeplasso(net, testloader, criterion, device):
    net.eval()
    grads = []
    for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(testloader):
        inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
        inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                    inputs_cat if inputs_cat.nelement() != 0 else None
        inputs_num.requires_grad_()
        outputs = net(inputs_num, inputs_cat)
        loss = criterion(outputs, targets)
        grad_params = autograd.grad(loss, inputs_num, create_graph=True, allow_unused=True)
        grads.append(grad_params[0].detach().cpu())

    grads = torch.cat(grads)
    importances = grads.pow(2).sum(dim=0).pow(1/2.)
    return importances
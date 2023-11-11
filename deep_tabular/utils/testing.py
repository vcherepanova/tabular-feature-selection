import torch
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score, roc_auc_score
import torch.autograd as autograd
import scipy

def evaluate_model(net, loaders, task, device):
    scores = []
    for loader in loaders:
        score = test_default(net, loader, task, device)
        scores.append(score)
    return scores


def test_default(net, testloader, task, device):
    net.eval()
    targets_all = []
    predictions_all = []
    with torch.no_grad():
        for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(testloader):
            inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
            inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None

            outputs = net(inputs_num, inputs_cat)
            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = outputs
            elif task == "regression":
                predicted = outputs
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predicted.cpu().tolist())

    if task == "multiclass":
        accuracy = accuracy_score(targets_all, predictions_all)
        balanced_accuracy = balanced_accuracy_score(targets_all, predictions_all, adjusted=False)
        balanced_accuracy_adjusted = balanced_accuracy_score(targets_all, predictions_all, adjusted=True)
        scores = {"score": accuracy,
                  "accuracy": accuracy,
                  "balanced_accuracy": balanced_accuracy,
                  "balanced_accuracy_adjusted": balanced_accuracy_adjusted}
    elif task == "regression":
        rmse = mean_squared_error(targets_all, predictions_all, squared=False)
        scores = {"score": -rmse,
                  "rmse": -rmse}
    elif task == "binclass":
        roc_auc = roc_auc_score(targets_all, predictions_all)
        scores = {"score": roc_auc,
                  "roc_auc": roc_auc}
    return scores


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


def get_feat_importance_lasso(net):
    importances = net.module.head.weight.detach().cpu().pow(2).sum(dim=0).pow(1 / 2.)
    return importances

def get_feat_importance_firstlasso(net):
    importances = net.module.layers[0].weight.detach().cpu().pow(2).sum(dim=0).pow(1 / 2.)
    return importances


class SaveAttentionMaps:
    def __init__(self):
        self.attention_maps = None
        #self.n_batches = 0

    def __call__(self, _, __, output):
        if self.attention_maps is None:
            self.attention_maps = output[1]['attention_probs'].detach().cpu().sum(0)
        else:
            self.attention_maps+=output[1]['attention_probs'].detach().cpu().sum(0)

def get_feat_importance_attention(net, testloader, device):
    net.eval()
    hook = SaveAttentionMaps()
    for block in net.layers:
        block['attention'].register_forward_hook(hook)

    for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(testloader):
        inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
        inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                 inputs_cat if inputs_cat.nelement() != 0 else None

        net(inputs_num, inputs_cat)

    n_blocks = len(net.layers)
    n_objects = len(testloader.dataset)
    n_heads = net.layers[0]['attention'].n_heads
    n_features = inputs_num.shape[1]
    n_tokens = n_features + 1
    attention_maps = hook.attention_maps
    average_attention_map = attention_maps/(n_objects*n_blocks*n_heads)
    assert attention_maps.shape == (n_tokens, n_tokens)

    # Calculate feature importance and ranks.
    average_cls_attention_map = average_attention_map[0]  # consider only the [CLS] token
    feature_importance = average_cls_attention_map[1:]  # drop the [CLS] token importance
    assert feature_importance.shape == (n_features,)

    feature_ranks = scipy.stats.rankdata(-feature_importance.numpy())
    feature_indices_sorted_by_importance = feature_importance.argsort(descending=True).numpy()
    return average_cls_attention_map, feature_importance, feature_ranks, feature_indices_sorted_by_importance

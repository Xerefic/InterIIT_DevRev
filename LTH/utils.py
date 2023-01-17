from imports import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def random_prune(model, px):
    prune_params = []
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            prune_params.append((m, "weight"))

    prune_params = tuple(prune_params)
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def l1_prune(model, px):
    prune_params = []
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            prune_params.append((m, "weight"))

    prune_params = tuple(prune_params)
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def l1_prune_custom(model, px, prune_ff_only=False):
    prune_params = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "encoder" in name and "attention" not in name:
                prune_params.append((m, "weight"))
            elif not prune_ff_only and "head" not in name:
                prune_params.append((m, "weight"))
    prune_params = tuple(prune_params)
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def remove_prune_custom(model, prune_ff_only=False):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "encoder" in name and "attention" not in name:
                prune.remove(m, "weight")
            elif not prune_ff_only and "head" not in name:
                prune.remove(m, "weight")


def remove_prune(model):
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            prune.remove(m, "weight")

def mask_prune_custom(model, mask_dict, prune_ff_only=False):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "encoder" in name and "attention" not in name:
                prune.CustomFromMask.apply(m, "weight", mask=mask_dict[name + ".weight_mask"])
            elif not prune_ff_only and "head" not in name:
                prune.CustomFromMask.apply(m, "weight", mask=mask_dict[name + ".weight_mask"])

def mask_prune(model, mask_dict):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            prune.CustomFromMask.apply(m, "weight", mask=mask_dict[name + ".weight_mask"])

def extract_mask(model_dict):
    mask_dict = {}
    for key in model_dict.keys():
        if "mask" in key:
            mask_dict[key] = copy.deepcopy(model_dict[key])
    return mask_dict

def check_sparsity_custom(model, prune_ff_only=False):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "encoder" in name and "attention" not in name:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))
            elif not prune_ff_only and "head" not in name:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    return 100 * (1 - zero_sum / sum_list)


def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))
    return 100 * (1 - zero_sum / sum_list)

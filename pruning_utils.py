import torch
import torch_pruning as pruning
import torch.nn.utils.prune as prune


def num_prunable_layers(model):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            count += 1
        elif isinstance(module, torch.nn.Linear):
            count += 1
    return count - 1


def prune_with_strategy(model_to_be_pruned, reference_model, strategy):
    DG = pruning.DependencyGraph()
    DG.build_dependency(model_to_be_pruned,
                        example_inputs=torch.randn(4, 3, 32, 32))
    num_pruned = 0
    i = 0
    for orig, pruned in zip(reference_model.named_modules(), model_to_be_pruned.named_modules()):
        name_orig, module_orig = orig
        name_pruned, module_pruned = pruned

        if isinstance(module_orig, torch.nn.Conv2d):
            prune_ratio = strategy[i]
            prune.ln_structured(module_orig, 'weight', prune_ratio, 2, 0)
            prune.remove(module_orig, 'weight')
            idxs = torch.where(module_orig.weight.sum(
                dim=(1, 2, 3)) == 0)[0].tolist()
            pruning_plan = DG.get_pruning_plan(
                module_pruned, pruning.prune_conv, idxs=idxs)
            # print(pruning_plan)
            num_pruned += pruning_plan.exec()
            i += 1
        elif isinstance(module_orig, torch.nn.Linear):
            if name_orig == "classifier.6":
                continue
            prune_ratio = strategy[i]
            prune.ln_structured(module_orig, 'weight', prune_ratio, 2, 0)
            prune.remove(module_orig, 'weight')
            idxs = torch.where(module_orig.weight.sum(dim=1) == 0)[0].tolist()
            pruning_plan = DG.get_pruning_plan(
                module_pruned, pruning.prune_linear, idxs=idxs)
            # print(pruning_plan)
            num_pruned += pruning_plan.exec()
            i += 1
    return model_to_be_pruned, num_pruned

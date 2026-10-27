import torch


def load_checkpoint(path , models , optimizers , device):
    # 恢复各网络、优化器与计数，回放数据重新采集；兼容原来的检查点
    checkpoint = torch.load(path , map_location = device , weights_only = True)
    for name , model in models.items():
        model.load_state_dict(checkpoint[name])
    for name , optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[name])
    return checkpoint["total_steps"] , checkpoint["total_updates"] , checkpoint["episodes_completed"]


def save_checkpoint(path , models , optimizers , total_steps , total_updates , episodes_completed):
    # 完整写入临时文件后再替换检查点，保持原有存档键名
    checkpoint = {name: model.state_dict() for name , model in models.items()}
    checkpoint.update({name: optimizer.state_dict() for name , optimizer in optimizers.items()})
    checkpoint.update({
        "total_steps": total_steps , "total_updates": total_updates , "episodes_completed": episodes_completed
    })
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint , temporary_path)
    temporary_path.replace(path)

from torch.optim import Adam, SGD, RMSprop


def call_optimizer(optim_name: str):
    # NOTE: Optimizer settings
    if optim_name.lower() == 'adam':
        optimizer = Adam
    elif optim_name.lower() == 'sgd':
        optimizer = SGD
    elif optim_name.lower() == 'rmsprop':
        optimizer = RMSprop
    else:
        raise NotImplementedError()
    return optimizer

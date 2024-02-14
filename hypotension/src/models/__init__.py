from .cnn import CausalOneDimCNN, AttentionCNN, Net
from .rnn import ValinaLSTM
from .galr import GALRBlock
from .attention_galr import XGALR
from .prev_galr_model import GALR_best_acc


def call_models(model_name: str, **kwargs):

    # TODO: Need to be re-arranged in the future.
    # NOTE: Model Setting
    if model_name.lower() == "lstm":
        model = ValinaLSTM(**kwargs)
    elif model_name.lower() == 'xgalr':
        model = XGALR(**kwargs)
    # elif model_name.lower() == 'multihead_attention':
    #     model = AttentionCNN(input_size=8,
    #                          embedding_dim=kwargs['hidden_dim'],
    #                          attention_dim=kwargs['attention_dim'],
    #                          sequences=3000,
    #                          num_heads=kwargs['num_heads'],
    #                          num_of_classes=2)
    # elif model_name.lower() == 'galr':
    #     model = GALRBlock(num_features=features,
    #                       hidden_channels=kwargs['hidden_channels'],
    #                       num_heads=kwargs['num_heads'],
    #                       norm=kwargs['norm'],
    #                       dropout=kwargs['dropout'],
    #                       num_of_classes=2,
    #                       block_num=1)
    #
    # elif model_name.lower() == 'attention_galr':
    #     model = AttentiveGALR(input_size=features,
    #                           # embedding_dim=kwargs['embedding_dim'],
    #                           sequences=sequences,
    #                           num_heads=kwargs['num_of_heads'],
    #                           # chunk_size=kwargs['chunk_size'],
    #                           # hop_size=kwargs['hop_size'],
    #                           # hidden_channels=kwargs['hidden_channels'],
    #                           # low_dimension=kwargs['low_dimension'],
    #                           # num_layers=kwargs['num_layers'],
    #                           num_classes=num_of_classes,
    #                           **kwargs)
    # elif model_name.lower() == 'multihead_attention_galr':
    #     model = MultiheadAttentionGALR(
    #         input_size=features,
    #         sequences=sequences,
    #         num_heads=kwargs['num_of_heads'],
    #         num_classes=num_of_classes,
    #         **kwargs
    #     )
    # elif model_name.lower() == 'multihead_attention_galr1':
    #     model = MultiheadAttentionGALR1(
    #         input_size=features,
    #         sequences=sequences,
    #         num_heads=kwargs['num_of_heads'],
    #         num_classes=num_of_classes,
    #         **kwargs
    #     )
    elif model_name.lower() == "yonsei":
        model = Net()
    # elif model_name.lower() == "galr_prev":
    #     model = GALR_best_acc(num_features=features, num_heads=kwargs['num_of_heads'],
    #                           hidden_channels=kwargs['hidden_channels'])
    else:
        raise NotImplementedError()
    return model

import argparse
import torch

from torch.utils.data import DataLoader
from src.utils.data_loader import load_files, HDF5_VitalDataset
from src.train.train_function import TrainWrapper
from distutils.util import strtobool
from src.models import call_models


def data_setting(dataset_path, split_ratio=0.2, seed_fix=True):
    # NOTE: Data loading
    # Should not shuffle different manner each time. (fixed random seed)
    train_list, test_list = load_files(dataset_path, split_ratio, fixed=seed_fix)

    # NOTE: Data setting
    print("Loading the Training set from disk: {}".format(dataset_path))
    train_dataset = HDF5_VitalDataset(train_list[:50])# 양 줄여두기

    print("Making Validation Set from Training set.")
    valid_dataset = train_dataset.set_valid()

    print("Loading the Test set from dist.")
    test_dataset = HDF5_VitalDataset(test_list[:50]) #일부만 자르기

    return train_dataset, valid_dataset, test_dataset


def load_test_data(dataset_path, split_ratio, seed_fix):
    # NOTE: Data loading
    # Should not shuffle different manner each time. (fixed random seed)
    _, test_list = load_files(dataset_path, split_ratio, fixed=seed_fix)
    print("Loading the Test set from dist.")
    test_dataset = HDF5_VitalDataset(test_list[500:])
    return test_dataset


def main(model_settings):

    if args.pretrained:
        model = call_models(args.model_name, **model_settings)
        param = torch.load(input("Saved model path: "))

        state_dict = param['state_dict']

        #Part for classifier normalization
        #################################
        #head = state_dict['fc.6.weight']
        #r = torch.norm(head, dim=1)
        #r = torch.pow(r, 1.0)  #-1.65
        #r = r[:, None]
        #head = torch.div(head, r)
        #state_dict['fc.6.weight'] = head

        model.load_state_dict(state_dict)
    else:
        model = call_models(args.model_name, **model_settings)

    # INFO: Data preparation.
    if args.train:
        train_dataset, valid_dataset, test_dataset = data_setting(args.data_path,
                                                                  split_ratio=args.split_ratio,
                                                                  seed_fix=args.seed_fix)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        train_wrapper = TrainWrapper(train_loader, valid_loader, test_loader, model, args.log_path,
                                     train_settings=train_settings)
        print("Loading complete")

        train_wrapper.fit()
        train_wrapper.save_hyperparameter(model_settings, name="model_settings")
        train_wrapper.test_ci(test_loader)
        #train_wrapper.test(test_loader)

    test_dataset = load_test_data(args.data_path, split_ratio=args.split_ratio, seed_fix=args.seed_fix)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    #train_wrapper.test(test_loader)

    train_wrapper = TrainWrapper(test_loader, test_loader, test_loader, model, args.log_path, train_settings=train_settings)

    train_wrapper.test(test_loader) # we don't consider confidence interval for calculation effeciency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # NOTE: Data path argument
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.2)
    parser.add_argument("--seed_fix", type=lambda x: bool(strtobool(x)), default=True)

    # NOTE: Call pretrained model
    parser.add_argument("--pretrained", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=True)

    # NOTE: Train with new model
    parser.add_argument("--model_name", type=str)

    # NOTE: Train settings
    parser.add_argument("--batch_size", type=int, default=512)##

    # NOTE: Log setting argument
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--summary_step", type=int, default=500)

    args = parser.parse_args()

    lstm_model_settings = {
        'features': 8,
        'number_of_classes': 2,
        'hidden_units': 16,
        'bidirectional': False,
        'layers': 3,
    }

    multihead_GALR_model_settings = {
        'input_size': 8,  ##MAC is dropped. 8 -> 7
        'num_classes': 2,
        'hidden_units': 16,
        'bidirectional': False,
        'gembedding_dim': 8,  # embedding on time 8->7
        'embedding_dim': 300,  # embedding on feature
        'sequences': 3000,
        'num_heads': 4,## original: 4
        'chunk_size': 100,
        'hop_size': 100,
        'hidden_channels': 32,
        'feature_attn': False,
        'low_dimension': False,
        'linear': True,
        'save_attn': True,#False, ## for save map
        'num_blocks': 3,
        'T': 0.5
    }

    train_settings = {
        # 'test_label_count': test_dataset.label_counts(),
        'save_count': args.summary_step,
        'optimizer': "sgd",
        'loss_fn': "focal",#
        'alpha': 0.3,
        'gamma': 5.0,
        'batch_size': args.batch_size,
        'test_batch': 256,
        'momentum': 0.8,
        'lr': 0.005,##
        'epochs': 10,##
        'weight_decay': 0,
        'save_attn': True #False ## for save map
    }

    main(multihead_GALR_model_settings)



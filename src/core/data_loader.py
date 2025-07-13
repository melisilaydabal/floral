import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.extract_latent_reps import FeatureExtractor
from src.models.curie_filter import curie_filtering
from src.models.label_sanitization import label_sanitize
from src.data.dataset import load_data, use_train_adv_data, use_alfa_train_adv_data, load_adv_train_data
from src.data.preprocess import Preprocessor

def prepare_data(args, config, seed):
    # Load full dataset (and test_adv if available)
    data = load_data(
        args,
        dataset=config['data']['dataset'],
        in_dim=config['data']['in_dim'],
        dir_dump_data=config['data']['dir_dump_data'],
    )
    X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv = data

    # Optionally use poisoned train_adv data
    if config['data']['is_train_adv_data']:
        data = use_train_adv_data(args, dataset=config['data']['dataset'],
                                  in_dim=config['data']['in_dim'],
                                  adv_rate=config['data']['adv_rate'],
                                  dir_dump_data=config['data']['dir_dump_data'])
        X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv = data

    if config['data']['is_train_adv_alfa_data']:
        X_train, y_train = use_alfa_train_adv_data(args, dataset=config['data']['dataset'],
                                                   in_dim=config['data']['in_dim'],
                                                   adv_rate=config['data']['adv_rate'],
                                                   dir_dump_data=config['data']['dir_dump_data'])
        X_all = torch.cat((X_train, X_val, X_test), dim=0)
        y_all = torch.cat((y_train, y_val, y_test), dim=0)

    # Curie Filtering
    if config['data']['is_curie_filter']:
        X_train, y_train = curie_filtering(config, seed, X_train, y_train,
                                           weight=0.05, count=20, theta=0.75)

    # Label Sanitization
    if config['data']['is_label_sanitize']:
        y_train = label_sanitize(config, seed, X_train, y_train, k=20, eta=0.75)

    # Feature Extraction
    if config['feature_extract']['is_feature_extract']:
        extractor = FeatureExtractor(
            in_dim=config['data']['in_dim'],
            out_dim=config['data']['num_classes'],
            model_path=config['feature_extract']['model_path']
        )
        loaders = {
            'train': torch.utils.data.DataLoader(TensorDataset(X_train, y_train),
                                                 batch_size=config['training']['batch_size'], shuffle=False),
            'val': torch.utils.data.DataLoader(TensorDataset(X_val, y_val),
                                               batch_size=config['training']['batch_size'], shuffle=False),
            'test': torch.utils.data.DataLoader(TensorDataset(X_test, y_test),
                                                batch_size=config['training']['batch_size'], shuffle=False),
        }
        X_train, y_train = extractor.extract_features(loaders['train'])
        X_val, y_val = extractor.extract_features(loaders['val'])
        X_test, y_test = extractor.extract_features(loaders['test'])

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'X_all': X_all, 'y_all': y_all,
        'y_test_adv': y_test_adv
    }

def get_data_loaders(config, data):
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=config.get('training').get('batch_size'), shuffle=True)
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    val_loader = DataLoader(val_dataset, batch_size=config.get('training').get('batch_size'))
    test_dataset = TensorDataset(data['X_test'], data['y_test'])
    test_loader = DataLoader(test_dataset, batch_size=config.get('training').get('batch_size'))
    return train_loader, val_loader, test_loader



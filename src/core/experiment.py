import os
import copy
from torch.utils.data import DataLoader, TensorDataset
from src.core.data_loader import prepare_data, get_data_loaders
from src.utils.logger import setup_experiment_folder
from src.players.player_model import Player_Model
from src.players.player_flip import Player_Flip
from src.utils.metrics import dump_metrics, dump_general_metric
from src.utils.plotter import plot_adv_labels

# Define some helpers
LIST_SMALL_DATASETS = ['moon']
LIST_LARGE_DATASETS = ['imdb', 'mnist_1vs7']
LIST_NN_MODELS = ['nn', 'nn_pgd']
LIST_SVM_MODELS = ['svm', 'ln-robust-svm']
LIST_SVM_MODELS_W_PGD_TRAIN = ['svm', 'ln-robust-svm']

def run_experiment(args, config, seed):
    # Create output directory for the current run
    dir_exp_folder = setup_experiment_folder(config, seed)

    # Load and preprocess data
    data = prepare_data(args, config, seed)
    X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv = (
        data['X_all'], data['y_all'], data['X_train'], data['y_train'],
        data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['y_test_adv'])
    train_loader, val_loader, test_loader = get_data_loaders(config, data)

    # Initialize players
    player_model = Player_Model(config)
    player_flip = Player_Flip(config, args.seed)

    train_adv_loader = None
    flip_history = []
    alphas_history = []
    # Set warmup rounds for adversarial training of SVM models, if = -1 means no warmup
    if config.get('model').get('name') in LIST_NN_MODELS:
        warmup_rounds = -1
    else:
        warmup_rounds = 0

    for round in range(int(config.get('game').get('num_rounds'))):
        print(f"Round: {round}")

        lr = config.get('optim').get('lr')
        # Decay learning rate
        factor = 10
        interval = 100
        if round % interval == 0 and round > 0:
            lr /= factor

        ######## PLAYER: MODEL ########
        if round == 0:
            # if config.get('model').get('name') in LIST_SVM_MODELS and config.get('data').get(
            #         'dataset') not in LIST_LARGE_DATASETS:
            #     player_model.compute_kernel_eigenvals(config, dir_exp_folder, X_train)

            if config.get('model').get('is_baseline') == True:
                player_model.train(X_train, y_train)

            if config.get('model').get('name') in LIST_NN_MODELS:
                if not player_flip.attack_type in ['dummy_attack']:
                    player_model.train_and_evaluate(config, dir_exp_folder, round, player_model.model, train_loader,
                                                    val_loader, test_loader,
                                                    player_model.optimizer, player_model.criterion, lr,
                                                    num_epochs=config.get('training').get('num_epochs'))
            else:
                player_model.train(X_train, y_train)

            if config.get('model').get('name') in LIST_SVM_MODELS:
                # Store model dual variables (Lagrange multipliers history)
                alphas_history.append(player_model.model.alphas)

        else:  # rounds > 0
            if config.get('model').get('name') in LIST_SVM_MODELS_W_PGD_TRAIN:
                # Compute gradient of dual obj wrt alphas (dual variables)
                alphas_gradient = player_model.model.compute_gradient_at_point(X_train, y_adv,
                                                                               player_model.model.alphas)
                # Compute PGD step
                pgd_step = player_model.model.pgd_svm_step(player_model.model.alphas, alphas_gradient, lr)
                # Check if PGD step satisfy constraints or not
                if player_model.model.check_project_constraints(y_adv, pgd_step, config.get('model').get('C')):
                    # Compute projected alphas
                    # For small datasets, we use QP projection
                    if config.get('data').get('dataset') not in LIST_LARGE_DATASETS:
                        projected_alphas = player_model.model.pgd_project(y_adv, pgd_step,
                                                                          config.get('model').get('C'))
                    else:   # For large datasets, we use fixed point iteration based projection
                        projected_alphas = player_model.model.pgd_project_fpi_based(y_adv, pgd_step,
                                                                                    config.get('model').get('C'))
                    # Update SVs and properties
                    if round > warmup_rounds:
                        player_model.model.update_sv(X_train, y_adv, projected_alphas)
                        player_model.model.compute_bias(X_train, projected_alphas)
                else:  # pgd_step is in feasible region
                    # Update SVs and properties
                    if round > warmup_rounds:
                        projected_alphas = copy.copy(pgd_step)
                        player_model.model.update_sv(X_train, y_adv, projected_alphas)
                        player_model.model.compute_bias(X_train, projected_alphas)

                # Store history of dual variables
                alphas_history.append(player_model.model.alphas)


            elif config.get('model').get('name') in LIST_NN_MODELS:
                # Train with adversarial dataset
                player_model.train_and_evaluate(config, dir_exp_folder, round, player_model.model, train_adv_loader,
                                                val_loader, test_loader,
                                                player_model.optimizer, player_model.criterion, lr,
                                                num_epochs=config.get('training').get('num_adv_epochs'))

        # Dump the performance of method after round
        if config.get('data').get('dataset') in LIST_SMALL_DATASETS:
            log_rate = 1
        elif config.get('data').get('dataset') in LIST_LARGE_DATASETS and config.get('game').get('num_rounds') > 1000:
            log_rate = 200
        elif config.get('data').get('dataset') in LIST_LARGE_DATASETS and 500 <= config.get('game').get(
                'num_rounds') <= 1000:
            log_rate = 20
        else:
            log_rate = 1

        if (round > warmup_rounds and round % log_rate == 0) or (round == warmup_rounds + 1):
            dump_round = round - warmup_rounds - 1
            avg_train_loss, avg_train_acc = player_model._evaluate(config, player_model.model, train_loader,
                                                                   player_model.criterion)
            dump_metrics(config, config.get('dump').get('dir_dump'), dir_exp_folder, config, avg_train_loss,
                         avg_train_acc, 'train', dump_round)
            if config.get('data').get('dataset') not in LIST_LARGE_DATASETS:
                avg_val_loss, avg_val_acc = player_model._evaluate(config, player_model.model, val_loader,
                                                                   player_model.criterion)
                dump_metrics(config, config.get('dump').get('dir_dump'), dir_exp_folder, config, avg_val_loss,
                             avg_val_acc, 'validation', dump_round)
            avg_test_loss, avg_test_acc = player_model._evaluate(config, player_model.model, test_loader,
                                                                 player_model.criterion)
            dump_metrics(config, config.get('dump').get('dir_dump'), dir_exp_folder, config, avg_test_loss,
                         avg_test_acc, 'test', dump_round)

        if (config.get('data').get('in_dim') <= 2
                and X_train.shape[1] <= 2
                and (round == warmup_rounds + 1 or round > config.get('game').get('num_rounds') - 2)):
            dump_round = round - warmup_rounds - 1
            player_model._visualize_decision_boundary(config, dir_exp_folder, X_train, y_train, "Train", dump_round)
            if round != 0:
                player_model._visualize_decision_boundary(config, dir_exp_folder, X_adv, y_adv, "Train_adv", dump_round)
            player_model._visualize_decision_boundary(config, dir_exp_folder, X_val, y_val, "Validation", dump_round)
            player_model._visualize_decision_boundary(config, dir_exp_folder, X_test, y_test, "Test", dump_round)

        model_path = os.path.join(config.get('model').get('dir_dump_model'),
                                  f"{config.get('data').get('dataset')}",
                                  f"_{config.get('model').get('name')}",
                                  f"_{config.get('optim').get('optimizer')}optim",
                                  f"_epoch{config.get('training').get('num_epochs')}",
                                  f"_lr{config.get('optim').get('lr')}",
                                  f"_isperturbed_{config.get('training').get('is_perturbed')}.pt")

        ######## PLAYER: ATTACKER ########
        if round <= warmup_rounds:  # Attacker does not attack in warmup period
            X_adv = X_train.clone().detach()
            y_train_adv = y_train.clone().detach()
            y_adv = y_train_adv.clone().detach()
            flip = []
            list_flip_indices = []
        else:
            if round >= (config.get('game').get('num_rounds') - warmup_rounds - 2):
                top_k_indices, top_k_points_x, top_k_points_y = player_flip.get_top_k_points(config, player_model.model,
                                                                                             X_train, y_train)
                dump_general_metric(config, config.get('dump').get('dir_dump'), dir_exp_folder, config,
                                    top_k_indices, f"_top_k_points_round{round}_seed{args.seed}")

            if player_flip.attack_type in ['dummy_attack']:
                X_adv = X_train.clone().detach()
                y_train_adv = y_train.clone().detach()
                y_adv = y_train_adv.clone().detach()
                flip = []
            elif player_flip.attack_type in ['randomized_sv_flip']:
                    y_train_adv, flip, list_flip_indices = player_flip.randomized_sv_flip(config,
                                                                                          player_model.model,
                                                                                          X_train, y_train)
                    player_model.model.set_flip_indices(list_flip_indices)
                    X_adv = X_train.clone().detach()
                    y_adv = y_train_adv.clone().detach().squeeze()

            flip_history.append(flip)
            if (config.get('data').get('in_dim') <= 2
                    and X_train.shape[1] <= 2
                    and (round == warmup_rounds + 1 or round > config.get('game').get('num_rounds') - 2)):
                plot_adv_labels(config, dir_exp_folder, round, X_train, y_train, y_train_adv.unsqueeze(0))

            if config.get('model').get('name') in LIST_NN_MODELS:
                train_adv_dataset = TensorDataset(X_adv, y_adv)
                train_adv_loader = DataLoader(train_adv_dataset, batch_size=config.get('training').get('batch_size'),
                                              shuffle=True)

    dump_general_metric(config, config.get('dump').get('dir_dump'), dir_exp_folder, config,
                        flip_history, "flip_history")

    if player_flip.attack_type in ['dummy_attack', 'randomized_sv_flip']:
        dump_model_path = os.path.join(dir_exp_folder)
        model_path = os.path.join(dump_model_path, f"{config.get('data').get('dataset')}"
                                                   f"_{config.get('model').get('name')}"
                                                   f"_{config.get('optim').get('optimizer')}optim"
                                                   f"_rounds{config.get('game').get('num_rounds')}"
                                                   f"_epoch{config.get('training').get('num_epochs')}"
                                                   f"_lr{config.get('optim').get('lr')}"
                                                   f"_trainsize{1 - config.get('data').get('val_size'):.2f}.pt")
        player_model.dump_model(model_path)
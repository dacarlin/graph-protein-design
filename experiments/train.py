from __future__ import print_function
import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from torch.utils.tensorboard import SummaryWriter



# Library code
sys.path.insert(0, '..')
from struct2seq import *
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Structure to sequence modeling')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden dimensions')
    parser.add_argument('--k_neighbors', type=int, default=16, help='Neighborhood size for k-NN')
    parser.add_argument('--vocab_size', type=int, default=20, help='Alphabet size')
    parser.add_argument('--features', type=str, default='full', help='Protein graph features')
    parser.add_argument('--mpnn', action='store_true', help='Use MPNN updates instead of attention')
    parser.add_argument('--restore', type=str, default='', help='Checkpoint file for restoration')
    parser.add_argument('--name', type=str, default='', help='Experiment name for logging')
    parser.add_argument('--file_data', type=str, default='../data/cath/chain_set.jsonl', help='input chain file')
    parser.add_argument('--file_splits', type=str, default='../data/cath/chain_set_splits.json', help='input chain file')
    parser.add_argument('--batch_tokens', type=int, default=10_000, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
    parser.add_argument('--device', type=str, default="cpu", help='device to use for computation')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing rate')
    args = parser.parse_args()
    return args

def setup_device_rng(args):
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    return device 

def setup_model(hyperparams, device):
    # Build the model
    model = struct2seq.Struct2Seq(
        num_letters=hyperparams['vocab_size'], 
        node_features=hyperparams['hidden'],
        edge_features=hyperparams['hidden'], 
        hidden_dim=hyperparams['hidden'],
        k_neighbors=hyperparams['k_neighbors'],
        protein_features=hyperparams['features'],
        dropout=hyperparams['dropout'],
        use_mpnn=hyperparams['mpnn']
    ).to(device)
    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model

def setup_cli_model():
    args = get_args()
    device = setup_device_rng(args)
    model = setup_model(vars(args), device)
    if args.restore:
        load_checkpoint(args.restore, model)
    return args, device, model

def load_checkpoint(checkpoint_path, model):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dicts['model_state_dict'])
    print('\tEpoch {}'.format(state_dicts['epoch']))
    return

def featurize(batch, device):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    return X, S, mask, lengths

def plot_log_probs(log_probs, total_step, folder=''):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    reorder = 'DEKRHQNSTPGAVILMCFWY'
    permute_ix = np.array([alphabet.index(c) for c in reorder])
    plt.close()
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    P = np.exp(log_probs.cpu().data.numpy())[0].T
    plt.imshow(P[permute_ix])
    plt.clim(0,1)
    plt.colorbar()
    plt.yticks(np.arange(20), [a for a in reorder])
    ax.tick_params(
        axis=u'both', which=u'both',length=0, labelsize=5
    )
    plt.tight_layout()
    plt.savefig(folder + 'probs{}.pdf'.format(total_step))
    return

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed_reweight(S, log_probs, mask, weight=0.1, factor=10.):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    # Upweight the examples with worse performance
    loss = -(S_onehot * log_probs).sum(-1)
    
    # Compute an error-weighted average
    loss_av_per_example = torch.sum(loss * mask, -1, keepdim=True) / torch.sum(mask, -1, keepdim=True)
    reweights = torch.nn.functional.softmax(factor * loss_av_per_example, 0)
    mask_reweight = mask * reweights
    loss_av = torch.sum(loss * mask_reweight) / torch.sum(mask_reweight)
    return loss, loss_av
  

args, device, model = setup_cli_model()
optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)
criterion = torch.nn.NLLLoss(reduction='none')

# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=1_000, max_length=256)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
train_set, validation_set, test_set = [
    Subset(dataset, [
        dataset_indices[chain_name] for chain_name in dataset_splits[key] 
        if chain_name in dataset_indices
    ])
    for key in ['train', 'validation', 'test']
]
loader_train, loader_validation, loader_test = [data.StructureLoader(
    d, batch_size=args.batch_tokens
) for d in [train_set, validation_set, test_set]]
print('Training:{}, Validation:{}, Test:{}'.format(len(train_set),len(validation_set),len(test_set)))

# Build basepath for experiment
if args.name != '':
    base_folder = 'log/' + args.name + '/'
else:
    base_folder = time.strftime('log/%y%b%d_%I%M%p/', time.localtime())
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
subfolders = ['checkpoints', 'plots']
for subfolder in subfolders:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)

with open(base_folder + 'args.json', 'w') as f:
    json.dump(vars(args), f)

writer = SummaryWriter(log_dir=base_folder) 

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []
total_step = 0
for e in range(args.epochs):
    # Training epoch
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(loader_train):

        start_batch = time.time()
        # Get a batch
        X, S, mask, lengths = featurize(batch, device)
        elapsed_featurize = time.time() - start_batch

        optimizer.zero_grad()
        log_probs = model(X, S, lengths, mask)
        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask, weight=args.smoothing)
        loss_av_smoothed.backward()
        optimizer.step()

        loss, loss_av = loss_nll(S, log_probs, mask)

        # Timing
        elapsed_batch = time.time() - start_batch
        elapsed_train = time.time() - start_train
        total_step += 1
        writer.add_scalar("perplexity/train", np.exp(loss_av.cpu().data.numpy()), total_step)
        writer.add_scalar("perplexity/train-smoothed", np.exp(loss_av_smoothed.cpu().data.numpy()), total_step)
        # print(total_step, elapsed_train, np.exp(loss_av.cpu().data.numpy()), np.exp(loss_av_smoothed.cpu().data.numpy()))

        # Accumulate true loss
        train_sum += torch.sum(loss * mask).cpu().data.numpy()
        train_weights += torch.sum(mask).cpu().data.numpy()

        # # DEBUG UTILIZATION Stats
        # if args.cuda:
        #     utilize_mask = 100. * mask.sum().cpu().data.numpy() / float(mask.numel())
        #     utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
        #     tps_train = mask.cpu().data.numpy().sum() / elapsed_batch
        #     tps_features = mask.cpu().data.numpy().sum() / elapsed_featurize
        #     print('Tokens/s (train): {:.2f}, Tokens/s (features): {:.2f}, Mask efficiency: {:.2f}, GPU max allocated: {:.2f}'.format(tps_train, tps_features, utilize_mask, utilize_gpu))

        # if total_step % 5000 == 0:
        #     torch.save({
        #         'epoch': e,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.optimizer.state_dict()
        #     }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))

    # Train image
    P = np.exp(log_probs.cpu().data.numpy())[0].T
    writer.add_image(batch[0]["name"], P, total_step)
    #plot_log_probs(log_probs, total_step, folder='{}plots/train_{}_'.format(base_folder, batch[0]['name']))

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            X, S, mask, lengths = featurize(batch, device, shuffle_fraction=args.shuffle)
            log_probs = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, mask)

            # Accumulate
            validation_sum += torch.sum(loss * mask).cpu().data.numpy()
            validation_weights += torch.sum(mask).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    print('Perplexity\tTrain:{}\t\tValidation:{}'.format(train_perplexity, validation_perplexity))

    # Validation image
    plot_log_probs(log_probs, total_step, folder='{}plots/valid_{}_'.format(base_folder, batch[0]['name']))

    # with open(logfile, 'a') as f:
    #     f.write('{}\t{}\t{}\n'.format(e, train_perplexity, validation_perplexity))

    writer.add_scalar("epoch/train/perplexity", train_perplexity)
    writer.add_scalar("epoch/validation/perplexity", validation_perplexity)

    # Save the model
    checkpoint_filename = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step)
    torch.save({
        'epoch': e,
        'hyperparams': vars(args),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict()
    }, checkpoint_filename)

    epoch_losses_valid.append(validation_perplexity)
    epoch_losses_train.append(train_perplexity)
    epoch_checkpoints.append(checkpoint_filename)

# Determine best model via early stopping on validation
best_model_idx = np.argmin(epoch_losses_valid).item()
best_checkpoint = epoch_checkpoints[best_model_idx]
train_perplexity = epoch_losses_train[best_model_idx]
validation_perplexity = epoch_losses_valid[best_model_idx]
best_checkpoint_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_model_idx + 1)
shutil.copy(best_checkpoint, best_checkpoint_copy)
load_checkpoint(best_checkpoint_copy, model)


# Test epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for _, batch in enumerate(loader_test):
        X, S, mask, lengths = featurize(batch, device)
        log_probs = model(X, S, lengths, mask)
        loss, loss_av = loss_nll(S, log_probs, mask)
        # Accumulate
        test_sum += torch.sum(loss * mask).cpu().data.numpy()
        test_weights += torch.sum(mask).cpu().data.numpy()

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity\tTest:{}'.format(test_perplexity))

with open(base_folder + 'results.txt', 'w') as f:
    f.write('Best epoch: {}\nPerplexities:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}'.format(
        best_model_idx+1, train_perplexity, validation_perplexity, test_perplexity
    ))


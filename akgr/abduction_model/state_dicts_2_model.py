import os, sys, argparse
sys.path.append('..')
import torch
# transformer (huggingface)
from transformer import TransformerModel
# dataloader
from akgr.dataloader import create_dataloader

from utils.load_util import load_model, save_model, load_yaml

def state_dicts_2_model(load_path, epoch, model, optimizer, scheduler):
    model, optimizer, scheduler, epoch, loss_log = \
        load_model(load_path, 'state_dicts', epoch, model, optimizer, scheduler)
    save_path = load_path.removesuffix('.tar') + '.pth'
    save_model(save_path, 'model', model, optimizer, scheduler, epoch, loss_log)

def my_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-dataloader', default='akgr/configs/config-dataloader.yml')
    parser.add_argument('--config-hyperpara', default='config-hyperpara.yml')
    parser.add_argument('--config-batchsize', default='akgr/configs/config-batchsize.yml')

    parser.add_argument('--data_root', default='../sampling/')
    parser.add_argument('-d', '--dataname', default='FB15k-237')
    parser.add_argument('--scale', default='debug')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)

    parser.add_argument('--checkpoint_root', default='./')
    parser.add_argument('-r', '--resume_epoch', type=int, default=0)
    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')

    parser.add_argument('--result_root', default='./')

    parser.add_argument('--save_frequency', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = my_parse_args()

    config_dataloader = load_yaml(args.config_dataloader)
    config_hyperpara = load_yaml(args.config_hyperpara)
    config_batchsize = load_yaml(args.config_batchsize)
    print(f'Converting state_dicts to model {args.dataname} {args.scale} {args.max_answer_size} {args.resume_epoch}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    dataloader, ntoken, qry_tokenizer, ans_tokenizer, pattern_filtered = \
        create_dataloader(args.dataname, args.scale, args.max_answer_size,
                        config_dataloader, config_batchsize[args.dataname],
                        args.data_root)

    model = TransformerModel(device=device, ntoken=ntoken,
        nfeature=config_hyperpara['nfeature'],
        special_token=config_dataloader['special_tokens'],
        nhead=config_hyperpara['nhead']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config_hyperpara["lr"]))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                total_iters=config_hyperpara["warm_up"])

    load_path = os.path.join(args.checkpoint_root, \
        f'{args.dataname}-{args.scale}-{args.max_answer_size}-{args.resume_epoch}.tar')
    state_dicts_2_model(load_path, args.resume_epoch, model, optimizer, scheduler)

if __name__ == '__main__':
    main()
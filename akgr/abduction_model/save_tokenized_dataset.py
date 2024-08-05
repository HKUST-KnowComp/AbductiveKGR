args = my_parse_args()
print(f'# Running main.py in {args.mode} mode with:')
print(f'args:\n{args}\n')

# Data representation
global config_dataloader
config_dataloader = load_yaml(args.config_dataloader)
global offset, special_tokens
offset = config_dataloader['offset']
special_tokens = config_dataloader['special_tokens']
print(f'config_dataloader:\n{config_dataloader}\n')

global pattern_filtered
pattern_filtered_path = 'akgr/metadata/pattern_filtered.csv'
pattern_filtered = pd.read_csv(pattern_filtered_path, index_col='id')

# Model information
model_name = args.modelname
is_gpt=('GPT2' in model_name)
is_act=('act' in model_name)
tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
src_len = config_dataloader['ans_len'] + 1
print(f'model_name:{model_name}\n')

# Batch size
config_batchsize = load_yaml(args.config_batchsize)
batch_size = config_batchsize[model_name][args.dataname]
if args.overwrite_batchsize != 0:
    batch_size = args.overwrite_batchsize
print(f'batch_size:{batch_size}\n')

print('=' * 50)

# Device
global device
if args.accelerate and args.mode != 'optimizing':
    accelerator = Accelerator()
    device = accelerator.device
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# Dataset
if args.mode == 'training':  splits = ['train', 'valid']
elif args.mode == 'testing': splits = [args.test_split]
elif args.mode == 'optimizing': splits = ['train']
elif args.mode == 'load-save-test': splits = ['train', 'test']

print('Creating dataset & dataloader')
global nentity, nrelation
dataset_dict, nentity, nrelation = new_create_dataset(
    dataname=args.dataname,
    scale=args.scale,
    answer_size=args.max_answer_size,
    pattern_filtered=pattern_filtered,
    data_root=args.data_root,
    splits=splits,
    is_gpt=is_gpt,
    is_act=is_act
)
if args.mode == 'testing' and args.test_proportion < 1:
    nrows = dataset_dict[args.test_split].shape[0]
    dataset_dict[args.test_split] = dataset_dict[args.test_split].select(random.sample(range(nrows), int(nrows * args.test_proportion)))
if args.mode == 'optimizing' and args.ppo_proportion < 1:
    nrows = dataset_dict['train'].shape[0]
    dataset_dict['train'] = dataset_dict['train'].select(random.sample(range(nrows), int(nrows * args.ppo_proportion)))
dataloader_dict = new_create_dataloader(
    dataset_dict=dataset_dict,
    batch_size=batch_size,
    drop_last=(args.mode == 'optimizing') #or (args.mode == 'testing' and args.accelerate)
)

# Tokenizer
print('Creating tokenizer')
tokenizer, ntoken = create_tokenizer(
    special_tokens=special_tokens,
    offset=offset,
    nentity=nentity,
    nrelation=nrelation,
    is_gpt=is_gpt
)
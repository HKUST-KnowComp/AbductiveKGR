from akgr.utils.load_util import jsonl_2_pickle, load_sampled_dataset

data_root = './sampled_data/'
dataname = 'FB15k-237'
scale = 'debug'
answer_size = 32
jsonl_2_pickle(data_root=data_root, dataname=dataname, scale=scale, answer_size=answer_size)
print('start')
jsonl_data = load_sampled_dataset(data_root=data_root, dataname=dataname, scale=scale, answer_size=answer_size, method='jsonl')
print('loaded')
pkl_data = load_sampled_dataset(data_root=data_root, dataname=dataname, scale=scale, answer_size=answer_size, method='pkl')
print('loaded')
print(jsonl_data == pkl_data)
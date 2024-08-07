CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --modelname='T5_disablepos_3_act_nt'\
    --data_root='./sampled_data/' -d='FB15k-237' --scale='full' -a=32  \
    --checkpoint_root='checkpoints/' -r=140\
    --result_root='./results/'\
    --save_frequency 5\
    --test_proportion=1\
    --test_split='test'\
    --overwrite_batchsize=1024\
    --mode='testing'\
    --test_top_k=0\
    --test_count0\
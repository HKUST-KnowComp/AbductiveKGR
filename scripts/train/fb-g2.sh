CUDA_VISIBLE_DEVICES=3 python -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d='FB15k-237' --scale='full' -a=32  \
    --checkpoint_root='checkpoints/'\
    --result_root='./results/'\
    --save_frequency 1\
    --mode='training'
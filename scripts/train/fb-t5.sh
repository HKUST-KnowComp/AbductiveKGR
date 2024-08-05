CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --modelname='T5_disablepos_3_act_nt'\
    --data_root='./sampled_data/' -d='FB15k-237' --scale='full' -a=32  \
    --checkpoint_root='checkpoints/'\
    --result_root='./results/'\
    --save_frequency 1\
    --mode='training'
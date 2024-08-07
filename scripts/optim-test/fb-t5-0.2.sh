CUDA_VISIBLE_DEVICES=1 python -m akgr.abduction_model.main \
    --modelname='T5_disablepos_3_act_nt'\
    --data_root='./sampled_data/' -d='FB15k-237' --scale='full' -a=32  \
    --checkpoint_root='checkpoints/' -r=140\
    --result_root='./results/'\
    --save_frequency 50\
    --mode='testing'\
    --ppo_resume_epoch=150\
    --test_count0\
    --test_top_k=0\
    --overwrite_batchsize=2048\
    --ppo_lr=2.4e-5\
    --ppo_smatch_factor=0.2\
    --ppo_init_kl_coef=0.2\
    --ppo_cliprange=0.2\
    --ppo_proportion=1\
    --ppo_minibatch=512\
    --ppo_horizon=4096
# default experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-dir',
    embedding_std_type='trainable',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
    )
)

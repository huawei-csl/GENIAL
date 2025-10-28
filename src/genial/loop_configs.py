all_configs = {
    "debug": {
        "experiment_name": "multiplier_2bi_4bo_permuti_allcells_notech_fullsweep_only",
        "synth_version": 0,
        "nb_to_generate": 10,
        "device": 5,  # Trainer & Recommender
        "score_type": "trans",  # Trainer
        "score_rescale_mode": "minmax",  # Trainer
        "max_epochs": 2,  # Trainer
        "check_val_every_n_epoch": 1,  # Trainer
        "nb_new_designs": 2,  # Recommender
        "keep_percentage": 1.0,  # Recommender
    }
}

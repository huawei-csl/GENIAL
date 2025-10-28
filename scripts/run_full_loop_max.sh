
# Set Input Variables
DEVICE_ID=1
CONFIG_FILE="model_config_sample_w_cls_n_vae_ft.yml"
TASK="finetune_from_ssl"
SSL_CHECKPOINT="114_0.0491_000.ckpt"

NB_WORKERS=60
OUT_DIR="200k_loop_10_rounds"
MAX_EPOCHS=2000
NB_INIT_DESIGNS=20000
NB_PROTOTYPES=10000

echo "||===================== Running task launcher ===================== ||"
repeat_nb_init=10
while [ $repeat_nb_init -gt 0 ]
do
task_launcher\
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
    --output_dir_name $OUT_DIR\
    --skip_synth\
    --skip_swact\
    --nb_workers $NB_WORKERS\
    --nb_new_designs $NB_INIT_DESIGNS\
    
    repeat_nb_init=$((repeat_nb_init-1))
done

echo "||===================== Running task analyzer ===================== ||"
task_analyzer\
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
    --output_dir_name $OUT_DIR\
    --nb_workers $NB_WORKERS\
    --skip_swact\
    --bulk_flow_dirname power_out\
    --rebuild_db\
    --technology asap7\

mkdir -p $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/$OUT_DIR/trainer_out/lightning_logs/version_0/checkpoints
cp $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/$SSL_CHECKPOINT $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/$OUT_DIR/trainer_out/lightning_logs/version_0/checkpoints/.
# Train model
echo "||===================== Running trainer ===================== ||"
python $SRC_DIR/src/genial/training/mains/trainer_enc_to_score_value.py\
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
    --output_dir_name $OUT_DIR\
    --yml_config_path $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE\
    --bulk_flow_dirname power_out\
    --max_epochs $MAX_EPOCHS\
    --batch_size 4000\
    --nb_workers $NB_WORKERS\
    --device $DEVICE_ID\
    --check_val_every_n_epoch 1\
    --trainer_task $TASK\
    --model_checkpoint $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/$OUT_DIR/trainer_out/lightning_logs/version_0/checkpoints/$SSL_CHECKPOINT\
    --score_type "power"\

echo "||===================== Running prototype generation ===================== ||"
python src/genial/utils/prototype_generator_v2.py\
    --do_prototype_pattern_gen\
    --skip_swact\
    --skip_synth\
    --nb_workers $NB_WORKERS\
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
    --output_dir_name $OUT_DIR\
    --dst_output_dir_name $OUT_DIR\
    --trainer_version_number 1\
    --nb_gener_prototypes $NB_PROTOTYPES\
    --max_epochs $MAX_EPOCHS\
    --batch_size 4000\
    --device $DEVICE_ID\
    --bulk_flow_dirname power_out\
    --yml_config_path  $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE\
    --score_type "power"\

repeat_nb=20
while [ $repeat_nb -gt 0 ]
do
    echo "Remaining Iterations: $repeat_nb"
    repeat_nb=$((repeat_nb-1))

    echo "||===================== Running launcher again ===================== ||"
    task_launcher\
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
        --nb_workers $NB_WORKERS\
        --output_dir_name $OUT_DIR\
        --skip_synth\
        --skip_swact\

    echo "||===================== Running analyzer again ===================== ||"
    task_analyzer\
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
        --output_dir_name $OUT_DIR\
        --bulk_flow_dirname power_out\
        --skip_swact\
        --technology asap7\
        --nb_workers $NB_WORKERS\
        --rebuild_db\

    echo "||===================== Running trainer ===================== ||"
    rm -rf $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/$OUT_DIR/trainer_out/lightning_logs/version_1
    python $SRC_DIR/src/genial/training/mains/trainer_enc_to_score_value.py\
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
        --output_dir_name $OUT_DIR\
        --yml_config_path $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE\
        --bulk_flow_dirname power_out\
        --max_epochs $MAX_EPOCHS\
        --batch_size 4000\
        --device $DEVICE_ID\
        --check_val_every_n_epoch 1\
        --trainer_task $TASK\
        --model_checkpoint $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/$OUT_DIR/trainer_out/lightning_logs/version_0/checkpoints/$SSL_CHECKPOINT\
        --nb_workers $NB_WORKERS\
        --score_type "power"\

    echo "||===================== Running prototype generation ===================== ||"
    python src/genial/utils/prototype_generator_v2.py\
        --do_prototype_pattern_gen\
        --skip_swact\
        --skip_synth\
        --nb_workers $NB_WORKERS\
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only\
        --output_dir_name $OUT_DIR\
        --dst_output_dir_name $OUT_DIR\
        --trainer_version_number 1\
        --nb_gener_prototypes $NB_PROTOTYPES\
        --max_epochs $MAX_EPOCHS\
        --batch_size 4000\
        --device $DEVICE_ID\
        --bulk_flow_dirname power_out\
        --yml_config_path  $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE\
        --score_type "power"\

done

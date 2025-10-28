#!/bin/bash

# Load the .env
source .env

# Activate the conda environment
source activate design_gen

cd $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech

FOLDER_PREFIX="15k_random_sampling_wo_ssl"

for i in {1..10}; do
  tar -czvf ${FOLDER_PREFIX}_merge_${i}.tar.gz ${FOLDER_PREFIX}_merge_${i}
  rm -r ${FOLDER_PREFIX}_merge_${i}
done

for i in {1..10}; do
  tar -czvf ${FOLDER_PREFIX}_prototype_gen_${i}.tar.gz ${FOLDER_PREFIX}_prototype_gen_${i}
  rm -r ${FOLDER_PREFIX}_prototype_gen_${i}
done

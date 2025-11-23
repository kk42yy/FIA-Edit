conda activate FIA-Edit
cd .../FIA-Edit
export CUDA_VISIBLE_DEVICES=0
# export HF_ENDPOINT=https://hf-mirror.com

## PIE-Bench
inferp=outputs/FIA-Edit_SD35
python evaluation/evaluate.py \
    --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
    --result_path $inferp/evaluation_result.csv \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --tgt_methods FIA-Edit \
    --src_image_folder data/annotation_images \
    --annotation_mapping_file data/mapping_file.json \
    --infer_dir $inferp/annotation_images
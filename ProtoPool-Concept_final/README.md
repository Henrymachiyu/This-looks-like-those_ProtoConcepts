# ProtoPool-Concept 

The code is based on the other repositories: https://github.com/cfchen-duke/ProtoPNet, and https://github.com/gmum/ProtoPool

## Train a model:
To reproduce our result:

For CUB-200-2011: python main_cap_final.py --num_classes 200 --batch_size 80 --num_descriptive 10 --num_prototypes 202 --use_scheduler --arch resnet50 --pretrained --proto_depth 256 --warmup_time 10 --warmup --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --inat --data_train '$train_path' --data_push '$push_path' --data_test '$test_path' --gpuid 0 --epoch 30 --lr 0.5e-3 --earlyStopping 12 --cap_start 0 --capl 4.5 --capcoef 3e-3 --only_warmuptr True --topk_loss True --k_top 10 --sep True

For Stanford Car: python main_cap_final.py --num_classes 196 --batch_size 80 --num_descriptive 10 --num_prototypes 195 ---use_scheduler --arch resnet50 --pretrained --proto_depth 256 --warmup_time 10 --warmup --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --inat --data_train '$train_path' --data_push '$push_path' --data_test '$test_path' --gpuid 0 --epoch 30 --lr 0.5e-3 --earlyStopping 12 --cap_start 0 --capl 4.5 --capcoef 3e-3 --only_warmuptr True --topk_loss True --k_top 10 --sep True


## Local analysis
To run local analysis

python local_analysis_final.py --data_train '$train_path' --data_push '$push_path' --data_test '$test_path' --model_dir "$model_path" --batch_size 80 --num_descriptive 10 --num_prototypes 202 --inat --num_classes 200 --arch resnet50 --gpuid 0 --pretrained --proto_depth 256 --prototype_activation_function log --last_layer --use_thresh --capl 4.5 --save_analysis_dir '$dir to save analysis' --target_img_dir '$test_path' --prototype_img_dir '$prototype_vis_dir'


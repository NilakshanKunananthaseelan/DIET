#TO Do
# - Ablation on norm_r
# - Ablation on curvature
# - Ablation on LoRA config
# - Ablation on cost funtion weights
# - Ablation on LoRA layer


for DATASET in OxfordFlowers OxfordPets Caltech101 StanfordCars
do
    DATASET_LOWER=$(echo $DATASET | tr '[:upper:]' '[:lower:]')
    echo $DATASET_LOWER

    for shots in 16; do
    for seed in 1 2 3 4 5; do
        for lambda_ot in 1; do
        
            for lr in  0.0009; do
                
                    for r in 4; do
                        alpha=1
                        
                    SAVE_PATH=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_hyp_lora_ot_ret/${DATASET_LOWER}/vision/${shots}shots/seed${seed}/

                    CUDA_VISIBLE_DEVICES=0 python main_lora_hypebuseman.py \
                        --dataset $DATASET \
                        --backbone ViT-B/16 \
                        --batch_size 32 \
                        --unlearn_lr $lr \
                        --norm_r 1 \
                        --lambda_hyp 30 \
                        --lambda_ot $lambda_ot \
                        --lambda_retain 0 \
                        --prototype_type eucl \
                        --cost_type busemann \
                        --seed $seed \
                        --save_path $SAVE_PATH \
                        --filename lora_weights \
                        --unlearn_epochs 30 \
                        --ot_type sinkhorn \
                        --position all \
                        --encoder vision \
                        --r 4 \
                        --alpha 1\
                        --shots $shots
                        
                    
                done
            done
        done
    done
    done
done


# for DATASET in cifar10 cifar100 svhn
# do
#     DATASET_LOWER=$(echo $DATASET | tr '[:upper:]' '[:lower:]')
#     echo $DATASET_LOWER

  
#     for seed in 1 2 3 4 5 ; do
#         for lambda_ot in 1; do
        
#             for lr in  0.0009; do
                
#                     for r in 4; do
#                         alpha=1
                        
#                     SAVE_PATH=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_hyp_lora_simple/${DATASET_LOWER}/vision/${seed}/

#                     CUDA_VISIBLE_DEVICES=1 python main_lora_hypebuseman.py \
#                         --dataset $DATASET \
#                         --backbone ViT-B/16 \
#                         --batch_size 32 \
#                         --unlearn_lr $lr \
#                         --norm_r 1 \
#                         --lambda_hyp 25 \
#                         --lambda_ot $lambda_ot \
#                         --lambda_retain 20 \
#                         --prototype_type eucl \
#                         --cost_type busemann \
#                         --seed $seed \
#                         --save_path $SAVE_PATH \
#                         --filename lora_weights \
#                         --unlearn_epochs 30 \
#                         --ot_type sinkhorn \
#                         --position all \
#                         --encoder vision \
#                         --r 4 \
#                         --alpha 1\
                        
#                     done
                
#             done
#         done
#     done
# done


# # for DATASET in  OxfordFlowers; do
# # DATASET_LOWER=$(echo $DATASET | tr '[:upper:]' '[:lower:]')
# # echo $DATASET_LOWER
# # for encoder in  vision
# # do
# #         for seed in 1 2 3 4 5
# #         do   

# #                 SAVE_PATH=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_new/${DATASET_LOWER}/${encoder}/${seed}
# #                 CUDA_VISIBLE_DEVICES=1 python main_lora_hypebuseman.py \
# #                                                 --dataset  $DATASET \
# #                                                 --backbone ViT-B/16 \
# #                                                 --batch_size 32 \
# #                                                 --lr 7e-4 \
# #                                                 --lambda_hyp 1 \
# #                                                 --lambda_ot 0.1 \
# #                                                 --lambda_retain 10 \
# #                                                 --prototype_type eucl \
# #                                                 --cost_type busemann \
# #                                                 --seed $seed \
# #                                                 --save_path $SAVE_PATH \
# #                                                 --filename lora_weights \
# #                                                 --unlearn_epochs 10  \
# #                                                 --ot_type sinkhorn \
# #                                                 --alpha 1\
# #                                                 --position all\
# #                                                 --encoder $encoder\
# #                                                 --r 4\
                                        
                
# # done
# # done
# # done   


# # CUDA_VISIBLE_DEVICES=1 python main_lora_hypebuseman.py \
# #                                         --dataset  $DATASET \
# #                                         --backbone ViT-B/16 \
# #                                         --batch_size 32 \
# #                                         --lr 2e-4 \
# #                                         --lambda_hyp 0.05 \
# #                                         --prototype_type eucl \
# #                                         --cost_type busemann \
# #                                         --seed $seed \
# #                                         --save_path $SAVE_PATH \
# #                                         --filename lora_weights \
# #                                         --unlearn_epochs 50  \
# #                                         --ot_type sinkhorn \
# #                                         --lambda_retain 80.0 \
# #                                         --position all\
# #                                         --encoder $encoder\
# #                                         --r 4\
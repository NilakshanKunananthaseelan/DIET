


for DATASET in   OxfordFlowers; do #cifar100 cifar100 Caltech101 OxfordFlowers StanfordCars Food101 EuroSAT SUN397 UCF101 svhn ; do
DATASET_LOWER=$(echo $DATASET | tr '[:upper:]' '[:lower:]')
echo $DATASET_LOWER
for encoder in  vision
do
        for seed in  1 2 3 4 5
        do   

                SAVE_PATH=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_hyp_lora_new/${DATASET_LOWER}/vision/${seed}
                r=4
                alpha=1
                # SAVE_PATH=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_gslora_no_retain_results/${DATASET_LOWER}/seed_${seed}
                # # SAVE_PATH=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_gslora_retain_results/${DATASET_LOWER}/seed_${seed}

                # r=2
                
                # alpha=4
                SAVE_DIR=/home/nilakshan/0-Unlearning/0-HypBuseman/VLM/checkpoints_x_gs_lora/${DATASET_LOWER}/vision/${seed}
                CUDA_VISIBLE_DEVICES=1 python main_cross_dataset.py \
                                        --dataset  $DATASET \
                                        --backbone ViT-B/16 \
                                        --batch_size 32 \
                                        --lr 2e-4 \
                                        --seed $seed \
                                        --save_path $SAVE_PATH \
                                        --save_dir $SAVE_DIR \
                                        --filename lora_weights \
                                        --unlearn_epochs 20  \
                                        --position all\
                                        --encoder $encoder\
                                        --r $r\
                                        --alpha $alpha\
                                        --eval_only
                                        
                                        # --backbone ViT-L/14\
                                        # --clip_path  /home/nilakshan/0-Unlearning/clip_models/ViT-L-14.pt\
                                        # --model_path /home/nilakshan/0-Unlearning/clip_moe/checkpoints_l_14\

done
done
done   



# CUDA_VISIBLE_DEVICES=1 python main_lora_hypebuseman.py \
#                                         --dataset  $DATASET \
#                                         --backbone ViT-B/16 \
#                                         --batch_size 32 \
#                                         --lr 2e-4 \
#                                         --lambda_hyp 0.05 \
#                                         --prototype_type eucl \
#                                         --cost_type busemann \
#                                         --seed $seed \
#                                         --save_path $SAVE_PATH \
#                                         --filename lora_weights \
#                                         --unlearn_epochs 50  \
#                                         --ot_type sinkhorn \
#                                         --lambda_retain 80.0 \
#                                         --position all\
#                                         --encoder $encoder\
#                                         --r 4\
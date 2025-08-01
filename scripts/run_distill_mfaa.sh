model_teachers=( wrn_40_2 wrn_40_2 resnet56 resnet110 resnet32x4 vgg13 vgg13       ResNet50    ResNet50 resnet32x4 resnet32x4 wrn_40_2 )
model_students=( wrn_16_2 wrn_40_1 resnet20 resnet32  resnet8x4  vgg8  MobileNetV2 MobileNetV2 vgg8     ShuffleV1 ShuffleV2   ShuffleV1)

run_exp(){
model_teacher=$1
model_student=$2
kd_methods=(MFAA)


for(( flag=0;flag<1;flag++)) do
    for(( i=0;i<${#kd_methods[@]};i++)) do
        jobname=$(pwd | awk -F "/" '{print $NF}')
        gpu_num=1
        kd_method=${kd_methods[i]}
        echo ${kd_method} ${model_teacher} ${model_student};
        CUDA_VISIBLE_DEVICES=0 python train_student_mfaa.py --dataset cifar100 \
        --path_t ./save_t/${model_teacher}_cifar100_lr_0.05_decay_0.0005_trial_0/ckpt_epoch_240.pth  \
        --distill ${kd_method} --model_s ${model_student}
        sleep 3s
    done;
done;
}

for(( j=0;j<${#model_teachers[@]};j++)) do
    echo ${model_teachers[j]} ${model_students[j]} $j
    run_exp ${model_teachers[j]} ${model_students[j]}
done;
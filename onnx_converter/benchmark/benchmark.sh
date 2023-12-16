#!/usr/bin/env bash

# source activate tf1

task=$1
model_dir=$2
dataset_dir=$3
selected_mode=$4

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

case ${task} in
        "TASK1")
                tasks=('test_anti_spoofing')
                ;;
        "TASK2")
                tasks=('test_face_recognition')
                ;;
        "TASK3")
                tasks=('test_imagenet')
                ;;                                
        "TASK4")
                tasks=('test_object_detection')
                ;;
        "TASK5")
                tasks=('test_pedestrian_detection')
                ;;
        "TASK6")
                tasks=('test_retina_face_detection')
                ;;
        "TASKS")
                tasks=(
                'test_anti_spoofing' \
                'test_face_recognition' \
                'test_pedestrian_detection' \
                'test_retina_face_detection'
                )
                ;;                
        *)
                echo "${task} is not exist!!!"
                exit
                ;;
esac

rm -rf work_dir
mkdir -p work_dir/benchmark/report

for task in ${tasks[@]};
do
    rm -rf work_dir/benchmark/report/${task}.html
    rm -rf work_dir/benchmark/report/${task}_accuracy.html
    pytest --html=work_dir/benchmark/report/${task}.html \
    benchmark/tests/${task}.py \
    --model_dir=${model_dir} \
    --dataset_dir=${dataset_dir} \
    --selected_mode=${selected_mode}
done

# python benchmark/scripts/mmd2pdf.py

# bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_RELEASE'
# bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_MASTER'
# bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_DEV'
# bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_OTHER'

# bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_RELEASE'
# bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_MASTER'
# bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_DEV'
# bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_OTHER'
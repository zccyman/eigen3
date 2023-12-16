# benchmark

## TASK1: test_anti_spoofing
```
pytest --html=work_dir/report/test_anti_spoofing.html \
benchmark/tests/test_anti_spoofing.py \
--model_dir='/home/henson/dataset/trained_models' \
--dataset_dir='/buffer' \
--is_release='False'
```

## TASK2: test_face_recognition
```
pytest --html=work_dir/report/test_face_recognition.html \
benchmark/tests/test_face_recognition.py \
--model_dir='/home/henson/dataset/trained_models' \
--dataset_dir='/buffer' \
--is_release='False'
```

## TASK3: test_imagenet
```
pytest --html=work_dir/report/test_imagenet.html \
benchmark/tests/test_imagenet.py \
--model_dir='/home/henson/dataset/trained_models' \
--dataset_dir='/buffer' \
--is_release='False'
```

## TASK4: test_object_detection
```
pytest --html=work_dir/report/test_object_detection.html \
benchmark/tests/test_object_detection.py \
--model_dir='/home/henson/dataset/trained_models' \
--dataset_dir='/buffer' \
--is_release='False'
```

## TASK5: test_pedestrian_detection
```
pytest --html=work_dir/report/test_pedestrian_detection.html \
benchmark/tests/test_pedestrian_detection.py \
--model_dir='/home/henson/dataset/trained_models' \
--dataset_dir='/buffer' \
--is_release='False'
```

## TASK6: test_retina_face_detection
```
pytest --html=work_dir/report/test_retina_face_detection.html \
benchmark/tests/test_retina_face_detection.py \
--model_dir='/home/henson/dataset/trained_models' \
--dataset_dir='/buffer' \
--is_release='False'
```

## test benchmark using shell
```
bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_RELEASE'
bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_MASTER'
bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_DEV'
bash benchmark/benchmark.sh 'TASK3' '/home/henson/dataset/trained_models' '/buffer' 'MR_OTHER'

bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_RELEASE'
bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_MASTER'
bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_DEV'
bash benchmark/benchmark.sh 'TASKS' '/home/henson/dataset/trained_models' '/buffer' 'MR_OTHER'
```

## test export visualization
```
https://github.com/mermaid-js/mermaid-cli/issues/113
docker pull minlag/mermaid-cli:8.8.0
npm install -g @mermaid-js/mermaid-cli --force

python benchmark/scripts/mmd2pdf.py --export_dir /home/henson/code/onnx-converter-pytest/work_dir
```

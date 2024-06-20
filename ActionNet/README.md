# ActionNet_Tensorflow
This is Tensorflow implemented ActionNet which is used to recognize action from programming screencast. 

## Installation
This software has been tested on Ubuntu16.04(x64) using python3.5, opencv3.3, and tensorflow1.9 with cuda 9.0, cudnn9.0 and GTX-1080Ti GPU.
Some other requirements:
  - skimage
  - imutils
  - tqdm

## Test model
Firstly, you should modify some parameters in 'analyze.py'. 
  - video_name: path to the video you want to test
  - path: path to the dataset
  - label_name: label.txt
  - ck_path: path to Tensorflow check point
  - num: class number before merge
  - nums: class number after merge
  - parts: T0 is the first frame and T1 is the second frame
  - label_map: how to merge label
Then run
```
Python3 analyze.py
```
The result will be stored in '/root/path/result'
Here are some results:
![image1](https://github.com/dehaisea/ActionNet/blob/master/data/result/00644.jpg)
![image2](https://github.com/dehaisea/ActionNet/blob/master/data/result/00781.jpg)

## Train model
1. Extract different region from video 
```
python3 clip.py
```
You can also use python multiprocessing version
```
python3 clip_multiprocessing.py
```
The different region data will be stored in `output_path`
2. Merge all dataset
Each video's different region data is stored in a floder. We need to merge all of them by
```
python3 merge_dataset.py
```
You can assign which folders will be merged by modifying `dir_list`.
The merged result will be stored in 'output_path'.
3. Convert data to tf_record
Modify the dataset path, this path would be the same with `output_path` in step 2.
```
python3 convert_data_to_tfrecord.py
```
tf_record would be stored in the current path
4. launch training
Modify the dataset path in 'finetune.sh'
Then run
```
./finetune.sh
```

## Evaluate model
Modify the parameters in 'evaluation.py', this step is the same with **Test model**
Then run
```
python3 evaluation.py
```
This step will show confusion matrix and some other measurment of this model.




### Hi there 👋

<!--
**B21-CAP007/B21-CAP007** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
## MACHINE LEARNING

### 1. Setting up the environment
We trained our models on tf-nightly-gpu but it should be okay if it's trained on official releases of tensorflow
First we should install tensorflow model for our object detection using git clone :

```git clone https://github.com/tensorflow/models.git```

We also need to install protobuf, use this link bellow to download it.
>https://github.com/protocolbuffers/protobuf/releases

After you've done downloading you should move the "protoc" named file in the zip and move it to models/research. The next thing is you now should be able to install the API using these steps bellow.

*** Python Package installation ***
```
cd models/research
Compile protos.
protoc object_detection/protos/*.proto --python_out=.
Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
if you're using protobuf version 3.5 or higher you should use this command

```python use_protobuf.py <path to directory> <path to protoc file>```

to test the installation :

```
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```
if it's working correctly it should be written like this
```
...
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 91.767s

OK (skipped=1)
```

### Problem that we encountered and how to fix it :
- pycocotools error 
>install C++ Build Tools using this link https://visualstudio.microsoft.com/visual-cpp-b and restart PC 
- tf-object-detection not found
>change tf-models-official to tf-models-nightly in the setup.py

### 2. Data Preprocessing

We use this dataset from a paper pubslished on ICCV 2019 
the paper link :
>https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf

the github link :
>https://github.com/xpwu95/ldl

the dataset link :
>https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ

We still need to properly clean the data and split the data between train and validation, you can use a note called img_note.txt to know which picture should be deleted because of broken xml file (of course you can re-label the picture using labelimg but we don't have any expertise to properly labeling the image) 

### 3. Generating Training Data
After labeling the image we should convert those xml files to csv using the xml_to_csv.py then use this command

```python xml_to_csv.py```

Different than the xml_to_csv file we need to tweak a few thing in this code if you want to change the label

```
def class_text_to_int(row_label):
    if row_label == 'fore':
        return 1
    else:
        return None
```

you should replace the row_label parameter and if you want to add another label then use elif condition returning 2 and so on.


Then we need to convert the csv file to the tfrecord, use this command

```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
```
```
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

### 4. Getting Ready for Training
We need a label map that maps an id to a name. We will put label_map.pbtxt in a folder called training, which is located in the object_detection directory. here's what the code is about

```
item {
    id: 1
    name: 'fore'
}
```

Lastly, we need to create a training configuration file. As a base model, we use ssd mobilenet v2 320x320 because we were having issue with other model because of limitaion in GPU. The Tensorflow OD API provides a lot of different models. For more information check out this link bellow and download the model you want to use.
>https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Then extract the model to the 
>Tensorflow\models\research\object_detection\

After we extract the model we need to edit the config for the model can be found inside the configs/tf2 folder.

Copy the config file to the training directory. Then open it inside a text editor and make the following changes:

>Line 13: change the number of classes to number of objects you want to detect

>Line 141: change fine_tune_checkpoint to the path of the model.ckpt file:

>fine_tune_checkpoint: "<path>/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"

>Line 143: Change fine_tune_checkpoint_type to detection

>Line 182: change input_path to the path of the train.records file:

>input_path: "<path>/train.record"

>Line 197: change input_path to the path of the test.records file:

>input_path: "<path>/test.record"

>Line 180 and 193: change label_map_path to the path of the label map:

>label_map_path: "<path>/label_map.pbtxt"

>Line 144 and 189: change batch_size to a number appropriate for your hardware, like 4, 8, or 16. In our case we only can fit 1.

### 5. Training the model
Execute this command below to train the model

```
python model_main_tf2.py --pipeline_config_path=training/ssd_mobilenet_v2_320x320_coco17_tpu-8.config --model_dir=training --alsologtostderr
```

and use this command to export the model we trained

```
python exporter_main_v2.py \ --trained_checkpoint_dir training \ --output_directory inference_graph \ --pipeline_config_path training/ssd_mobilenet_v2_320x320_coco17_tpu-8.config
```


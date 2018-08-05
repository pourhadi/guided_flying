

declare -r GCS_BUCKET="gs://guided-flying"
declare -r UNIQUE="$(date +%Y%m%d_%H%M%S)"
declare -r JOB_ID="$1"
declare -r JOB_DIR="${GCS_BUCKET}/training/jobs/${JOB_ID}"

gsutil cp pipeline.config ${GCS_BUCKET}/data

cd ~/sgws_model/models/research && python setup.py sdist
cd ~/sgws_model/models/research/slim && python setup.py sdist

cd ~/sgws_model/models/research && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

### Train

if [ "$2" != "eval_only" ]
then
cd ~/sgws_model/models/research && gcloud ml-engine jobs submit training ${JOB_ID}_${UNIQUE} \
    --runtime-version 1.5 \
    --job-dir=${JOB_DIR}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --config ~/sgws_model/sgsw-prototype/cloud.yml \
    -- \
    --train_dir=${JOB_DIR}/train \
    --pipeline_config_path=${GCS_BUCKET}/data/pipeline.config
fi

if [ "$2" != "train_only" ]
then
    ### Eval

cp ~/sgws_model/sgsw-prototype/dist/pycocotools-2.0.tar.gz ~/sgws_model/models/research/dist/pycocotools-2.0.tar.gz
cd ~/sgws_model/models/research && gcloud ml-engine jobs submit training eval_${JOB_ID}_${UNIQUE}  \
    --runtime-version 1.5 \
    --job-dir=${JOB_DIR}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,dist/pycocotools-2.0.tar.gz \
    --module-name object_detection.eval \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=${JOB_DIR}/train \
    --eval_dir=${JOB_DIR}/eval \
    --pipeline_config_path=${GCS_BUCKET}/data/pipeline.config
fi


#     python ~/Projects/models/research/object_detection/dataset_tools/create_pet_tf_record.py \
#     --label_map_path=labels.pbtxt \
#     --data_dir=`pwd` \
#     --output_dir=`pwd`


    ### Eval

#     cd ~/Projects/models/research && python setup.py sdist
# cd ~/Projects/models/research/slim && python setup.py sdist

# cd ~/Projects/models/research && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# cd ~/Projects/models/research && gcloud ml-engine jobs submit training eval_job_20180521_090148_2 \
#     --runtime-version 1.5 \
#     --job-dir=gs://sgws_model/training/jobs/job_20180521_090148/train \
#     --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,dist/pycocotools-2.0.tar.gz \
#     --module-name object_detection.eval \
#     --region us-central1 \
#     --scale-tier BASIC_GPU \
#     -- \
#     --checkpoint_dir=gs://sgws_model/training/jobs/job_20180521_090148/train \
#     --eval_dir=gs://sgws_model/training/jobs/job_20180521_090148/eval \
#     --pipeline_config_path=gs://sgws_model/data/pipeline.config
    

#     cd ~/Projects/models/research && gcloud ml-engine jobs submit training job_20180516_150306 \
#     --runtime-version 1.5 \
#     --job-dir=gs://sgws_classification_model/coco/training/jobs/job_20180516_150306 \
#     --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
#     --module-name object_detection.train \
#     --region us-central1 \
#     --config ~/Projects/coco/cloud.yml  \
#     -- \
#     --train_dir=gs://sgws_classification_model/coco/training/jobs/job_20180516_150306/train \
    # --pipeline_config_path=gs://sgws_classification_model/coco/data/pipeline.config

    # python convert_to_tfrecord.py --output_path=eval.record --images_dir=images/images/eval/ --labels_dir=images/annotations/eval/


    # cd ~/Projects/models/research && gcloud ml-engine jobs submit training job_8_2 \
    # --runtime-version 1.5 \
    # --job-dir=${GCS_BUCKET}/training/jobs/job_8/train \
    # --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    # --module-name object_detection.train \
    # --region us-central1 \
    # --config ~/Projects/models/research/object_detection/samples/cloud/cloud.yml \
    # -- \
    # --train_dir=${GCS_BUCKET}/training/jobs/job_8/train \
    # --pipeline_config_path=${GCS_BUCKET}/data/pipeline.config


# gsutil cp gs://sgws_model/training/jobs/job_20180520_200530/train/model.ckpt-89929.* .
#     cd ~/Projects/models/research && python object_detection/export_inference_graph.py \
#     --input_type image_tensor \
#     --pipeline_config_path gs://sgws_model/data/pipeline.config \
#     --trained_checkpoint_prefix model.ckpt-89929 \
#     --output_directory ~/Projects/models/output/mobilenet/v1



#     freeze_graph.py --input_graph=/Users/danielpourhadi/Projects/tensorflow/output/v1/frozen_inference_graph.pb \
#   --input_checkpoint=model.ckpt-89929 \
#   --input_binary=true \
#   --output_graph=/tmp/frozen_mobilenet_v1_224.pb \
#   --output_node_names=MobileNetV1/Predictions/Reshape_1

#   bazel run --config=opt \
#   tensorflow/contrib/lite/toco:toco -- \
#   --savedmodel_directory=/Users/danielpourhadi/Projects/models/output/mobilenet/v1/saved_model \
#   --output_file=/Users/danielpourhadi/Projects/models/output/mobilenet/v1/model.tflite \
# --input_shapes=1,400,400,3 \
#   --input_arrays=image_tensor

#   bazel run --config=opt \
#   //tensorflow/contrib/lite/toco:toco -- \
#   --input_file=/Users/danielpourhadi/Projects/tensorflow/output/v1/frozen_inference_graph.pb \
#   --output_file=/tmp/model.tflite \
#   --inference_type=FLOAT \
#   --input_shape=1,400,400,3 \
#   --input_array=image_tensor \
#   --output_array=final_result

#   python freeze_graph.py --input_graph=/Users/danielpourhadi/Downloads/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb   --input_checkpoint=/Users/danielpourhadi/Projects/tensorflow/output/v1/model.ckpt   --input_binary=true   --output_graph=/tmp/frozen_mobilenet_v1_224.pb   --output_node_names=MobileNetV1/Predictions/Reshape_1
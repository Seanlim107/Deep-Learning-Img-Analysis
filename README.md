# Deep-Learning-Img-Analysis
Download dataset and checkpoints from https://drive.google.com/drive/folders/1KDyaRq4JP8JW2J-1oDnj7zZR_eJEQiUf?usp=drive_link
Definition:
run_job and run_job_alt run the same file corresponding to the index, run_job uses prigpu, while run_job_alt uses preemptgpu.
The index following with the file and stage is as follows:
1) runs train.py: For training the CNN backbone for classifying 26 classes, checkpoint is "BaselineCNN_ckpt.pth"
2) runs train_contrastive.py: For training the contrastive network. Checkpoint is "My_Contrastive_No_Backbone_ckpt.pth"
2_2 runs train_contrastive_backbone.py, which loads the checkpoint from (1) and loads it as a backbone. By default, CNCL is available while CNFCL requires changing the "type" parameter in config.yaml to 1 and also renaming the variable cont_model_name in either train_ZS.py or train_contrastive_backbone.py to "My_Contrastive_Backbone_ckpt.pth" to function. Checkpoint is "My_Contrastive_Backbone_Fixed_ckpt.pth"
train_contrastive.py runs from scratch. The reason I seperated them is because I wanted to run them in parallel on Hyperion.
3) runs train_ZL.py: For evaluating Zero Shot Learning. Checkpoint is "My_Contrastive_Backbone_Fixed_ZS.pth"
4) runs inference_py: Provides best plots for contrastive learning phase and zero shot learning phase.

Setup:
1) Python 3.9.5
2) Simply running the file create_env.sh should do the trick.

Training:

1) Place the checkpoints in the folder.
2) Run the files depending on the index.
3) If hyperparameters are to be changed, the run_job files should be run in the following order: "1", "2_2", "3". Furthermore, the checkpoints created from the files have to be renamed to the names of the checkpoints above.
4) the parameters batch_size, train_size, simi_ratio and all parameters in train can be changed at any given time.

Inference:
1) run it as is, ensure that BaselineCNN_Fixed_ckpt.pth is in the main directory.
2) files should be available in the folders inference_ZS and inference_Contrastive, which may be created if it doesnt exist in the main directory.
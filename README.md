## Requirements
- pytorch >= 1.12.1
- torchvision
- pyyaml
- tensorboard
- opencv-python
- tqdm
- kornia

## Installation
You can replace the second command from the bottom to install
[pytorch](https://pytorch.org/get-started/previous-versions/#v110) 
based on your CUDA version.
```
conda create -n py38torch1121 python=3.8
conda activate py38torch1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install pyyaml tensorboard opencv-python tqdm kornia
```


## Download the dataset
Download the Trainset, Valset/Testset (gallery and probe) from the [Track 1 Codalab website](https://codalab.lisn.upsaclay.fr/competitions/20040).
You should decompress these files by following command:
```
# For Phase 1
mkdir mgr_2024_track1_phase1
unzip MGR24_TrainSet_Parsing_pkl_1000IDs.zip
mv MGR24_TrainSet_Parsing_pkl_1000IDs/* mgr_2024_track1_phase1/
rm MGR24_TrainSet_Parsing_pkl_1000IDs -rf

unzip MGR24_ValSet_Parsing_Gallery_pkl.zip
mv MGR24_ValSet_Parsing_Gallery_pkl/* mgr_2024_track1_phase1/
rm MGR24_ValSet_Parsing_Gallery_pkl -rf

unzip MGR24_ValSet_Parsing_Probe_pkl.zip
mkdir -P mgr_2024_track1_phase1/probe
mv MGR24_ValSet_Parsing_Probe_pkl/* mgr_2024_track1_phase1/probe/
rm MGR24_ValSet_Parsing_Probe_pkl -rf

# For Phase 2
mkdir mgr_2024_track1_phase2
unzip MGR24_TrainSet_Parsing_pkl_1000IDs.zip
mv MGR24_TrainSet_Parsing_pkl_1000IDs/* mgr_2024_track1_phase2/
rm MGR24_TrainSet_Parsing_pkl_1000IDs -rf

unzip MGR24_TestSet_Parsing_Gallery_pkl.zip
mv MGR24_TestSet_Parsing_Gallery_pkl/* mgr_2024_track1_phase2/
rm MGR24_TestSet_Parsing_Gallery_pkl -rf

unzip MGR24_TestSet_Parsing_Probe_pkl.zip
mkdir -P mgr_2024_track1_phase2/probe
mv MGR24_TestSet_Parsing_Probe_pkl/* mgr_2024_track1_phase2/probe/
rm MGR24_TestSet_Parsing_Probe_pkl -rf
```
## model zoo
Results and models are available in the [best weights](https://drive.google.com/drive/folders/1W9YQIHRC9tWizKkrIeaxoLL8296HDYkq?usp=sharing) and [pre-trained weights](https://drive.google.com/drive/folders/1pvkSFfMoAwHK8gQUDAmZwGZSn6tVUHZc?usp=sharing).


## Train the dataset
For the phase 1:

Modify the `dataset_root` in `configs/parsinggait/parsinggait_mgr_track1_phase1.yaml`, and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/parsinggait/parsinggait_mgr_track1_phase1.yaml --phase train
```

For the phase 2, please replace 'phase1' with 'phase2' in the config file name.


## Generate the result
For the phase 1:

Modify the `dataset_root` in `configs/parsinggait/parsinggait_mgr_track1_phase1.yaml`, put the weight in `Gait3D-Benchmark/output/Gait3D-Parsing/ParsingGait/ParsingGait/checkpoints/` and then run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/parsinggait/parsinggait_mgr_track1_phase1.yaml --phase test
```
The result will be generated in `MGR_result/current_time.csv`.

For the phase 2, please replace 'phase1' with 'phase2' in the config file name.


---


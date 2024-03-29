# Timestamp-Supervised Action Segmentation from the Perspective of Clustering (TSASPC)

This repository provides a PyTorch implementation of the paper [Timestamp-Supervised Action Segmentation from the Perspective of Clustering. (IJCAI 2023)](https://arxiv.org/abs/2212.11694)

## Environment

Our environment:
```
Ubuntu 16.04.7 LTS
CUDA Version: 11.1
```
Based on anaconda or miniconda, you can install the required packages as follows:
```
conda create --name timestamp python=3.9
conda activate timestamp
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib
pip install tensorboard
pip install xlsxwriter
pip install scikit-learn
```

## Prepare Data
* Download the three datasets, which contains the features and the ground truth labels. (~30GB) (try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)))
* Extract the compressed file to the `data/` folder.
* The three `.npy` files in the folder `data/` are the timestamp annotations provided by [Li et al.](https://github.com/ZheLi2020/TimestampActionSeg).

## Pseudo-Label Ensembling (PLE)
Before training, we regard PLE as a pre-processing step since it relies on only the visual features of frames. You can run the following commands to generate pseudo-label sequences by the PLE algorithm for all videos in the 50salads dataset:
```
python generate_pseudo.py --dataset 50salads --metric euclidean --feature 1024   # RGB features
python generate_pseudo.py --dataset 50salads --metric euclidean --feature 2048   # optical flow features
python intersection_pseudo.py --dataset 50salads
```
Afterwards, you can find the generated pseudo-label sequences in the folder `data/I3D_merge/50salads/`, and the console will also output the evaluation metrics for the pseudo-label sequences: labeling rate and accuracy of pseudo-labels.
```
label rate: 0.5117809793880469
label acc: 0.9549147886799857
```
You can also use above commands (change the `--dataset` argument) to generate pseudo-label sequences for other two datasets.

## Iterative Clustering (IC)
After PLE, you can train the segmentation model with the IC algorithm.
```
python main.py --action=train --dataset=DS --split=SP
```
where `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other datasets. 
* The output of evaluation is saved in `result/` folder as an excel file. 
* The `models/` folder saves the trained model and the `results/` folder saves the predicted action labels of each video in test dataset.

Here is an example:
```
python main.py --action=train --dataset=50salads --split=2
# F1@0.10: 77.8032
# F1@0.25: 75.0572
# F1@0.50: 64.0732
# Edit: 68.2274
# Acc: 79.3653
```
Please note that we follow [the protocol in MS-TCN++](https://github.com/sj-li/MS-TCN2/issues/2) when evaluating, which is to select the epoch number that can achieve the best average result for all the splits to report the performance.

**If you get error: `AttributeError: module 'distutils' has no attribute 'version'`, you can install a lower version of setuptools:**
```
pip uninstall setuptools
pip install setuptools==59.5.0
```
## Evaluation
Normally we get the prediction and evaluation after training and do not have to run this independently. In case you want to test the saved model again by prediction and evaluation, please change the `time_data` in `main.py` and run:
```
python main.py --action=predict --dataset=DS --split=SP
```

## Acknowledgment

The model used in this paper is a refined MS-TCN model. Please refer to the paper [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://github.com/yabufarha/ms-tcn). We adapted the code of the PyTorch implementation of [Li et al.](https://github.com/ZheLi2020/TimestampActionSeg). Thanks to the original authors for their works!


## Citation

```
@inproceedings{du2023timestamp,
  title={Timestamp-Supervised Action Segmentation from the Perspective of Clustering},
  author={Du, Dazhao and Li, Enhan and Si, Lingyu and Xu, Fanjiang and Sun, Fuchun},
  booktitle={IJCAI},
  year={2023}
}
```

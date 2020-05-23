# KWS_Max-pooling_RHE
pytorch source code for paper "Mining effective negative training samples for keyword spotting"
Please cite the work below if you want to use the code or want to do research related to our work

```
@inproceedings{hou2020small,
  title={Mining Effective Negative Training Samples for Keyword Spotting},
    author={Jingyong Hou and Yangyang Shi and Mari Ostendorf and Mei-Yuh Hwang and Lei Xie},
      booktitle={ICASSP2020},
        year={2020}
}
```

## Running environment
### Python 2.7.15
### pytorch 0.4.1
### CUDA 8.0 or higher
### Kaldi
You should know basic knowledge of Kaldi before looking at the run script. I use Kaldi to extract Fbank features and do a global CMVN using the statictics from all training set. You should add cmd.sh, path.sh, steps and utils to your working dir before you run the script.

### You may first follow run_data_prepare.sh to prepare the data and extract the features. Then, please follow the run_train_proposed_public_ticmini2.sh script to learn how to run the code.

## Results on public dataset: https://www.openslr.org/87 
This dataset is different with the dataset used in above paper, the positive data is identical with the positive data in the paper, but the negative data is newlly recorded and totally different with the negative data in the paper.

Using default configuration in the paper:
![image](https://github.com/jingyonghou/KWS_Max-pooling_RHE/raw/master/det_curves/gru_rhe.png)


##TODO: 
* Data Augmentation experiment
* Further tuning of hyperparameters on the public dataset


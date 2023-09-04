# Active Learning through Segmentation for Crowd Counting

## Introduction

This project implements "ActiveSegCrowd" approach for crowd counting task, which is a novel strategy that applying active learning in crowd counting.

<img src="https://raw.githubusercontent.com/ZihanDai/images/master/pipeline.png" style="zoom:50%;" />

## Installation

This repository is build upon the [deepALplus](https://github.com/SineZHAN/deepALplus/tree/master) [1]. To install the project, please run the following command after download the project:

1. Running **requirement.txt** file

   ```
   pip install -r requirements.txt    
   ```

2. Installing the [faiss](https://github.com/facebookresearch/faiss) manually, 

   GPU version:

   ```
   pip install faiss-gpu
   ```

   CPU version:

   ```
   pip install faiss-cpu
   ```

3. Installing the scikit-image

   ```
   pip install scikit-image
   ```

## Usage

To successfully run the active learning for crowd counting, please follow the instructions outlined below:

#### Step 1: Generate a Kaggle API Token

1. **Log In to Kaggle**
   - Visit [Kaggle's website](https://www.kaggle.com/) and log in to your account.

2. **Access API Settings**

   <img src="https://raw.githubusercontent.com/ZihanDai/images/master/%E6%88%AA%E5%B1%8F2023-09-04%2016.23.14.png" alt="w" style="zoom:25%;" />

   - Click on the user icon situated at the top right corner.
   - Select 'Settings' from the dropdown menu.

3. **Create a New API Token**

   <img src="https://raw.githubusercontent.com/ZihanDai/images/master/%E6%88%AA%E5%B1%8F2023-09-04%2016.24.43.png" style="zoom:25%;" />

   - Scroll down to locate the 'API' section.
   - Click on the 'Create New API Token' button to generate a token.

4. **Setup API Token**

   - A "kaggle.json" file will be downloaded to your system.
   - Move this file to the root directory of your project (i.e., within the 'Final_Project' folder).

#### Step 2: Executing the Program

In this repository, we have implemented four state-of-the-art crowd counting approaches: "HeadCount", "ConsistencyCrowd", "CrowdRank", and "ActiveSegCrowd". You can execute any of these approaches by modifying the `--ALstrategy` parameter in the command line as shown below. The parameters you can configure are as follows:

- **ALstrategy**: Specifies the active learning strategy to be employed. Replace "RandomSampling" with your chosen strategy.
- **initseed**: Indicates the initial subset of the dataset to be used.
- **quota**: Represents the labeling budget available.
- **batch**: Specifies the number of labels queried in each active learning cycle.
- **dataset_name**: Indicates the name of the dataset to be used. The options available are: "shanghaitechA [2]", "shanghaitechB [2]", "UCF-QNRF [3]", and "IOCfish [4]".
- **iteration**: Sets the number of repetitions for the active learning process.
- **initsampler**: (New) This parameter defines the sampler used to construct the initial pool of the dataset. Though the default setting is "Random", it can be replaced with a more intelligent selection strategy.
- **tag**: Used to customize the output file name.
- **net**: Allows for the change of the network utilized. The default network is CSRNet, but SGANet [5] has also been implemented (-net sgnaet).
- **-g**: Specifies the GPU ID for running the script (0 in this case).

##### Example Command:

```
python main.py \
      --ALstrategy RandomSampling \
      --initseed 10 \
      --quota 60 \
      --batch 10 \
      --dataset_name shanghaitechA \
      --seed 1 \
      --iteration 3 \
      --tag demo \
      --net sganet \
      -g 0
```

In this example command, the active learning strategy is set to "RandomSampling". Modify the parameters as needed to suit your specific requirements.

##### Note:

Training the crowd counting model using SGANet requires to change the hyperparameter in the 'parameters.py' file.

# Reference

[1] Zhan, X., Wang, Q., Huang, K., Xiong, H., Dou, D., & Chan, A. B. (2022). A comparative survey of deep active learning. arXiv preprint arXiv:2203.13450.

[2] Y. Zhang, D. Zhou, S. Chen, S. Gao and Y. Ma, "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network," *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*

[3] H. Idrees, M. Tayyab, K. Athrey, D. Zhang, S. Al-Maddeed, N. Rajpoot, M. Shah, Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds

[4] Sun, G., An, Z., Liu, Y., Liu, C., Sakaridis, C., Fan, D-P., & Van Gool, L. (2023). Indiscernible object counting in underwater scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR).

[5] Wang, Q., & Breckon, T.P. (2022). Crowd Counting via Segmentation Guided Attention Networks and Curriculum Loss. IEEE Transactions on Intelligent Transportation Systems. IEEE.

# ConFusion-Net: Semantic Segmentation for Scene Point Clouds via Learning Context Fusion Features

This is the official implementation of **ConFusion-Net**, more details will be provided after the article is accepted. 

## Getting Started
### (0) Code structure
We adapt the codebase of [RandLA-Net](https://github.com/QingyongHu/RandLA-Net), thanks for their siginificant contribution in semantic segmentation of point clouds.

```
├── ConFusion-Net
    ├── ConFusionNet.py # network architecture
    ├── data_prepare.py # data preprocessing
    ├── hepler_ply.py # .ply reader & writer
    ├── hepler_tf_util.py # tensorflow modules
    ├── helper_tool.py # dataset configs and data tools
    ├── main_SensatUrban.py # main file
    ├── tester_SensatUrban.py # model tester
    ├── vis_SensatUrban.py # visualization file
    ├── cpp_wrappers # cpp extension files
    │   └── ...
    ├── utils # cpp build files
    │   └── ...
    └── compile_op.sh # compile all extensions
```

### (1) Setup
This code has been tested with Python 3.6, Tensorflow 2.5.0, CUDA 11.6 and cuDNN 8.8.0 on Ubuntu 20.04.
 
- Clone the repository 
```
git clone --depth=1 https://github.com/Kukdo/ConFusion-Net
cd ConFusion-Net
```

- Setup python environment
```
conda create -n cfnet python=3.6
conda activate cfnet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Training & Evaluation
- Download the dataset
The [SensatUrban](http://arxiv.org/abs/2009.03137) dataset can be downloaded from [here](https://forms.gle/m4HJiqZxnq8rmjc8A). Then uncompress it to root folder and rename as `data`.

The data should be in the following format:
```
ConFusion-Net/data
          └── train/
                  ├── birmingham_block_0.ply
                  ├── birmingham_block_1.ply 
		  ...
	    	  └── cambridge_block_34.ply 
          └── test/
                  ├── birmingham_block_2.ply
		  ├── birmingham_block_8.ply
		  ...
	    	  └── cambridge_block_27.ply 
```

- Data preprocessing (This process may take a while)
```
python data_prepare.py --dataset_path data
```

- Data reorganization
```
mkdir dataset; cd dataset; mkdir SensatUrban; cd SensatUrban;
mkdir original_block_ply; mv ../../data/train/* original_block_ply; mv ../../data/test/* original_block_ply;
mv ../../grid* ./; cd ../../; rm -r data;
```

The processed data should be in the following format:
```
ConFusion-Net/dataset/
          └── original_block_ply/
                  ├── birmingham_block_0.ply
                  ├── birmingham_block_1.ply 
		  ...
	    	  └── cambridge_block_34.ply 
          └── grid_0.200/
	     	  ├── birmingham_block_0_KDTree.pkl
                  ├── birmingham_block_0.ply
		  ├── birmingham_block_0_proj.pkl 
		  ...
	    	  └── cambridge_block_34.ply 
```

- Training: 
```
python main_SensatUrban.py --mode train --gpu 0 
```

- Evaluation:
```
python main_SensatUrban.py --mode test --gpu 0 
```

### (3) Visulization
Modify the `base_dir, save_dir, original_data_dir` to your own path.
```
python vis_SensatUrban.py
# original_data_dir <- the input collected point clouds path (.ply files)
# base_dir <- semantic prediction path (.label files)
# save_dir <- output semantic point clouds path (.ply files)
```
If you want to visualize the GT semantics, uncomment the `GT = Plot(...)` and `save_ply_o3d(GT, ...)`.

### (4) Testing your own data
Copy your .ply to the `data/test` and preprocess again, then modify or add your `$YOURNAME.ply` into `main_SensatUrban.py`'s test_file_name list.

Run `python main_SensatUrban.py --mode test --gpu 0` for labeling your own data using the model trained from SensatUrban dataset.

## BibTeX
```
@article{Zhneg24CFNet,
  title     = {ConFusion-Net: Semantic Segmentation for Scene Point Clouds via Learning Context Fusion Features},
  author    = {Zheng, Liu and Zhipeng, Jiang and Jianjun, Zhang and Ming, Zhang† and Renjie, Chen and Ying, He},
  booktitle = {arXiv},
  year      = {2024}
}
```

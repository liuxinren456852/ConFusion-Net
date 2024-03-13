# ConFusion-Net: Semantic Segmentation for Scene Point Clouds via Learning Context Fusion Features

This is the official implementation of **ConFusion-Net**, more details will be provided after the article is accepted. 
 


	
### (1) Setup
This code has been tested with Python 3.6, Tensorflow-gpu 2.5.0, CUDA 11.6 and cuDNN 8.8.0 on Ubuntu 20.04.
 
- Clone the repository 
```
git clone --depth=1 https://github.com/Kukdo/ConFusion-Net && cd ConFusion-Net
```
- Setup python environment
```
conda create -n cfnet python=3.6
conda activate cfnet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Training & Evaluation
- Preparing the dataset
```
python input_preparation.py --dataset_path $YOURPATH
cd $YOURPATH; 
cd ../; mkdir original_block_ply; mv data_release/train/* original_block_ply; mv data_release/test/* original_block_ply;
mv data_release/grid* ./
```
The data should organized in the following format:
```
/Dataset/SensatUrban/
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

- Start training: 
```
python main_SensatUrban.py --mode train --gpu 0 
```
- Evaluation:
```
python main_SensatUrban.py --mode test --gpu 0 

# City-scale-Road-Audit
Website - https://cvit.iiit.ac.in/research/projects/cvit-projects/city-scale-road-audit

## Publications
Accepted at IROS. Will be published soon!!

If you use this software in your research, please cite our publications:
## Packages
For instructions please refer to the README on each folder:

* [train](train) contains tools for training the network for semantic segmentation.Use python main_iros_road_combined_train.py --savedir release_version_test --datadir /Neutron6/sudhirkumar/DataSet/release_version_v1/ --num-epochs <> --batch-size <> --decoder --iouVal
* [trained_models](trained_models) Contains the trained models used in the papers. NOTE: the pytorch version is slightly different from the torch models.

## Requirements:

* [**The dataset**](http://bit.ly/road-audit-dataset/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels.
* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
* [**PyTorch 0.2 and above**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0). 
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)

In Anaconda you can install with:
```
conda install numpy matplotlib torchvision Pillow
conda install -c conda-forge visdom
```

If you use Pip (make sure to have it configured for Python3.6) you can install with: 

```
pip install numpy matplotlib torchvision Pillow visdom
```

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/

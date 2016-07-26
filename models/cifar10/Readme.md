### Cifar 10 residual network models.
The script [build_resnet.py](../../app/cifar10/build_resnet.py) can be used to generate cifar-10 models described in the residual networks [paper](https://arxiv.org/abs/1512.03385).

```
usage: build_resnet.py [-h] -n N -m MAIN_BRANCH [-f FIRE_FILTER_MULT]
                       [-o OUTPUT_FOLDER]

This script generates cifar10 resnet train_val.prototxt files

optional arguments:
  -h, --help            show this help message and exit
  -n N, --N N           Number of block per stage (or N), as described in
                        paper. Total number of layers will be 3N + 2
  -m MAIN_BRANCH, --main_branch MAIN_BRANCH
                        normal, bottleneck
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Train and Test prototxt will be generated as
                        train.prototxt and test.prototxt
```

As an example the [resnet_110](./resnet_110) folder contains the prototxt files generated for training the 110 layer network for cifar 10 dataset. The model files can be generated as follows:
```
python app/cifar10/build_resnet.py -m bottleneck -n 36 -o ./

Output:
....
....
Number of params:  0.220944  Million
Number of flops:  30.73296  Million
```

#### Some results

***Note*** - We do not use augmentation, the numbers are 2-3% below the ones reported in original paper, this is just to demonstrate how to use pynetbuilder to reproduce residual networks.

 * Training batch size 128
 * LR - 0.1, gamma 0.1.  Steps 32K, 48k.
 * Iterations - 60K
 
| Model | Accuracy |
|---:|---:|
|Resnet_20|0.8795|
|Resnet_32|0.8922|
|Resnet_44|0.892|
|Resnet_56|0.8896|
|Resnet_110|0.8921|

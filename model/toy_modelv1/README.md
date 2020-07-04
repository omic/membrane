# VGGVox

* Python adaptation of VGGVox speaker identification model, based on Nagrani et al 2017, "[VoxCeleb: a large-scale speaker identification dataset](https://arxiv.org/pdf/1706.08612.pdf)"
* Evaluation code only, based on the author's [Matlab code](https://github.com/a-nagrani/VGGVox/)
and [pretrained model](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

## Instructions
* Install python3 and the required packages
* To run:
```
python3 verification.py verify --input [input file] --test [test file] --metric [metric function (default:'cosine')] --threshold [threshold of metric function for verification (default:0.1)]
```
An example:
```
python3 verification.py verify --input data/wav/enroll/19-enroll.wav --test data/wav/test/19.wav --metric 'cosine' --threshold 0.1
```
* Results will be stored in `res/results.csv`. Each line has format: `[input file name], [test file name], [metric function], [distance], [threshold], [verified?]`
```


# DM413

code for  "Adaptive Distribution Calibration for FSL via Optimal Transport"

## Backbone Training
We use the same backbone network and training strategies as [S2M2_R](https://arxiv.org/pdf/1907.12087.pdf). Please refer to [here](https://github.com/nupurkmr9/S2M2_fewshot) for the backbone training.

## Extract and save features
After training the backbone as [S2M2_R](https://arxiv.org/pdf/1907.12087.pdf), extract features as below:

Create an empty `checkpoints` directory. Then run: 
```python
python save_plk.py --dataset [miniImagenet/CUB] 
```

Or you can directly download the extracted features/pretrained models from the [link](
https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing)(for miniimagenet and cub) and [link](https://drive.google.com/file/d/1CV7VdGTYffeS964Ov0P78YsdnMz3DCWY/view?usp=sharing).

After downloading the extracted features, please adjust your file path according to the code.

## Evaluate our distribution calibration
To evaluate our distribution calibration method, run:
```python
python evaluate_ADC.py
```



**Reference**

1. [Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://github.com/nupurkmr9/S2M2_fewshot)

2. [Free Lunch for Few-Shot Learning: Distribution Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration#iclr2021-oral-free-lunch-for-few-shot-learning-distribution-calibration)

3. [Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://github.com/yhu01/PT-MAP)


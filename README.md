# RCCR
(ICME 2022) Domain Adaptive Semantic Segmentation via Regional Contrastive Consistency Regularization

## Citing RCCR
If you find RCCR useful in your research, please consider citing:
```bibtex
@inproceedings{zhou2022domain,
  title={Domain adaptive semantic segmentation with regional contrastive consistency regularization},
  author={Zhou, Qianyu and Zhuang, Chuyun and Lu, Xuequan and Ma, Lizhuang},
  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2022},
  organization={IEEE}
}
```

### Requirements
*  CUDA/CUDNN 
*  Python3
*  Pytorch
*  Scipy==1.2.0
*  Other requirements
    ```bash
    pip install -r requirements.txt
    ```

# Run training and testing

### Example of training a model with unsupervised domain adaptation on GTA5->CityScapes on a single gpu
    ```bash
    python3 train_spasr_gtav.py --config ./configs/configUDA_spasr_gtav.json --name UDA
    ```


### Example of testing a model with domain adaptation with CityScapes as target domain
    ```bash
    python3 evaluateUDA.py --model-path *checkpoint.pth*
    ```

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [DACS](https://github.com/vikolss/DACS)


## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.

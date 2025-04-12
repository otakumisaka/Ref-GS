# Ref-GS : Directional Factorization for 2D Gaussian Splatting

## [Project Page](https://ref-gs.github.io/) | [Paper](https://arxiv.org/pdf/2412.00905) | [arXiv](https://arxiv.org/abs/2412.00905)

> Ref-GS : Directional Factorization for 2D Gaussian Splatting<br>
> [Youjia Zhang](https://ref-gs.github.io/), [Anpei Chen](https://apchenstu.github.io/), [Yumin Wan](https://ref-gs.github.io/), [Zikai Song](https://skyesong38.github.io/), [Junqing Yu](https://scholar.google.com/citations?hl=zh-CN&user=_UjqBfcAAAAJ), [Yawei Luo](https://scholar.google.com/citations?hl=zh-CN&user=pnVwaGsAAAAJ), [Wei Yang](https://weiyang-hust.github.io/)<br>
> CVPR 2025

![teaser](assets/teaser.jpg)

## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n ref_gs python=3.7.16
conda activate ref_gs
pip install -r requirements.txt
```

## 📦 Dataset
We mainly test our method on [Shiny Blender Synthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip), [Shiny Blender Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), [Glossy Synthetic](https://liuyuan-pal.github.io/NeRO/) and [NeRF Synthetic dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Please run the script `nero2blender.py` to convert the format of the Glossy Synthetic dataset.

Put them under the `data` folder:
```bash
data
└── refnerf
    └── car
    └── toaster
└── nerf_synthetic
    └── hotdog
    └── lego
```

## 🏃 Training
We provide the script to test our code on each scene of datasets. Just run:
```
sh train.sh
```
You may need to modify the path in `train.sh`

## 🫡 Acknowledgments

This work is built on many amazing research works and open-source projects,

- [NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections](https://dorverbin.github.io/nerf-casting/)
- [3D Gaussian Splatting with Deferred Reflection](https://github.com/gapszju/3DGS-DR/tree/main)
- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)

We are grateful to the authors for releasing their code.

## 📜 Citation

If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```
@article{zhang2024ref,
  title={Ref-GS: Directional Factorization for 2D Gaussian Splatting},
  author={Zhang, Youjia and Chen, Anpei and Wan, Yumin and Song, Zikai and Yu, Junqing and Luo, Yawei and Yang, Wei},
  journal={arXiv preprint arXiv:2412.00905},
  year={2024}
}
```

## Contact

For feedback, questions, or press inquiries please contact [Youjia Zhang](Youjiazhang@hust.edu.cn).

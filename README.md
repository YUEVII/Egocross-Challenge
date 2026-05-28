<h1 align="center">
The Right Inference Strategy Is All You Need: Nearly Training-Free Domain-Wise
Inference for EgoCross Challenge
</h1>

<p align="center">
Leyi Wu<sup>1,3,*</sup>, Yifan Zhao<sup>1,*</sup>, Jinjie
Zhang<sup>1,*</sup>, Yinchuan Li<sup>3</sup>, Yingcong
Chen<sup>1,2,†</sup>
</p>

<p align="center">
<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>Knowin
</p>

<p align="center">
Team Name: WFJ-KnowinEnvision
</p>

## News

- **2026.05** 🏆 We are honored to win **1st place** in the CVPR 2026 1st
  Cross-Domain EgoCross Challenge.

## Resources

- **Paper:** "The Right Inference Strategy Is All You Need: Nearly
  Training-Free Domain-Wise Inference for EgoCross Challenge" (link coming
  soon)
- **Challenge:** [1st Cross-Domain EgoCross Challenge @ EgoVis, CVPR
  2026](https://egocross-benchmark.github.io/)
- **Workshop:** [Third Joint Egocentric Vision Workshop @ CVPR
  2026](https://egovis.github.io/cvpr26)

## Environment

```bash
conda create -n egocross python=3.10
conda activate egocross
pip install torch accelerate "transformers>=4.57.0" qwen-vl-utils pillow pyyaml
```

Install the PyTorch build that matches your CUDA environment if the default
`pip install torch` wheel is not suitable for your machine.

## Test Set

Prepare the official EgoCross test set under `testset/`:

- JSON file:
  [egocross_testbed_imgs.json](https://github.com/MyUniverse0726/EgoCross/blob/main/datasets/egocross_testbed_imgs.json)
- Dataset:
  [myuniverse/EgoCross](https://huggingface.co/datasets/myuniverse/EgoCross)

The expected layout is:

```text
Egocross-Challenge/
└── testset/
    ├── egocross_testbed_imgs.json
    ├── CholecTrack20/
    ├── EgoSurgery/
    ├── ENIGMA/
    ├── ExtrameSportFPV/
    └── EgoPet/
```

## Checkpoints

Download checkpoints and place them under the `ckpts/` directory:

```text
Egocross-Challenge/
└── ckpts/
    ├── base_model/
    ├── industry/
    └── xsports/
```

- `ckpts/base_model/`: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- `ckpts/industry/`: [EgoCross_sft_qwen3vl4B_industry](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_industry)
- `ckpts/xsports/`: [EgoCross_sft_qwen3vl4B_xsports](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_xsports)

Animal, EgoSurgery, and CholecTrack20 runners use `ckpts/base_model/` by
default or by explicit argument. Industry and XSports use their domain-specific
SFT checkpoints together with `ckpts/base_model/`.

## Run

Run all commands from the repository root:

```bash
cd Egocross-Challenge
```

### Animal / EgoPet

```bash
python run_animal.py ckpts/base_model
```

### Industry / ENIGMA

```bash
python run_industry.py
```

Optional explicit checkpoint paths:

```bash
python run_industry.py \
  --base-model ckpts/base_model \
  --sft-model ckpts/industry
```

### XSports / ExtrameSportFPV

```bash
python run_xsports.py
```

Optional explicit checkpoint paths:

```bash
python run_xsports.py \
  --model-path ckpts/xsports \
  --base-model-path ckpts/base_model
```

### EgoSurgery

```bash
python run_EgoSurgery.py --model-path ckpts/base_model
```

### CholecTrack20

```bash
python run_CholecTrack20.py --model-path ckpts/base_model
```

Each runner reads the official test set from `testset/egocross_testbed_imgs.json`
and writes or updates `submission.json` by default.
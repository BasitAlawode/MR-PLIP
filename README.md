# MR-PLIP: Multi-Resolution Pathology-Language Model with Text-Guided Visual Representation


### Abstract
In Computational Pathology (CPath), the introduction of Vision-Language Models (VLMs) has opened new avenues for research, focusing primarily on aligning image text pairs at a single magnification level. However, this approach might not be sufficient for tasks like cancer subtype classification, tissue phenotyping, and survival analysis due to the limited level of detail that a single-resolution image can provide. Addressing this, we propose a novel multi-resolution paradigm leveraging Whole Slide Images (WSIs) to extract histology patches at multiple resolutions and generate corresponding textual descriptions through advanced CPath VLM. This method aims to capture a broader range of information, supported by novel loss functions, enriches feature representation, improves discriminative ability, and enhances generalization across different resolutions. Pre-trained on a comprehensive TCGA dataset with 34 million image-language pairs at various resolutions, our fine-tuned model outperforms State-Of-The-Art (SOTA) counterparts across multiple datasets and tasks, demonstrating its effectiveness in CPath.



## Environment Setup 

This setup is tested only on Linux.

1. Clone this repository (Alternatively: download this repository as a zip file and extract it)
```
git clone https://github.com/BasitAlawode/MR-PLIP.git
```

2. Navigate to MR-PLIP folder
```
cd MR-PLIP
```

2. Install Packages
```
conda create -n mrplip python=3.10 -y
conda activate mrplip
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Text Generation with Quilt-LLaVA
```
```

## Acknowledgement
 - Our work is based on [Quilt-LLaVA](https://github.com/aldraus/quilt-llava) and by extension the [LLaVA model](https://github.com/haotian-liu/LLaVA).


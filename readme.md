# VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation

<img src="documentation/assets/VoxTellLogo.png"/>

This repository will contain the official implementation of our paper:

### **VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation**

VoxTell is a **3D vision–language segmentation model** that directly maps free-form text prompts, from single words to full clinical sentences, to volumetric masks. By leveraging **multi-stage vision–language fusion**, VoxTell achieves state-of-the-art performance on anatomical and pathological structures across CT, PET, and MRI modalities, excelling on familiar concepts while generalizing to related unseen classes.

> **Authors**: Maximilian Rokuss*, Moritz Langenberg*, Yannick Kirchhoff, Fabian Isensee, Benjamin Hamm, Constantin Ulrich, Sebastian Regnery, Lukas Bauer, Efthimios Katsigiannopulos, Tobias Norajitra, Klaus Maier-Hein  
> **Paper**: [![arXiv](https://img.shields.io/badge/arXiv-2511.11450-B31B1B.svg)](https://arxiv.org/abs/2511.11450)

---

## Overview

VoxTell is trained on a **large-scale, multi-modality 3D medical imaging dataset**, aggregating **158 public sources** with over **62,000 volumetric images**. The data covers:

- Brain, head & neck, thorax, abdomen, pelvis  
- Musculoskeletal system and extremities  
- Vascular structures, major organs, substructures, and lesions  

<img src="documentation/assets/VoxTellConcepts.png"/>

This rich semantic diversity enables **language-conditioned 3D reasoning**, allowing VoxTell to generate volumetric masks from flexible textual descriptions, from coarse anatomical labels to fine-grained pathological findings.

---

## Architecture

VoxTell combines **3D image encoding** with **text-prompt embeddings** and **multi-stage vision–language fusion**:

- The **image encoder** maps volumetric input into latent features.  
- The **prompt decoder** transforms free-text input into multi-scale text embeddings.  
- The **image decoder** fuses visual and textual information at multiple resolutions, extending MaskFormer-style query–image fusion with **deep supervision**, producing high-fidelity volumetric masks.

<img src="documentation/assets/VoxTellArchitecture.png" />

---

## 🛠 Installation

Coming soon.

---

## Citation

```bibtex
@misc{rokuss2025voxtell,
      title={VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation}, 
      author={Maximilian Rokuss and Moritz Langenberg and Yannick Kirchhoff and Fabian Isensee and Benjamin Hamm and Constantin Ulrich and Sebastian Regnery and Lukas Bauer and Efthimios Katsigiannopulos and Tobias Norajitra and Klaus Maier-Hein},
      year={2025},
      eprint={2511.11450},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.11450}, 
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please contact:

📧 maximilian.rokuss@dkfz-heidelberg.de / moritz.langenberg@dkfz-heidelberg.de
# CUET-823 @ DravidianLangTech 2025
## Multimodal Misogyny Meme Detection in Tamil

System submission for the **Shared Task on Misogyny Meme Detection** at DravidianLangTech@NAACL 2025.  
**Result: F1 = 0.7812 | 4th place out of all participating teams.**

---

## Task
Binary classification of Tamil-English code-mixed memes as *misogynistic* or *non-misogynistic*, using both text and image modalities.

## Dataset
Provided by the shared task organizers (Ponnusamy et al., 2024).

| Split | Misogynistic | Non-Misogynistic | Total |
|-------|-------------|-----------------|-------|
| Train | 285 | 851 | 1,136 |
| Dev | 74 | 210 | 284 |
| Test | 89 | 267 | 356 |

## Approach

**Text:** Tamil-English code-mixed text was preprocessed (stopword removal, URL/emoji removal) and transliterated into Tamil script using the `indic-transliteration` library. Fine-tuned mBERT and IndicBERT.

**Image:** Images resized to 224×224 and normalized. Fine-tuned ResNet, EfficientNet, and ViT.

**Fusion:** Text and image features concatenated and passed through a fully connected classification head.

**Best model: mBERT + EfficientNet**
- Text embeddings: 768-dim (mBERT)
- Image features: 1280-dim (EfficientNet)
- Combined via dense FC layer → binary output

**Class imbalance handling:** Back-translation (Tamil→English→Tamil, Tamil→Malayalam→Tamil) and image augmentation (brightness, grayscale, posterization) applied to the minority class only.

## Results

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| mBERT (text only) | 0.68 | 0.71 | 0.69 |
| ResNet (image only) | 0.77 | 0.74 | 0.74 |
| **mBERT + EfficientNet** | **0.80** | **0.77** | **0.78** |
| mBERT + ViT | 0.79 | 0.72 | 0.75 |
| IndicBERT + ResNet | 0.79 | 0.73 | 0.75 |

## Repository Structure
```
├── notebooks/
│   ├── unimodal_text.ipynb       # mBERT and IndicBERT fine-tuning
│   ├── unimodal_image.ipynb      # ResNet, ViT, EfficientNet experiments
│   └── multimodal_fusion.ipynb   # mBERT + EfficientNet (best model)
├── data/
│   └── README.md                 # Dataset access instructions (task organizers)
└── README.md
```

## Dependencies
```
transformers
torch
torchvision
indic-transliteration
deep-translator
scikit-learn
Pillow
```

## Citation
```
@inproceedings{mallik2025cuet823,
  title     = {CUET-823@DravidianLangTech 2025: Shared Task on Multimodal Misogyny Meme Detection in Tamil Language},
  author    = {Mallik, Arpita and Dhar, Ratnajit and Das, Udoy and Labib, Momtazul Arefin and Rahman, Samia and Murad, Hasan},
  booktitle = {Proceedings of the Fifth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages},
  year      = {2025}
}
```

# Forgotten Polygons: Multimodal Large Language Models are Shape-Blind

## Description
Despite strong performance on vision-language tasks, Multimodal Large Language Models (MLLMs) struggle with mathematical problem-solving, with both open-source and state-of-the-art models falling short of human performance on visual-math benchmarks. To systematically examine visual-mathematical reasoning in MLLMs, we (1) evaluate their understanding of geometric primitives, (2) test multi-step reasoning, and (3) explore an initial solution to improve visual reasoning capabilities. Our findings reveal fundamental shortcomings in shape recognition, with top models achieving under 50% accuracy in identifying regular polygons. We analyze these failures through the lens of dual-process theory and show that MLLMs rely on System 1 (intuitive, memorized associations) rather than System 2 (deliberate reasoning). Consequently, MLLMs fail to count the sides of both familiar and novel shapes, suggesting they have neither learned the concept of ``sides'' nor effectively process visual inputs. Finally, we propose Visually Cued Chain-of-Thought (VC-CoT) prompting, which enhances multi-step mathematical reasoning by explicitly referencing visual annotations in diagrams, boosting GPT-4o’s accuracy on an irregular polygon side-counting task from 7% to 93%. Our findings suggest that System 2 reasoning in MLLMs remains an open problem, and visually-guided prompting is essential for successfully engaging visual reasoning.

![CoT_diagram](https://github.com/user-attachments/assets/3c75dc51-f2c3-4fc7-ad8d-906d6f2c1866)


## Requirements
Python 3.9.16

PyTorch Version: 2.2.1

transformers: 4.48.3

To install requirements:

```setup
pip install -r requirements.txt
```

---

## 1. Image Generation (Optional)
📂 **Folder:** `image_generation_code`  
This folder contains code for generating various types of shape datasets, including:
- **Regular polygons**
- **Abstract shapes**
- **Shapes annotated for our method (VC-CoT)**
- **Preprocessing for the MathVerse experiment**

🔹 **Do I need to run this?**  
No, you **do not need to generate images**, except for the MathVerse preprocessing.  
All images used in our experiments are in the **zipped** images folder. However, the code is available if you wish to generate additional images.

---

## 2. Evaluating MLLMs
📂 **Folder:** `evaluation`  
This folder contains code to evaluate **13 different models** on various tasks.

### Running an Evaluation  
We provide all necessary **images (zipped)** and **dataframes (CSVs)** in the `CSVs_for_evaluation` folder. Make sure to download them and move all data into your directory before running evaluation.  
To run a **shape identification task** using **LLaVA-1.5**, execute the following command inside the `evaluation` folder:

```bash
python3 evaluate_MLLMs.py --model_version llava-1.5 --task shape_id --dataset_size full
```
## Tasks

The tasks in evaluate_MLLMs.py are as follows:

- **`shape_id`**: Uses `all_shapes.csv` and asks:  
  *"What shape is in the image?"*

- **`sides_id`**: Also uses `all_shapes.csv` and asks:  
  *"How many sides does the shape in the image have?"*

- **`two_shapes`**: Uses `two_shapes.csv` as input. A multi-step reasoning task requiring:
  1. Identifying the two shapes.
  2. Mapping each shape to its number of sides.
  3. Summing the total number of sides.  

- **`abstract`**: Uses `abstract_shapes.csv` and asks:  
  *"How many sides does this shape have?"*  
  This includes abstract shapes such as merged polygons, irregular polygons, and common shapes like stars and arrows.

- **`triangle_cross_ABC_123`** & **`hept_ABC_123`**:  
  These tasks evaluate **Visually-Cued Chain-of-Thought (VC-CoT)** prompting using different types of prompts.  
  - `triangle_cross_ABC_123` uses `triangle_on_cross_ABC_123.csv`.  
  - `hept_ABC_123` uses `heptagons_ABC_123.csv`.  

- **`mathverse_CoT`**: Uses `mathverse_revised.csv` and evaluates the **vision-dominant split** of the MathVerse dataset.  
  It compares **VC-CoT** against **direct prompting** and **MathVerse's CoT prompting**.

Each task is designed to test different aspects of shape understanding, multi-step reasoning, and the effectiveness of visual cues for MLLMs.

## 3. Accuracy Calculation Notebook  
📂 **File:** `evaluation/MLLMs_accuracy_calculations.ipynb`  

This Jupyter notebook compiles all metrics used in the paper. It provides details on:
- How to **preprocess the results** of each task.
- Separating tasks using **"#######"** for clarity.

The notebook is structured to help analyze model performance across different tasks.

---

## 4. Visualization Code  
📂 **Folder:** `visualization`  

This folder contains code for generating key visualizations from our study, including:
- **T-SNE plots (`get_vision_embeddings.ipynb`)**
- **Nearest neighbors experiment (`get_vision_embeddings.ipynb`)**
- **Google N-grams figure seen in the Appendix (`google_N-grams.ipynb`)**

To generate these visualizations, navigate to the `visualization` folder.

---
## Authors

This repository was created by:

- **[Michal Golovanevsky]** ([GitHub](https://github.com/michalg04))
- **[William Rudman]** ([GitHub](https://github.com/wrudman))
- **[Vedant Palit]** ([GitHub](https://github.com/vedantpalit))

Feel free to reach out or open an issue.

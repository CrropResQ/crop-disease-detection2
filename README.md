
\# CropResQ – Vision Transformer Models



\## Objective



Run and compare 7 modern vision models on the CropResQ dataset using the same experimental setup.



\## Models



| Member    | Models                   | Framework   |

| --------- | ------------------------ | ----------- |

| Vaishnav  | ViT, DINOv2, CoAtNet     | \*\*PyTorch\*\* |

| Mohammad  | Swin Transformer, MaxViT | \*\*PyTorch\*\* |

| Varun     | DeiT                     | \*\*PyTorch\*\* |

| Prathvish | SigLIP                   | \*\*PyTorch\*\* |



\## Approach



\* Use \*\*PyTorch for all models\*\*.

\* Use the \*\*same dataset and train/validation/test split\*\* as the previous experiments.

\* Use the \*\*same hyperparameters\*\*; no new hyperparameter tuning.

\* Use pretrained models wherever available.

\* Use model-specific preprocessing when required.

\* Use mixed precision if GPU supports it.



\## Evaluation



For every model, record:



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* Confusion Matrix

\* Training Time



\## Final Comparison



After all 7 models are completed:



1\. Compare their results.

2\. Select the best-performing models.

3\. Test \*\*2-model combinations\*\*.

4\. Test \*\*3-model combinations\*\*.

5\. Select the best hybrid model for CropResQ.



\### Basic Workflow



```text

Dataset

&#x20;  ↓

7 Individual Models

&#x20;  ↓

Evaluate

&#x20;  ↓

Compare Results

&#x20;  ↓

Select Best Models

&#x20;  ↓

2-Model + 3-Model Hybrid

&#x20;  ↓

Final Model

```




### Cell-by-Cell Documentation for **Quantized Mixtral** Notebook

---

### 1. **Introduction (Markdown Cell)**:
   - **Content**: 
     > "In this code we are using Mixtral with LORA on A100 GPU. I was still not able to train the model due to memory constraints. However, this code showcases my understanding and implementation of a quantized model along with LORA."
   - **Purpose**: This introductory markdown explains that the notebook attempts to train a Mixtral model with LORA on an A100 GPU. However, due to memory limitations, the model couldn't be fully trained, but the notebook highlights the user's approach to handling quantized models.

---

### 2. **Library Installation (Code Cell)**:
   - **Command**: 
     ```bash
     !pip install -U -q bitsandbytes accelerate transformers peft datasets
     ```
   - **Purpose**: Installs key libraries required for model quantization and training. 
     - `bitsandbytes`: Used for quantized training.
     - `accelerate`: For distributed GPU training.
     - `transformers`: Hugging Face's core library for loading models.
     - `peft`: Parameter-efficient fine-tuning with Hugging Face.
     - `datasets`: Handles large datasets efficiently.

---

### 3. **Torch Installation (Code Cell)**:
   - **Command**: 
     ```bash
     !pip install torch
     ```
   - **Purpose**: Ensures that PyTorch (a deep learning framework) is installed. This is a necessary dependency for training models using the Hugging Face library.

---

### 4. **CUDA Configuration (Code Cell)**:
   - **Command**: 
     ```bash
     !export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```
   - **Purpose**: Configures PyTorch’s CUDA allocation to use expandable memory segments. This setting optimizes memory management for models running on the GPU, allowing for better handling of large models and batches during training.

---

### 5. **Model Selection (Code Cell)**:
   - **Command**: 
     ```python
     TARGET_MODEL = "mistralai/Mistral-7B-v0.1"
     ```
   - **Purpose**: Specifies the target model for training. In this case, the model is "Mistral-7B-v0.1", a large-scale model from the "mistralai" repository.

---


### 6. **Setting Hugging Face Token (Code Cell)**:
   - **Command**: 
     ```python
     import os
     os.environ['HUGGINGFACE_TOKEN'] = 'huggingface-api-key'
     ```
   - **Purpose**: Sets the Hugging Face authentication token as an environment variable. This token allows the notebook to access models and datasets from the Hugging Face Hub.

---

### 7. **Dataset Installation and Requirements Check (Code Cell)**:
   - **Purpose**: This cell checks the availability of required libraries (`datasets`, `pandas`, etc.) by printing the installed versions. 
   - **Output**: The cell confirms that all required libraries, such as `datasets`, `numpy`, `pandas`, etc., are already installed in the environment.

---

### 8. **Loading Dataset and Data Conversion to Pandas (Code Cell)**:
   - **Command**:
     ```python
     train = dataset['train'].to_pandas().reset_index(drop=True)
     validation = dataset['validation'].to_pandas().reset_index(drop=True)
     test = dataset['test'].to_pandas().reset_index(drop=True)
     ```
   - **Purpose**: This step loads the dataset (probably the meta-review dataset) from Hugging Face and converts it into Pandas DataFrames for easier manipulation.
     - The index is reset to ensure no extra columns interfere during further preprocessing and tokenization steps.
   - **Preview**: The command `print(train.head())` and similar ones preview the first few rows of each dataset (train, validation, and test) to ensure the structure is correct after loading.

---



### 9. **Text Preprocessing and Stopword Removal (Code Cell)**:
   - **Command**:
     ```python
     import re
     import nltk
     from nltk.corpus import stopwords
     from transformers import AutoTokenizer
     nltk.download('stopwords')
     stop_words = set(stopwords.words('english'))
     ```
   - **Purpose**: 
     - This code imports necessary libraries and downloads the stopwords dataset from NLTK.
     - A function `is_latex_symbol` is defined to identify and preserve LaTeX symbols during text processing.
     - The function `preprocess_text` is responsible for cleaning and preprocessing input/output text while maintaining LaTeX symbols, removing stopwords, and handling special characters.
   - **Why**: Preprocessing the input/output text ensures that the model receives clean data. Tokenizing and normalizing the text helps the model perform better, while preserving LaTeX symbols is critical for academic content.

---

### 10. **Preprocessing the Test Data (Code Cell)**:
   - **Command**:
     ```python
     test_cleaned = test.dropna(subset=['Input', 'Output'])
     test_cleaned.loc[:, 'input_tokenized'] = test_cleaned['Input'].apply(preprocess_text)
     test_cleaned.loc[:, 'output_tokenized'] = test_cleaned['Output'].apply(preprocess_text)
     ```
   - **Purpose**: 
     - This code cleans the test dataset by dropping rows with missing `Input` or `Output` fields.
     - It then applies the `preprocess_text` function to both the input and output columns, resulting in cleaned and tokenized text stored as `input_tokenized` and `output_tokenized`.
   - **Why**: Cleaning the dataset by removing missing values and tokenizing the text ensures that the data is ready for training and evaluation without errors.

---

### 11. **Memory Management and Cleanup (Code Cell)**:
   - **Command**:
     ```python
     import gc
     gc.collect()
     ```
   - **Purpose**: This command triggers garbage collection to free up unused memory. It's important when dealing with large datasets and models to ensure optimal memory usage.
   - **Why**: Memory optimization is critical in training large models, especially when using limited GPU resources. Regular garbage collection helps prevent memory leaks.


---

### 12. **Tokenizer and Dataset Setup (Code Cell)**:
   - **Command**:
     ```python
     from transformers import AutoTokenizer
     from datasets import Dataset

     tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_auth_token=True)
     ```
   - **Purpose**: 
     - The tokenizer from the `TARGET_MODEL` is loaded using `AutoTokenizer`. This tokenizer will be used to process input text into token IDs that the model can understand.
     - The `use_auth_token=True` option authenticates the user to access the model from Hugging Face's Hub.
   - **Why**: Tokenization is essential for converting raw text into numerical representations (tokens) for model training and evaluation.

---

### 13. **Loading the Model with 4-bit Quantization (Code Cell)**:
   - **Command**:
     ```python
     from peft import LoraConfig, TaskType
     from transformers import BitsAndBytesConfig, LlamaForCausalLM
     import torch
     
     peft_config = LoraConfig(
         r=2, lora_alpha=16, lora_dropout=0.1, bias="none",
         task_type=TaskType.CAUSAL_LM, inference_mode=False,
         target_modules=["q_proj", "v_proj"]
     )
     
     bnb_config = BitsAndBytesConfig(
         load_in_4bit=True, bnb_4bit_quant_type="nf4",
         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
     )
     ```
   - **Purpose**: 
     - Configures LORA (Low-Rank Adaptation) and BitsAndBytes (bnb) to handle 4-bit quantization during model loading. 
     - The `LoraConfig` specifies parameters for the fine-tuning technique to make training more efficient. 
     - The `BitsAndBytesConfig` enables loading the model in 4-bit precision to reduce memory usage.
   - **Why**: Quantizing the model to 4-bit precision is a crucial step to fit large models, such as Mistral-7B, into limited GPU memory during training and inference.

---

### 14. **Loading the Quantized Model (Code Cell)**:
   - **Command**:
     ```python
     base_model = LlamaForCausalLM.from_pretrained(
         TARGET_MODEL, num_labels=2, quantization_config=bnb_config, 
         device_map="auto", use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
     )
     base_model.config.pretraining_tp = 1 # 1 is 7b
     base_model.config.pad_token_id = tokenizer.pad_token_id
     ```
   - **Purpose**: 
     - Loads the base Mistral model (`TARGET_MODEL`) using 4-bit quantization and distributes it across available devices (`device_map="auto"`).
     - The model is instantiated as a causal language model (`LlamaForCausalLM`) with two labels.
     - Authentication is handled via Hugging Face token.
   - **Why**: Loading the model with quantization and distributing it across devices ensures that memory usage is optimized, enabling the use of larger models.

---

### 15. **Applying PEFT (Parameter-Efficient Fine-Tuning) to the Model (Code Cell)**:
   - **Command**:
     ```python
     model = get_peft_model(base_model, peft_config)
     ```
   - **Purpose**: 
     - Applies the parameter-efficient fine-tuning (PEFT) configuration to the base model using the LORA technique. 
     - PEFT allows only a small subset of parameters to be fine-tuned, reducing memory requirements and speeding up training.
   - **Why**: PEFT is a popular technique for training large language models on limited hardware resources by fine-tuning fewer parameters.

---

### 16. **Model Structure (Printing the Model) (Code Cell)**:
   - **Command**:
     ```python
     print(model)
     ```
   - **Purpose**: 
     - Prints the entire structure of the quantized model after applying PEFT (Parameter-Efficient Fine-Tuning) and LORA configurations.
     - The printed model includes various layers with specific settings such as 4-bit quantization (`Linear4bit`), attention layers, and normalization layers.
   - **Why**: Printing the model helps in visualizing the entire architecture and verifying that all configurations (like 4-bit quantization and LORA) have been successfully applied.

---

### 17. **Data Collator Setup (Code Cell)**:
   - **Command**:
     ```python
     from transformers import DataCollatorForLanguageModeling

     data_collator = DataCollatorForLanguageModeling(
         tokenizer=tokenizer, 
         mlm=False,  # For causal language modeling, set `mlm=False`
         pad_to_multiple_of=8  # Align tensor sizes
     )
     ```
   - **Purpose**:
     - The `DataCollatorForLanguageModeling` prepares batches of data for causal language modeling.
     - `mlm=False` ensures that the model uses a standard causal language modeling approach rather than masked language modeling.
     - Padding is applied to ensure that tensor sizes are multiples of 8, which improves efficiency on certain hardware architectures.
   - **Why**: A data collator is essential for preparing batches of input data for model training, ensuring that data is properly padded and batched.


---

### 18. **Installing scikit-learn (Code Cell)**:
   - **Command**:
     ```bash
     !pip install scikit-learn
     ```
   - **Purpose**: 
     - Installs the `scikit-learn` library, which provides tools for machine learning tasks such as metrics computation (e.g., accuracy, ROC-AUC). This library is crucial for calculating evaluation metrics during the model's training and testing phases.
   - **Why**: Metrics like accuracy and ROC-AUC are essential for evaluating classification performance in language models.

---

### 19. **Defining Metrics (Code Cell)**:
   - **Command**:
     ```python
     import numpy as np
     from sklearn.metrics import accuracy_score, roc_auc_score
     
     def compute_metrics(eval_pred):
         predictions, labels = eval_pred
         predictions = np.argmax(predictions, axis=1)
         
         accuracy_val = accuracy_score(labels, predictions)
         roc_auc_val = roc_auc_score(labels, predictions)
         return {
             'accuracy': accuracy_val,
             'roc_auc': roc_auc_val
         }
     ```
   - **Purpose**: 
     - This function calculates evaluation metrics such as accuracy and ROC-AUC for the model's predictions.
     - `accuracy_score` measures the proportion of correctly predicted labels.
     - `roc_auc_score` evaluates how well the model discriminates between classes.
   - **Why**: These metrics provide quantitative insights into the model's performance, particularly for classification tasks in language modeling.

---

### 20. **Installing the `trl` Library (Code Cell)**:
   - **Command**:
     ```bash
     !pip install trl
     ```
   - **Purpose**: 
     - Installs the `trl` (Transformers Reinforcement Learning) library. This package enables reinforcement learning with Hugging Face models and is particularly useful for fine-tuning tasks involving reward-based optimization.
   - **Why**: The `trl` library is needed for advanced fine-tuning techniques, particularly when exploring reinforcement learning for language models.


---

### 21. **PYTORCH CUDA Configuration (Code Cell)**:
   - **Command**:
     ```bash
     !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```
   - **Purpose**: 
     - This command sets the `PYTORCH_CUDA_ALLOC_CONF` environment variable to `expandable_segments:True`. This setting helps avoid memory fragmentation during CUDA operations by allocating memory in expandable segments.
   - **Why**: When training large models on GPUs, memory fragmentation can lead to `OutOfMemoryError`. Setting this configuration helps mitigate the issue by better managing memory allocation.

---

### 22. **Training the Model (Code Cell)**:
   - **Command**:
     ```python
     from transformers import TrainingArguments, Trainer

     training_args = TrainingArguments(
         output_dir=OUTPUT_DIR,
         learning_rate=5e-5,
         per_device_train_batch_size=1,
         per_device_eval_batch_size=1,
         gradient_accumulation_steps=16,
         max_grad_norm=0.3,
         optim='paged_adamw_32bit',
         lr_scheduler_type="cosine",
         num_train_epochs=1,
         weight_decay=0.01,
         evaluation_strategy="steps",
         save_strategy="steps",
         load_best_model_at_end=True,
         warmup_steps=5,  # Adjusted for this example
         eval_steps=5,
         logging_steps=5,
         report_to='none',
         dataloader_num_workers=2,
     )

     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=train_dataset,
         eval_dataset=validation_dataset,
         tokenizer=tokenizer,
         data_collator=data_collator,
         compute_metrics=compute_metrics,
     )

     trainer.train()
     ```
   - **Purpose**: 
     - This section initializes training with the specified hyperparameters using Hugging Face's `Trainer` API. Key arguments include learning rate, batch sizes, gradient accumulation steps, and the optimizer configuration.
     - The training is run with quantized models using the `paged_adamw_32bit` optimizer, which is more memory-efficient.
   - **Why**: Using the `Trainer` API simplifies the training process by handling backpropagation, weight updates, evaluation, and logging, making it easier to fine-tune large language models.

---

### 23. **Memory Error During Training (Markdown Cell)**:
   - **Content**:
     > "As we can see we ran out of memory during training."
   - **Purpose**: 
     - This markdown cell comments on the memory issues encountered during training, specifically a CUDA out-of-memory error.
   - **Why**: Memory constraints are a common issue when fine-tuning large models, even with optimizations like 4-bit quantization. This error highlights the need for careful memory management when training models on limited hardware.

---


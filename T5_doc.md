
### Documentation for the "t5-trained" notebook

---

**Overview:**

This notebook was developed and executed on Kaggle, using two T4 GPUs. The primary goal is to fine-tune the T5 model (or another Hugging Face model) with a dataset of reviews while addressing GPU memory constraints by truncating input text during the preprocessing phase. This ensures that the training process remains within the GPU memory limits.

---

### Key Sections and Code Explanation:

1. **Environment Setup**:
   - **Libraries Installed**:
     ```python
     !pip install -U -q bitsandbytes accelerate transformers
     ```
     - **Purpose**: Install key libraries.
     - **Libraries**:
       - `bitsandbytes`: A memory-efficient optimizer for model training, especially useful for low-bit quantized models.
       - `accelerate`: Hugging Face’s library for distributed training across multiple GPUs.
       - `transformers`: The core library from Hugging Face for working with models like T5.

2. **Environment Variable Setup**:
   - Hugging Face Token Setup:
     ```python
     os.environ['HUGGINGFACE_TOKEN'] = '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'
     ```
     - **Purpose**: Sets the Hugging Face token as an environment variable to authenticate and download models or datasets from the Hugging Face hub.

3. **Loading Dataset**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("zqz979/meta-review")
   ```
   - **Library**: `datasets` is part of Hugging Face for handling large datasets efficiently.
   - **Dataset**: "zqz979/meta-review" dataset is loaded, which contains academic meta-review data used for model training.
   - **Why**: This dataset is being used for fine-tuning a T5 model to generate summaries or process review-based tasks.

4. **Data Preprocessing**:
   - **Pandas for Data Handling**:
     ```python
     #Since data is in Dictionary format. I am converting it to pandas for easy implementation.
     ```
     - The dataset is converted to a Pandas DataFrame for easier manipulation, given its tabular nature.

---

### 5. **Data Conversion to Pandas**:
   The dataset is originally in a dictionary format, so it's converted into Pandas DataFrames for easier manipulation:
   ```python
   train = dataset['train'].to_pandas()
   validation = dataset['validation'].to_pandas()
   test = dataset['test'].to_pandas()
   ```
   - **Why**: Pandas provides a more flexible framework for data manipulation, making it easier to clean and preprocess the dataset before training.

---

### 6. **Preprocessing the Data**:
   - **Preprocessing Steps**: 
     ```python
     test_cleaned = test.dropna(subset=['Input', 'Output'])
     test_cleaned.loc[:, 'input_tokenized'] = test_cleaned['Input'].apply(preprocess_text)
     test_cleaned.loc[:, 'output_tokenized'] = test_cleaned['Output'].apply(preprocess_text)
     ```
     - **Dropping NA Values**: Removes rows where the 'Input' or 'Output' columns have missing values.
     - **Tokenization**: The `preprocess_text` function tokenizes both the 'Input' and 'Output' text fields. Tokenization is essential for converting raw text into a form that the model can understand (i.e., turning it into tokens).
     - **Removing Empty Tokens**: 
       ```python
       test_cleaned = test_cleaned[test_cleaned['input_tokenized'].str.strip() != '']
       test_cleaned = test_cleaned[test_cleaned['output_tokenized'].str.strip() != '']
       ```
       Ensures that no empty tokens remain after tokenization.

---

### 7. **Garbage Collection**:
   ```python
   import gc
   gc.collect()
   ```
   - **Purpose**: Manually triggering garbage collection to free up memory, which is essential when working with limited GPU memory.

---

### 8. **Model Selection**:
   - **Choosing T5-Small**:
     ```markdown
     I decided to use t5 small due to the memory constraints. The t5 model is a small text-to-text model which also supports latex.
     ```
     - **T5-Small**: A smaller version of the T5 model is chosen due to the memory limitations. T5 (Text-to-Text Transfer Transformer) is a versatile model that can handle multiple tasks by converting all problems into text-to-text formats. Additionally, T5 supports LaTeX, which is useful in this notebook's use case of handling academic meta-reviews.
     - **Why**: T5-Small has fewer parameters than its larger counterparts, which helps reduce the memory footprint while still providing good performance.

---

### Summary of Libraries Used:

- **Pandas**: Used for data manipulation, converting the dataset into DataFrames, and handling missing data or preprocessing.
- **Hugging Face Transformers (`transformers`)**: Provides the model architecture (T5 in this case) and tokenization tools.
- **Garbage Collection (`gc`)**: Helps in managing memory usage by manually collecting unreferenced objects.

---

### 9. **Tokenization**:
   - **Tokenization Function**:
     ```python
     def tokenize_function(row):
         # Tokenize the input (meta-review) and output (summary)
         model_inputs = tokenizer(row['input_tokenized'], max_length=1052, truncation=True, padding='max_length')
         
         # Tokenize output/labels (the summary)
         with tokenizer.as_target_tokenizer():
             labels = tokenizer(row['output_tokenized'], max_length=1052, truncation=True, padding='max_length')
         
         # Return only required fields for training
         model_inputs['labels'] = labels['input_ids']
         return {
             'input_ids': model_inputs['input_ids'],
             'attention_mask': model_inputs['attention_mask'],
             'labels': model_inputs['labels']
         }
     ```
     - **What it does**: This function tokenizes the input and output texts. It ensures that the text is truncated to a fixed length (`max_length=1052`) to handle memory constraints and that the input and target text is properly converted to token IDs.
     - **Why**: Tokenization is necessary to convert the textual data into numerical representations that can be processed by the T5 model.

   - **Applying Tokenization**:
     ```python
     train_tokenized = train_cleaned.apply(tokenize_function, axis=1, result_type='expand')
     validation_tokenized = validation_cleaned.apply(tokenize_function, axis=1, result_type='expand')
     test_tokenized = test_cleaned.apply(tokenize_function, axis=1, result_type='expand')
     ```
     - The `tokenize_function` is applied to the training, validation, and test sets to tokenize all the input/output pairs for model training and evaluation.

   - **Converting to Hugging Face Dataset**:
     ```python
     train_dataset = Dataset.from_pandas(train_tokenized.reset_index(drop=True))
     validation_dataset = Dataset.from_pandas(validation_tokenized.reset_index(drop=True))
     test_dataset = Dataset.from_pandas(test_tokenized.reset_index(drop=True))
     ```
     - After tokenizing, the DataFrames are converted back into a Hugging Face `Dataset` format, which is optimized for model training.

---

### 10. **Model and Training Setup**:
   - **Loading the T5 Model**:
     ```python
     from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
     model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", device_map='auto')
     ```
     - The `T5ForConditionalGeneration` or `AutoModelForSeq2SeqLM` from Hugging Face is loaded with the pre-trained weights for the `t5-small` model. The `device_map='auto'` ensures that the model is mapped to available GPU(s).

   - **Setting Training Arguments**:
     ```python
     training_args = TrainingArguments(
         output_dir="./results",            # output directory for checkpoints
         evaluation_strategy="epoch",       # evaluate after each epoch
         per_device_train_batch_size=8,     # batch size for training
         per_device_eval_batch_size=8,      # batch size for evaluation
         num_train_epochs=11,               # number of epochs for training
         weight_decay=0.01,                 # weight decay to prevent overfitting
         logging_dir="./logs",              # logging directory
         remove_unused_columns=False,       # retain all input features
         learning_rate=5e-5,                # initial learning rate
         lr_scheduler_type="cosine"         # learning rate scheduler type
     )
     ```
     - **Purpose of Each Argument**:
       - `output_dir`: Directory where model checkpoints and outputs will be saved.
       - `evaluation_strategy`: Evaluates the model after each epoch to monitor performance.
       - `per_device_train_batch_size` / `per_device_eval_batch_size`: Specifies the batch size for training and evaluation.
       - `num_train_epochs`: Specifies how many times the model will iterate over the entire dataset.
       - `weight_decay`: Helps in regularization to avoid overfitting.
       - `learning_rate`: The initial learning rate for optimization.
       - `lr_scheduler_type`: Adjusts the learning rate dynamically using a cosine decay.

   - **Initializing the Trainer**:
     ```python
     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=train_dataset,
         eval_dataset=validation_dataset
     )
     ```
     - **Purpose**: The `Trainer` is initialized with the model, training arguments, and datasets (train and validation). This provides an efficient way to handle the training loop, evaluation, and checkpointing.

   - **Training the Model**:
     ```python
     trainer.train()
     ```
     - **What it does**: Begins the training process using the provided training dataset and arguments. The training loop handles backpropagation, weight updates, evaluation, and logging automatically.

---

### 11. **Saving the Model**:
   ```markdown
   Saved the model for future use.
   ```
   - After training, the model and its checkpoints are saved in the specified `output_dir` for future inference or fine-tuning.

---

### Summary of Important Libraries:

- **Transformers (`transformers`)**: Core Hugging Face library used for loading pre-trained models (like T5), tokenization, and the training loop.
- **Pandas (`pandas`)**: Used for preprocessing and handling data as DataFrames.
- **Datasets (`datasets`)**: Hugging Face library to handle large datasets efficiently.
- **Trainer**: Hugging Face's `Trainer` API is used to streamline the training, evaluation, and logging process.

---


### 12. **Evaluation Metrics - ROUGE Score Installation**:
   ```python
   !pip install rouge-score
   ```
   - **Purpose**: The `rouge-score` package is installed for evaluating the model's performance. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of summaries by comparing the overlap of n-grams between the generated summary and the reference summary.

   - **Why ROUGE?**: ROUGE is commonly used for evaluating tasks such as summarization, where the precision, recall, and F1-score of overlapping n-grams (words, sequences of words) give insight into the quality of the summary.

---

### 13. **Test Tokenization for Sentence**:
   ```python
   test_senten_tokenize = test_cleaned['input_tokenized'][0]
   test_senten_tokenize_output = test_cleaned['Output'][0]
   ```
   - **Purpose**: This step tests tokenization on a single sentence from the test dataset. The first input and output examples are extracted to check how they have been tokenized.

   - **Why**: This kind of test is helpful for debugging or verifying that the tokenization process is working as expected before proceeding to the model's evaluation and inference steps.

---

### 14. **Displaying Tokenized Output**:
   ```python
   test_senten_tokenize_output
   ```
   - **Output**: 
     ```
     'The paper presents results using syntactic information (primarily through constituency trees) on the task of recognizing argument discourse units. No reviewer recommends acceptance of the paper: - The empirical results appear strong, though the reviewers raise questions about some of the experimental choices.  - The writing is unclear and reviewers point out many missing or incorrect references in the bibliography. - There is little methodological novelty - known techniques are applied to a topic that has not been studied much. Overall, the area chair agrees with the reviewers that this work does not yet meet the bar for ICLR.'
     ```
   - **Purpose**: Displays the tokenized output (summary) of the test sample. This output is used for validation and debugging purposes to ensure that the preprocessed data matches expectations before being passed to the model.

---

### 15. **Model Evaluation (ROUGE Scores)**:
   While the direct evaluation code is not included in the examined cells, the purpose of this section would be to use the ROUGE score installed earlier to evaluate the model's summarization quality. Generally, the evaluation would involve:
   
   - **Generating Summaries**: Running the model on test data to generate predictions.
   - **Computing ROUGE Scores**: Comparing the generated summaries with reference summaries using ROUGE metrics for recall, precision, and F1-score.

---

### Summary of Additional Libraries:

- **ROUGE-Score**: Installed and used for evaluating the model’s performance on summarization tasks. ROUGE metrics are well-suited for this task because they focus on the overlap of words and sequences of words between the generated and reference summaries.

---

### 16. **Testing Sentence Tokenization**:
   ```python
   test_sentence = test_cleaned['Input'][0]
   combined_input = prompt + test_sentence
   ```
   - **Purpose**: This step selects a test sentence from the cleaned test data and combines it with a prompt. This is likely part of the process to check how the model performs when given an input sentence.

   - **Why**: Testing individual components like sentence tokenization helps ensure that the input to the model is formatted correctly before running the full inference or evaluation.

---

### 17. **Model Device Allocation**:
   ```python
   device = next(model.parameters()).device
   ```
   - **Purpose**: Retrieves the device (CPU or GPU) where the model is currently loaded. This step is useful for verifying whether the model is correctly assigned to the GPU for faster computation.

   - **Why**: Ensuring that the model runs on the correct device is essential for efficient training and inference, especially when using large models like T5.

---

### 18. **Testing the Model**:
   ```markdown
   # Testing the model to see how it is working
   ```
   - **Purpose**: This section begins the testing of the fine-tuned model on the test dataset to evaluate its performance.

   - The actual code for testing the model involves:
     - **Generating Predictions**: Running the model on the test input to generate summaries.
     - **Comparing Results**: Comparing the generated output with the ground truth (expected summaries).

---

### 19. **Displaying Tokenized Inputs and Outputs**:
   ```python
   test_cleaned
   ```
   - **Purpose**: Displays the cleaned test dataset, including tokenized inputs and outputs, for inspection. The tokenized data will be used as input for the model testing and evaluation.

   - **Output**: 
     This DataFrame contains 1,645 rows of tokenized input and output columns. These are the inputs (e.g., meta-reviews) and expected outputs (summaries) that will be fed into the model for evaluation.

---

### Summary of Key Concepts Covered:

- **Tokenization Testing**: Ensures that input and output sentences are correctly preprocessed before being passed to the model.
- **Device Assignment**: Confirms that the model is using the available GPU for faster execution.
- **Model Testing**: Verifies that the model can handle real test cases after fine-tuning.
- **Data Inspection**: Displays the tokenized data used for evaluating the model's performance.

---

### 20. **Testing the Model with Prompt**:
   - **Explanation**:
     ```markdown
     Testing the model with prompt. The maximum length has been kept at 512 to save memory and runtime.
     ```
   - **What it does**: In this section, the model is tested with a prompt, and the input is truncated to a maximum length of 512 tokens. This helps in reducing memory usage and keeping the inference runtime manageable.
   - **Why**: Limiting the maximum length of the input ensures that the model can process the text without running into memory issues, which is a common constraint when fine-tuning large models like T5 on smaller hardware setups.

---

### 21. **Summary Generation Function**:
   ```python
   def generate_summary(model, tokenizer, input_text, max_length=512):
       inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
       inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input tensors to the correct device
       
       # Generate summary using the model
       summaries = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_length)
       
       # Decode and return the summary
       return tokenizer.decode(summaries[0], skip_special_tokens=True)
   ```
   - **Purpose**: This function generates a summary using the model for a given `input_text`.
     - **Tokenization**: The input text is tokenized and padded/truncated to a maximum of 512 tokens to fit the model's input size.
     - **Device Assignment**: The input tensors are moved to the correct device (likely a GPU).
     - **Summary Generation**: The model generates a summary using its `generate` method.
     - **Decoding**: The generated summary is decoded back into human-readable text.
   
   - **Why**: This function is used to test the model's ability to generate summaries for individual inputs during inference.

---

### 22. **Evaluating the Model**:
   ```python
   def evaluate_model_on_original(model, tokenizer, original_validation_dataset):
       references = []
       predictions = []
       
       for i in range(len(original_validation_dataset)):
           input_text = original_validation_dataset.iloc[i]['input_tokenized']  # Input meta-review
           reference_summary = original_validation_dataset.iloc[i]['output_tokenized']  # Ground truth summary
           
           # Generate summary from the model
           predicted_summary = generate_summary(model, tokenizer, input_text)
           
           # Store the reference and prediction
           references.append(reference_summary)
           predictions.append(predicted_summary)
       
       # Compute ROUGE scores
       results = rouge_metric.compute(predictions=predictions, references=references)
       return results
   ```
   - **Purpose**: This function evaluates the model's performance on the original validation dataset by comparing the predicted summaries with the ground truth summaries.
     - **Reference and Predictions**: It loops through the validation dataset, generating predictions and comparing them with the ground truth.
     - **ROUGE Score Calculation**: It calculates the ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for the predictions and references.

   - **Why**: This is the core evaluation loop for checking how well the model performs on unseen validation data by using ROUGE as the evaluation metric.

---

### 23. **Example Usage and Results**:
   ```python
   # Example usage:
   results = evaluate_model_on_original(model, tokenizer, test_cleaned)
   
   # Display the ROUGE scores
   print(f"ROUGE-1: {results['rouge1']}")
   print(f"ROUGE-2: {results['rouge2']}")
   print(f"ROUGE-L: {results['rougeL']}")
   ```
   - **Purpose**: Demonstrates how to run the evaluation function on the test data and print out the ROUGE scores.
   - **Why**: Displaying the ROUGE scores gives a quantifiable measure of the model’s performance in terms of summarization quality.

---

### Summary of Key Concepts:

- **Summary Generation**: The `generate_summary` function demonstrates how to use the model to generate summaries for given input text, keeping the input size limited to avoid memory issues.
- **Model Evaluation**: The evaluation loop compares model-generated summaries with ground-truth summaries using ROUGE metrics.
- **ROUGE Scores**: These scores are printed out to provide feedback on the model's summarization quality based on the overlap of words and sequences.

---


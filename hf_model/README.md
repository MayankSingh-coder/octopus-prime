# Multi-Layer Perceptron Language Model

This is a language model trained using a multi-layer perceptron architecture.

## Model Details

- **Model Type**: Multi-Layer Perceptron
- **Context Size**: 2
- **Embedding Dimension**: 20
- **Hidden Layers**: [32, 16]
- **Vocabulary Size**: 7
- **Tokenizer Type**: wordpiece
- **Trained**: Yes

## Usage with Hugging Face Transformers

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Load the model
config = AutoConfig.from_pretrained("path/to/model")
model = AutoModel.from_pretrained("path/to/model", config=config)
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Generate text
input_text = "your input text here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Custom Usage

You can also use the model with the custom API:

```python
from neural_network_lm.models.multi_layer_perceptron import MultiLayerPerceptron

# Load the model
model = MultiLayerPerceptron.load_model("path/to/model.pkl")

# Generate text
input_context = "your input text here"
generated_words, info = model.predict_next_n_words(input_context, n=10, temperature=1.0)
print(f"{input_context} {' '.join(generated_words)}")
```

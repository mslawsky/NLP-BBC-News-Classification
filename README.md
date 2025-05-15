# BBC News Text Classification - Data Exploration & Preprocessing 📰

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.1+-purple.svg)](https://pandas.pydata.org/)

A comprehensive tutorial on text preprocessing and data preparation for natural language processing using TensorFlow. This project demonstrates essential NLP techniques including data loading, text standardization, tokenization, and sequence encoding using the BBC News Classification Dataset.

![Text Processing Pipeline](text-processing-pipeline.png)

---

## Table of Contents 📋
- [Project Overview](#project-overview-)
- [Dataset Details](#dataset-details-)
- [Processing Pipeline](#processing-pipeline-)
- [Implementation Steps](#implementation-steps-)
- [Results](#results-)
- [Real-World Applications](#real-world-applications-)
- [Installation & Usage](#installation--usage-)
- [Key Learnings](#key-learnings-)
- [Future Improvements](#future-improvements-)
- [Acknowledgments](#acknowledgments-)
- [Contact](#contact-)

---

## Project Overview 🔎

This project implements the foundational data preprocessing steps required for text classification tasks. Using the BBC News dataset as a practical example, we explore how to transform raw text data into numerical representations suitable for machine learning models.

**Key Objectives:**
- Parse and load text data from CSV files
- Standardize text by removing stopwords and converting to lowercase
- Vectorize text using TensorFlow's TextVectorization layer
- Encode categorical labels for multi-class classification
- Prepare data for neural network training

**Technical Stack:**
- TensorFlow 2.x and Keras for text processing
- Python standard library (csv) for file operations
- NumPy and Pandas for data manipulation
- Python 3.6+ for implementation

---

## Dataset Details 📊

The BBC News Classification Dataset contains news articles across 5 categories:

- **Total Articles**: 2,225 news articles
- **Categories**: 5 (sport, business, politics, tech, entertainment)
- **Format**: CSV file with category and text columns
- **Text Length**: Variable (first article has 737 words)
- **Language**: English

**Data Structure:**
```csv
category,text
tech,"tv future in the hands of viewers with home..."
business,"worldcom boss left books alone former worldcom..."
sport,"tigers wary of farrell gamble leicester say they..."
```

**Key Characteristics:**
- Each row contains a category label and the full article text
- Articles are real BBC news content across diverse topics
- Text includes punctuation, mixed case, and common English stopwords
- Dataset is balanced across categories for effective training

---

## Processing Pipeline 🔄

The text preprocessing pipeline consists of four main steps:

### 1. Data Loading 📂
```python
def parse_data_from_file(filename):
    """Load sentences and labels from CSV"""
    sentences = []
    labels = []
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if row:
                labels.append(row[0])
                sentences.append(row[1])
    
    return sentences, labels
```

### 2. Text Standardization 🔤
```python
def standardize_func(sentence):
    """Convert to lowercase and remove stopwords"""
    # Convert to lowercase
    sentence = sentence.lower()
    
    # Split into words
    words = sentence.split()
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in STOPWORDS]
    
    return " ".join(filtered_words)
```

### 3. Text Vectorization 🔢
```python
def fit_vectorizer(sentences):
    """Create and adapt TextVectorization layer"""
    vectorizer = tf.keras.layers.TextVectorization()
    vectorizer.adapt(sentences)
    return vectorizer
```

### 4. Label Encoding 🏷️
```python
def fit_label_encoder(labels):
    """Create StringLookup for label encoding"""
    label_encoder = tf.keras.layers.StringLookup(num_oov_indices=0)
    label_encoder.adapt(labels)
    return label_encoder
```

---

## Implementation Steps 💻

### Step 1: Data Loading
First, we parse the BBC News CSV file to extract articles and their categories:

```python
# Load the data
sentences, labels = parse_data_from_file("./data/bbc-text.csv")

# Output statistics
print(f"Dataset size: {len(sentences)} articles")
print(f"Categories: {set(labels)}")
print(f"First article length: {len(sentences[0].split())} words")
```

**Output:**
- 2,225 articles loaded successfully
- 5 unique categories identified
- First article contains 737 words

### Step 2: Text Standardization
We clean the text by removing stopwords and converting to lowercase:

```python
# Define stopwords list
STOPWORDS = ["a", "about", "above", "after", "again", "against", ...]

# Standardize all sentences
standard_sentences = [standardize_func(sentence) for sentence in sentences]

# Compare before and after
print(f"Original: {len(sentences[0].split())} words")
print(f"Standardized: {len(standard_sentences[0].split())} words")
```

**Impact:**
- Removes ~40% of words (common stopwords)
- Normalizes text case for consistency
- Reduces vocabulary size significantly

### Step 3: Text Vectorization
Convert text to numerical sequences using TensorFlow:

```python
# Create and fit vectorizer
vectorizer = fit_vectorizer(standard_sentences)

# Get vocabulary statistics
vocabulary = vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocabulary)} unique words")

# Vectorize sentences
sequences = vectorizer(standard_sentences)
print(f"Sequence shape: {sequences.shape}")
```

**Results:**
- Vocabulary of 33,088 unique words
- Each text converted to fixed-length sequence
- Includes [UNK] token for out-of-vocabulary words

### Step 4: Label Encoding
Encode categorical labels as integers:

```python
# Create and fit label encoder
label_encoder = fit_label_encoder(labels)

# Get label mapping
label_vocab = label_encoder.get_vocabulary()
print(f"Label categories: {label_vocab}")

# Encode labels
encoded_labels = label_encoder(labels)
print(f"Encoded labels shape: {encoded_labels.shape}")
```

**Mapping:**
- sport → 0
- business → 1
- politics → 2
- tech → 3
- entertainment → 4

---

## Results 📈

After preprocessing, our data is ready for model training:

| Preprocessing Step | Input | Output |
|-------------------|-------|--------|
| Data Loading | CSV file | 2,225 texts + labels |
| Standardization | Raw text | Clean text (436 words avg) |
| Vectorization | Text strings | Integer sequences |
| Label Encoding | String labels | Integer labels (0-4) |

**Key Achievements:**
- ✅ Successfully loaded and parsed CSV data
- ✅ Reduced vocabulary by removing stopwords
- ✅ Created consistent numerical representations
- ✅ Prepared data for neural network training

---

## Real-World Applications 🌍

This preprocessing pipeline is essential for various NLP applications:

1. **News Categorization**: Automatic article classification for news websites
2. **Content Moderation**: Filtering inappropriate content on social platforms
3. **Email Classification**: Spam detection and email routing
4. **Customer Support**: Ticket categorization and routing
5. **Document Organization**: Automatic filing and tagging systems

**Example Use Case:**
```python
def classify_new_article(text, vectorizer, label_encoder, model):
    """Classify a new article using preprocessing pipeline"""
    # Standardize text
    clean_text = standardize_func(text)
    
    # Vectorize
    sequence = vectorizer([clean_text])
    
    # Predict
    prediction = model.predict(sequence)
    label_idx = np.argmax(prediction)
    
    # Decode label
    categories = label_encoder.get_vocabulary()
    return categories[label_idx]
```

---

## Installation & Usage 🚀

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas (optional)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/bbc-news-text-classification.git

# Navigate to project
cd bbc-news-text-classification

# Install dependencies
pip install tensorflow numpy pandas

# Download dataset
# Available from course materials or Kaggle
```

### Running the Code
```bash
# Run the preprocessing pipeline
python bbc_text_preprocessing.py

# Or use Jupyter notebook
jupyter notebook BBC_News_Text_Processing.ipynb
```

---

## Key Learnings 💡

This project teaches several fundamental NLP concepts:

1. **CSV Data Handling**: Working with real-world text datasets
2. **Text Standardization**: Importance of consistent preprocessing
3. **Stopword Removal**: Reducing noise in text data
4. **Tokenization**: Converting text to numerical format
5. **Sequence Padding**: Creating uniform input sizes
6. **Label Encoding**: Handling categorical targets

**Best Practices:**
- Always examine your data before preprocessing
- Create reusable preprocessing functions
- Validate preprocessing steps with examples
- Consider computational efficiency for large datasets

---

## Future Improvements 🚀

Potential enhancements for this preprocessing pipeline:

1. **Advanced Tokenization**: Use subword tokenization (BPE, WordPiece)
2. **Custom Stopwords**: Domain-specific stopword lists
3. **Stemming/Lemmatization**: Further text normalization
4. **N-grams**: Capture phrase-level information
5. **TF-IDF Weighting**: Alternative to simple tokenization
6. **Data Augmentation**: Synthetic data generation
7. **Multilingual Support**: Extend to other languages

**Next Steps:**
- Build and train a classification model
- Implement model evaluation metrics
- Deploy as a web service
- Add real-time prediction capabilities

---

## Acknowledgments 🙏

- This project is based on the "Explore the BBC News archive" assignment from the ["TensorFlow in Practice" specialization](https://www.coursera.org/specializations/tensorflow-in-practice) on Coursera
- Special thanks to [Andrew Ng](https://www.andrewng.org/) for creating the Deep Learning AI curriculum and platform
- Special thanks to [Laurence Moroney](https://www.linkedin.com/in/laurence-moroney/) for his excellent instruction and for developing the course materials
- The BBC News Classification Dataset was curated for educational purposes
- This notebook was created as part of the "Deep Learning AI TensorFlow Developer Professional Certificate" program

---

## Contact 📫

For inquiries about this project:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© 2025 Melissa Slawsky. All Rights Reserved.

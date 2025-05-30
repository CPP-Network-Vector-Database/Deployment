## FAISS Performance Tracker

### Features

- Multi-Model Support: 
    - Tests 7 different sentence transformer models
- Comprehensive Metrics: 
    - Tracks execution time, CPU usage, and memory consumption
- Multiple Operations: 
    - Benchmarks insertion, deletion, update, and query operations
- Scalable Testing: 
    - Tests operations with varying sizes (2.5K to 30K embeddings)
- Visual Reports: 
    - Generates performance plots for each model
- CSV Export: 
    - Saves detailed results in CSV format for further analysis
- Automated Setup: 
    - Includes shell script for easy environment setup

### Supported Models

1. `paraphrase-MiniLM-L12-v2`
2. `all-MiniLM-L6-v2`
3. `distilbert-base-nli-stsb-mean-tokens`
4. `microsoft/codebert-base`
5. `bert-base-nli-mean-tokens`
6. `sentence-transformers/average_word_embeddings_komninos`
7. `all-mpnet-base-v2`


### Installation

#### Option 1: Automated Setup 

1. Clone the repository:
   ```bash
   git clone https://github.com/CPP-Network-Vector-Database/Deployment
   cd faiss-performance-tracker
   ```

2. Make the shell script executable:
   ```bash
   chmod +x run_tracker.sh
   ```

3. Run the setup and execution script:
   ```bash
   ./run_tracker.sh
   ```

#### Option 2: Manual Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the dataset path in `faiss_performance_tracker.py`:
   ```python
   csv_path = "path/to/your/dataset.csv"
   ```

4. Run the script:
   ```bash
   python3 faiss_performance_tracker.py
   ```

### Dataset Format

The script expects a CSV file with network packet data containing the following columns:
- `frame.number`
- `frame.time`
- `ip.src`
- `ip.dst`
- `tcp.srcport`
- `tcp.dstport`
- `_ws.col.protocol`
- `frame.len`

### Output

#### CSV Files

1. Individual Model Results: `PipelineResults/{model_name}_metrics.csv`
   - Contains performance metrics for each model separately

2. Combined Results: `PipelineResults/all_models_performance_metrics.csv`
   - Contains all models' results in a single file for easy comparison

CSV Structure:
```csv
model_name,operation_type,operation_size,execution_time,cpu_usage,memory_usage
```

#### Performance Plots

Visual charts showing:
- Execution Time: Time taken for each operation type vs. operation size
- CPU Usage: CPU percentage used for each operation
- Memory Usage: Memory consumption in MB

### Configuration

#### Custom Dataset Path

Set the `CSV_PATH` environment variable:
```bash
export CSV_PATH="/path/to/your/dataset.csv"
./run_tracker.sh
```

#### Modifying Test Parameters

Edit `faiss_performance_tracker.py` to change:
- Operation sizes: Modify the `num_ops` list in the main function
- Models tested: Update the `modelList` array
- Query parameters: Change the `k` value for top-k similarity search

### Performance Considerations

- Memory Usage: Each model requires loading embeddings into memory
- Execution Time: Full benchmark can take 30 minutes to several hours depending on hardware
- CPU Intensive: Uses multiple CPU cores for embedding generation and FAISS operations

### Troubleshooting

#### Common Issues

1. Out of Memory Error:
   - Reduce the `num_ops` values
   - Use fewer models in `modelList`
   - Ensure sufficient RAM is available

2. Dataset Not Found:
   - Update `csv_path` in the script
   - Ensure the CSV file exists and has the correct format

3. FAISS Installation Issues:
   - For GPU support: `pip install faiss-gpu`
   - For CPU only: `pip install faiss-cpu`

#### GPU Support

To enable GPU acceleration:
1. Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`
2. Install CUDA-compatible PyTorch version
3. Modify the index creation to use GPU indices
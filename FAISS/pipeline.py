import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

class FAISSPerformanceTracker:
    def __init__(self, model_name = 'all-mpnet-base-v2'): # sentence model to be passed can be changed 
        self.model = SentenceTransformer(model_name) # load the model
        self.model_name = model_name  # Store the model name for saving plots
        
        # Performance tracking lists- to keep track of time, memory, and cpu usage as the number of insertions/deletions/updates change
        self.insertion_sizes = []
        self.deletion_sizes = []
        self.update_sizes = []
        self.query_sizes = []
        
        self.insertion_times = []
        self.deletion_times = []
        self.update_times = []
        self.query_times = []
        
        self.insertion_cpu = []
        self.deletion_cpu = []
        self.update_cpu = []
        self.query_cpu = []
        
        self.insertion_memory = []
        self.deletion_memory = []
        self.update_memory = []
        self.query_memory = []

    def _measure_performance(self, func, *args, **kwargs): # this returns the resultof the function
        pid = os.getpid()
        process = psutil.Process(pid)
        
        # Initial resource usage
        start_cpu = process.cpu_percent(interval=None)
        start_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        start_time = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Final resource usage
        end_cpu = process.cpu_percent(interval=None)
        end_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        end_time = time.time()
        
        # Compute differences
        cpu_usage = end_cpu - start_cpu
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage # same fucntion as was beingused before

    def track_insertion(self, index, new_packet_texts):
        # Normalize embeddings because all sentenceTransformer based models need normalization
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        # Track metrics by appending it to the previously initialized lists
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, index, num_deletions):
        # Select random indices to delete (the number is passed by us as an argument)
        delete_indices = np.random.choice(index.ntotal, num_deletions, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, index, delete_indices
        )
        
        # Track metrics
        self.deletion_sizes.append(num_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)

    def track_update(self, index, num_updates, new_packet_texts):
        # Select random indices to update
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_packet_texts
        )
        
        # Track metrics
        self.update_sizes.append(num_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage) # same as before

    def _insert_embeddings(self, index, new_embeddings):
        index.add(new_embeddings)
        return index

    def _delete_embeddings(self, index, delete_indices):
        index.remove_ids(delete_indices)
        return index

    def _update_embeddings(self, index, update_indices, new_packet_texts):
        index.remove_ids(update_indices)
        
        # Compute and normalize new embeddings
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Add new embeddings
        index.add(new_embeddings)
        return index

    def save_metrics_to_csv(self, save_path=None):
        """Save all performance metrics to a single CSV file in long format"""
        if save_path is None:
            save_path = f"performance_metrics_{self.model_name.replace('/', '_').replace('-', '_')}.csv"
        
        # Create a list to store all records
        all_records = []
        
        # Add insertion records
        for i in range(len(self.insertion_sizes)):
            all_records.append({
                'model_name': self.model_name,
                'operation_type': 'insertion',
                'operation_size': self.insertion_sizes[i],
                'execution_time': self.insertion_times[i],
                'cpu_usage': self.insertion_cpu[i],
                'memory_usage': self.insertion_memory[i]
            })
        
        # Add deletion records
        for i in range(len(self.deletion_sizes)):
            all_records.append({
                'model_name': self.model_name,
                'operation_type': 'deletion',
                'operation_size': self.deletion_sizes[i],
                'execution_time': self.deletion_times[i],
                'cpu_usage': self.deletion_cpu[i],
                'memory_usage': self.deletion_memory[i]
            })
        
        # Add update records
        for i in range(len(self.update_sizes)):
            all_records.append({
                'model_name': self.model_name,
                'operation_type': 'update',
                'operation_size': self.update_sizes[i],
                'execution_time': self.update_times[i],
                'cpu_usage': self.update_cpu[i],
                'memory_usage': self.update_memory[i]
            })
        
        # Add query records
        for i in range(len(self.query_sizes)):
            all_records.append({
                'model_name': self.model_name,
                'operation_type': 'query',
                'operation_size': self.query_sizes[i],
                'execution_time': self.query_times[i],
                'cpu_usage': self.query_cpu[i],
                'memory_usage': self.query_memory[i]
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(all_records)
        df.to_csv(save_path, index=False)
        print(f"Performance metrics saved to {save_path}")
        return save_path

    def plot_performance_metrics(self, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # Time Plots
        plt.subplot(1, 3, 1)
        plt.plot(self.insertion_sizes, self.insertion_times, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_times, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_times, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_times, label='Query', marker='x')
        plt.title('Execution Time')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)

        # CPU Usage Plots
        plt.subplot(1, 3, 2)
        plt.plot(self.insertion_sizes, self.insertion_cpu, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_cpu, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_cpu, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_cpu, label='Query', marker='x')
        plt.title('CPU Usage')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('CPU Percentage')
        plt.legend()
        plt.grid(True)

        # Memory Usage Plots
        plt.subplot(1, 3, 3)
        plt.plot(self.insertion_sizes, self.insertion_memory, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_memory, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_memory, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_memory, label='Query', marker='x')
        plt.title('Memory Usage')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Performance Metrics for {self.model_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()

    def query_top_k(self, index, query_texts, k=5):
        # Generate and normalize query embeddings
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Search for top-k using inner product (equivalent to cosine similarity for normalized vectors)
        distances, indices = index.search(query_embeddings, k)
        
        return distances, indices  # distances = cosine similarities
    
    def track_query(self, index, query_texts, k=5):
        num_queries = len(query_texts)
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_texts, k
        )
        # Track metrics
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)

    def _query_embeddings(self, index, query_texts, k):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        index.nprobe = 10  # Adjust for recall vs. speed
        # nprobe is basically how many clusters you're willing to look at while you search
        #A higher nprobe means searching more clusters, which:
            # Increases recall (accuracy) because more candidates are considered.
            # Decreases speed because more data is examined
        # a lower nprobe is just the opposite
        return index.search(query_embeddings, k)

def main(): 
    csv_path = "ip_flow_dataset.csv" 
    
    df = pd.read_csv(
        csv_path,
        header=0,
        names=[
            "frame.number", "frame.time", "ip.src", "ip.dst",
            "tcp.srcport", "tcp.dstport", "_ws.col.protocol", "frame.len"
        ],
        dtype=str,  # force all columns to be strings
        skiprows=1  # skip the header row in your CSV since we're giving the column names
    )

    # Packet data to text format for BERT processing
    df["packet_text"] = (
        df["ip.src"].fillna('') + " " +
        df["ip.dst"].fillna('') + " " +
        df["_ws.col.protocol"].fillna('') + " " +
        df["tcp.srcport"].fillna('') + " " +
        df["tcp.dstport"].fillna('') + " " +
        df["_ws.col.protocol"].fillna('') + " " +
        df["frame.len"].fillna('')
    )

    modelList = [
    'paraphrase-MiniLM-L12-v2',
    "all-MiniLM-L6-v2", 
    'distilbert-base-nli-stsb-mean-tokens', 
    'microsoft/codebert-base', 
    'bert-base-nli-mean-tokens', 
    'sentence-transformers/average_word_embeddings_komninos', 
    'all-mpnet-base-v2'
    ]
    
    plots_dir = "PipelineResults"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a list to store all model results for combined CSV
    all_model_results = []
    
    for mod in modelList:
        print(f"\nProcessing model: {mod}")
        
        # clean the model name for filename
        clean_model_name = mod.replace('/', '_').replace('-', '_')
        # Generate embeddings
        model = SentenceTransformer(mod)
        embeddings = model.encode(df["packet_text"].tolist(), convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Creating FAISS IVFFlat index
        dimension = embeddings.shape[1]
        nlist = min(100, int(np.sqrt(len(embeddings))))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        # Initialize performance tracker
        tracker = FAISSPerformanceTracker(mod)  # Pass the model name to the tracker
        # Track performance for various operations and get the graph at the end of it
        for num_ops in [2500, 5000, 7500, 10000, 20000, 30000]:
            print(f"  Running {num_ops} operations...")
            
            # Insertion tracking
            new_packet_texts = [f"192.168.1.{i} 192.168.1.{i+1} TCP {i*10}" for i in range(num_ops)]
            tracker.track_insertion(index, new_packet_texts)
            # Deletion tracking
            tracker.track_deletion(index, num_ops)
            # Update tracking
            update_texts = [f"10.0.0.{i} 10.0.0.{i+1} UDP {i*5}" for i in range(num_ops)]
            tracker.track_update(index, num_ops, update_texts)
            
            # Track querying performance
            # UPDATE!!!!!: Using a variable number of queries that scales with num_ops
            query_texts = [f"192.168.1.{i} 192.168.1.{i+1} TCP {i*10}" for i in range(num_ops)]
            tracker.track_query(index, query_texts, k=5)
            
            # Querying and getting the nearest neighbors with cosine distances
            if num_ops == 30000:  # we'll show the cosine similarities only after the largest batch
                test_query = ["192.168.1.1 192.168.1.2 TCP 100"]
                distances, indices = tracker.query_top_k(index, test_query, k=5)
                print("  Top 5 Neighbors (indices):", indices[0])
                print("  Cosine Similarities:", distances[0])
        
        # Save individual model results to CSV
        csv_filename = f"{plots_dir}/{clean_model_name}_metrics.csv"
        tracker.save_metrics_to_csv(save_path=csv_filename)
        
        # Save plot
        plot_filename = f"{plots_dir}/{clean_model_name}.png"
        tracker.plot_performance_metrics(save_path=plot_filename)
        
        # Collect data for combined CSV
        all_records = []
        for i in range(len(tracker.insertion_sizes)):
            all_records.append({
                'model_name': mod,
                'operation_type': 'insertion',
                'operation_size': tracker.insertion_sizes[i],
                'execution_time': tracker.insertion_times[i],
                'cpu_usage': tracker.insertion_cpu[i],
                'memory_usage': tracker.insertion_memory[i]
            })
        for i in range(len(tracker.deletion_sizes)):
            all_records.append({
                'model_name': mod,
                'operation_type': 'deletion',
                'operation_size': tracker.deletion_sizes[i],
                'execution_time': tracker.deletion_times[i],
                'cpu_usage': tracker.deletion_cpu[i],
                'memory_usage': tracker.deletion_memory[i]
            })
        for i in range(len(tracker.update_sizes)):
            all_records.append({
                'model_name': mod,
                'operation_type': 'update',
                'operation_size': tracker.update_sizes[i],
                'execution_time': tracker.update_times[i],
                'cpu_usage': tracker.update_cpu[i],
                'memory_usage': tracker.update_memory[i]
            })
        for i in range(len(tracker.query_sizes)):
            all_records.append({
                'model_name': mod,
                'operation_type': 'query',
                'operation_size': tracker.query_sizes[i],
                'execution_time': tracker.query_times[i],
                'cpu_usage': tracker.query_cpu[i],
                'memory_usage': tracker.query_memory[i]
            })
        all_model_results.extend(all_records)
    
    # Save combined results for all models
    combined_df = pd.DataFrame(all_model_results)
    combined_csv_path = f"{plots_dir}/all_models_performance_metrics.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined performance metrics for all models saved to {combined_csv_path}")

if __name__ == "__main__":
    main()
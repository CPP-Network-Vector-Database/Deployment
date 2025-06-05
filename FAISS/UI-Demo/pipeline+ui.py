import streamlit as st
import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FAISSPerformanceTracker:
    def __init__(self, model_name='distilbert-base-nli-stsb-mean-tokens'):
        self.model = None
        self.model_name = model_name
        
        # Performance tracking lists
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
    
    def load_model(self, model_name):
        if self.model is None or self.model_name != model_name:
            with st.spinner(f"Loading model: {model_name}..."):
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name
                st.success(f"Model {model_name} loaded successfully!")
    
    def reset_metrics(self):
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
    
    def get_metrics_summary(self):
        metrics_data = []
        
        operations = [
            ('Insertion', self.insertion_sizes, self.insertion_times, self.insertion_cpu, self.insertion_memory),
            ('Deletion', self.deletion_sizes, self.deletion_times, self.deletion_cpu, self.deletion_memory),
            ('Update', self.update_sizes, self.update_times, self.update_cpu, self.update_memory),
            ('Query', self.query_sizes, self.query_times, self.query_cpu, self.query_memory)
        ]
        
        for op_name, sizes, times, cpu, memory in operations:
            if len(times) > 0:
                metrics_data.append({
                    'Operation': op_name,
                    'Total Operations': len(times),
                    'Avg Time (s)': f"{np.mean(times):.4f}",
                    'Min Time (s)': f"{np.min(times):.4f}",
                    'Max Time (s)': f"{np.max(times):.4f}",
                    'Avg CPU (%)': f"{np.mean(cpu):.2f}",
                    'Avg Memory (MB)': f"{np.mean(memory):.2f}",
                    'Total Items Processed': sum(sizes)
                })
        
        return pd.DataFrame(metrics_data) if metrics_data else None
    
    def _measure_performance(self, func, *args, **kwargs):
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
        cpu_usage = max(0, end_cpu - start_cpu)  # Ensure non-negative
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage
    
    def track_insertion(self, index, new_packet_texts):
        if self.model is None:
            raise ValueError("Model not loaded. Please select and load a model first.")
        
        # Normalize embeddings
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        # Track metrics
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)
        
        return len(new_embeddings)
    
    def track_deletion(self, index, num_deletions):
        if index.ntotal == 0:
            return 0
        
        # Select random indices to delete
        actual_deletions = min(num_deletions, index.ntotal)
        delete_indices = np.random.choice(index.ntotal, actual_deletions, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, index, delete_indices
        )
        
        # Track metrics
        self.deletion_sizes.append(actual_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)
        
        return actual_deletions
    
    def track_update(self, index, num_updates, new_packet_texts):
        if self.model is None:
            raise ValueError("Model not loaded. Please select and load a model first.")
        
        if index.ntotal == 0:
            return 0
        
        # Select random indices to update
        actual_updates = min(num_updates, index.ntotal)
        update_indices = np.random.choice(index.ntotal, actual_updates, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_packet_texts[:actual_updates]
        )
        
        # Track metrics
        self.update_sizes.append(actual_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage)
        
        return actual_updates
    
    def track_query(self, index, query_texts, k=5):
        if self.model is None:
            raise ValueError("Model not loaded. Please select and load a model first.")
        
        if index.ntotal == 0:
            return [], []
        
        num_queries = len(query_texts)
        # Measure performance
        result, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_texts, k
        )
        
        # Track metrics
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)
        
        return result
    
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
    
    def _query_embeddings(self, index, query_texts, k):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Set nprobe for IVF index
        if hasattr(index, 'nprobe'):
            index.nprobe = min(10, index.nlist)
        
        actual_k = min(k, index.ntotal)
        return index.search(query_embeddings, actual_k)
    
    def create_performance_plot(self):
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Execution Time', 'CPU Usage', 'Memory Usage'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time plot
        if self.insertion_sizes:
            fig.add_trace(go.Scatter(x=self.insertion_sizes, y=self.insertion_times, 
                                   mode='lines+markers', name='Insertion', line=dict(color='blue')), row=1, col=1)
        if self.deletion_sizes:
            fig.add_trace(go.Scatter(x=self.deletion_sizes, y=self.deletion_times, 
                                   mode='lines+markers', name='Deletion', line=dict(color='red')), row=1, col=1)
        if self.update_sizes:
            fig.add_trace(go.Scatter(x=self.update_sizes, y=self.update_times, 
                                   mode='lines+markers', name='Update', line=dict(color='green')), row=1, col=1)
        if self.query_sizes:
            fig.add_trace(go.Scatter(x=self.query_sizes, y=self.query_times, 
                                   mode='lines+markers', name='Query', line=dict(color='orange')), row=1, col=1)
        
        # CPU plot
        if self.insertion_sizes:
            fig.add_trace(go.Scatter(x=self.insertion_sizes, y=self.insertion_cpu, 
                                   mode='lines+markers', name='Insertion', line=dict(color='blue'), 
                                   showlegend=False), row=1, col=2)
        if self.deletion_sizes:
            fig.add_trace(go.Scatter(x=self.deletion_sizes, y=self.deletion_cpu, 
                                   mode='lines+markers', name='Deletion', line=dict(color='red'), 
                                   showlegend=False), row=1, col=2)
        if self.update_sizes:
            fig.add_trace(go.Scatter(x=self.update_sizes, y=self.update_cpu, 
                                   mode='lines+markers', name='Update', line=dict(color='green'), 
                                   showlegend=False), row=1, col=2)
        if self.query_sizes:
            fig.add_trace(go.Scatter(x=self.query_sizes, y=self.query_cpu, 
                                   mode='lines+markers', name='Query', line=dict(color='orange'), 
                                   showlegend=False), row=1, col=2)
        
        # Memory plot
        if self.insertion_sizes:
            fig.add_trace(go.Scatter(x=self.insertion_sizes, y=self.insertion_memory, 
                                   mode='lines+markers', name='Insertion', line=dict(color='blue'), 
                                   showlegend=False), row=1, col=3)
        if self.deletion_sizes:
            fig.add_trace(go.Scatter(x=self.deletion_sizes, y=self.deletion_memory, 
                                   mode='lines+markers', name='Deletion', line=dict(color='red'), 
                                   showlegend=False), row=1, col=3)
        if self.update_sizes:
            fig.add_trace(go.Scatter(x=self.update_sizes, y=self.update_memory, 
                                   mode='lines+markers', name='Update', line=dict(color='green'), 
                                   showlegend=False), row=1, col=3)
        if self.query_sizes:
            fig.add_trace(go.Scatter(x=self.query_sizes, y=self.query_memory, 
                                   mode='lines+markers', name='Query', line=dict(color='orange'), 
                                   showlegend=False), row=1, col=3)
        
        # Update layout
        fig.update_xaxes(title_text="Number of Operations", row=1, col=1)
        fig.update_xaxes(title_text="Number of Operations", row=1, col=2)
        fig.update_xaxes(title_text="Number of Operations", row=1, col=3)
        
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="CPU Percentage", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=3)
        
        fig.update_layout(
            title=f'Performance Metrics for {self.model_name}',
            height=500,
            showlegend=True
        )
        
        return fig

def load_csv_data(csv_path):
    try:
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
            df["frame.len"].fillna('')
        )
        
        return df["packet_text"].tolist()
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    nlist = min(100, max(1, int(np.sqrt(len(embeddings)))))
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    
    index.train(embeddings)
    index.add(embeddings)
    return index

def initialize_session_state():
    if 'tracker' not in st.session_state:
        st.session_state.tracker = FAISSPerformanceTracker()
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'data_embedded' not in st.session_state:
        st.session_state.data_embedded = False

def main():
    st.set_page_config(page_title="FAISS Performance Tracker", layout="wide")
    st.title("FAISS Performance Tracker with Multiple Models")
    st.markdown("Choose your model, load data, and perform CRUD operations to visualize performance metrics")
    
    # Initialize session state
    initialize_session_state()
    
    # Model selection in sidebar
    st.sidebar.header("Model Configuration")
    
    modelList = [
        'paraphrase-MiniLM-L12-v2',
        "all-MiniLM-L6-v2", 
        'distilbert-base-nli-stsb-mean-tokens', 
        'microsoft/codebert-base', 
        'bert-base-nli-mean-tokens', 
        'sentence-transformers/average_word_embeddings_komninos', 
        'all-mpnet-base-v2'
    ]
    
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        modelList,
        index=2,  # Default to distilbert
        help="Select the sentence transformer model to use for embeddings"
    )
    
    # Check if model has changed
    model_changed = (hasattr(st.session_state.tracker, 'model_name') and 
                    st.session_state.tracker.model_name != selected_model)
    
    if model_changed:
        st.sidebar.warning("Model changed! You'll need to re-embed your data.")
        if st.sidebar.button("Switch Model & Reset"):
            # Reset everything for new model
            st.session_state.tracker = FAISSPerformanceTracker(selected_model)
            st.session_state.model_loaded = False
            st.session_state.data_embedded = False
            if 'index' in st.session_state:
                del st.session_state.index
            if 'data_texts' in st.session_state:
                del st.session_state.data_texts
            st.rerun()
    
    # Load model button
    if not st.session_state.model_loaded or st.session_state.tracker.model is None:
        if st.sidebar.button("Load Model"):
            try:
                st.session_state.tracker.load_model(selected_model)
                st.session_state.model_loaded = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error loading model: {str(e)}")
    else:
        st.sidebar.success(f"Model loaded: {st.session_state.tracker.model_name}")
        
        # Option to reset metrics for comparison
        if st.sidebar.button("🧹 Reset Performance Metrics"):
            st.session_state.tracker.reset_metrics()
            st.sidebar.success("Metrics reset!")
            st.rerun()
    
    # Data loading section
    st.sidebar.header("Data Configuration")
    
    if st.session_state.model_loaded:
        csv_path = st.sidebar.text_input(
            "CSV File Path:", 
            value="/home/pes1ug22am100/Documents/WeeklyProgress/Week8-9/FAISS/ip_flow_dataset.csv",
            help="Enter the path to your CSV file"
        )
        
        data_source = st.sidebar.radio(
            "Data Source:",
            ["Load from CSV", "Use Sample Data"],
            help="Choose whether to load data from CSV or use sample data"
        )
        
        if st.sidebar.button("Load & Embed Data") or not st.session_state.data_embedded:
            if data_source == "Load from CSV":
                packet_texts = load_csv_data(csv_path)
                if packet_texts is None:
                    st.sidebar.warning("Could not load CSV, switching to sample data")
                    data_source = "Use Sample Data"
            
            if data_source == "Use Sample Data" or packet_texts is None:
                packet_texts = [
                    "192.168.1.1 192.168.1.2 TCP 80 443",
                    "10.0.0.1 10.0.0.2 UDP 53 8080",
                    "172.16.0.1 172.16.0.2 HTTP 80 8080",
                    "192.168.0.1 192.168.0.2 HTTPS 443 443",
                    "10.10.10.1 10.10.10.2 FTP 21 21",
                    "172.16.1.1 172.16.1.2 SSH 22 22"
                ]
            
            if packet_texts:
                try:
                    with st.spinner("Creating embeddings..."):
                        embeddings = st.session_state.tracker.model.encode(packet_texts, convert_to_numpy=True)
                        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    with st.spinner("Creating FAISS index..."):
                        index = create_faiss_index(embeddings)
                    
                    st.session_state.index = index
                    st.session_state.data_texts = packet_texts.copy()
                    st.session_state.data_embedded = True
                    
                    st.sidebar.success(f"Loaded and embedded {len(packet_texts)} entries!")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error during embedding: {str(e)}")
    else:
        st.sidebar.info("Please load a model first")
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("Please select and load a model from the sidebar to get started!")
        return
    
    if not st.session_state.data_embedded:
        st.info("Please load and embed your data from the sidebar to continue!")
        return
    
    # Operations section
    st.sidebar.header("Operations")
    operation = st.sidebar.selectbox(
        "Choose Operation:",
        ["Insert", "Delete", "Update", "Query", "View Data"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{operation} Operation")
        
        if operation == "Insert":
            st.write("Add new packet data to the database")
            new_data = st.text_area(
                "Enter packet data (one per line):",
                placeholder="192.168.1.10 192.168.1.20 TCP 80 8080\n10.0.0.5 10.0.0.10 UDP 53 53",
                height=100
            )
            num_entries = st.slider("Number of entries to generate (if text area is empty):", 1, 1000, 10)
            
            if st.button("Insert Data"):
                try:
                    if new_data.strip():
                        new_texts = [line.strip() for line in new_data.split('\n') if line.strip()]
                    else:
                        # Generate random data
                        new_texts = [f"192.168.{i//255}.{i%255} 10.0.{i//255}.{i%255} TCP {80+i} {8000+i}" 
                                   for i in range(num_entries)]
                    
                    with st.spinner("Inserting data..."):
                        inserted_count = st.session_state.tracker.track_insertion(st.session_state.index, new_texts)
                        st.session_state.data_texts.extend(new_texts)
                    
                    st.success(f"Inserted {inserted_count} entries successfully!")
                    st.info(f"Total entries in database: {st.session_state.index.ntotal}")
                except Exception as e:
                    st.error(f"Error during insertion: {str(e)}")
        
        elif operation == "Delete":
            st.write("Remove entries from the database")
            num_to_delete = st.slider("Number of entries to delete:", 1, 
                                    max(1, st.session_state.index.ntotal), 1)
            
            if st.button("Delete Data"):
                if st.session_state.index.ntotal > 0:
                    try:
                        with st.spinner("Deleting data..."):
                            deleted_count = st.session_state.tracker.track_deletion(st.session_state.index, num_to_delete)
                            # Remove from data_texts as well (approximately)
                            if len(st.session_state.data_texts) >= deleted_count:
                                st.session_state.data_texts = st.session_state.data_texts[:-deleted_count]
                        
                        st.success(f"Deleted {deleted_count} entries successfully!")
                        st.info(f"Remaining entries in database: {st.session_state.index.ntotal}")
                    except Exception as e:
                        st.error(f"Error during deletion: {str(e)}")
                else:
                    st.warning("No data to delete!")
        
        elif operation == "Update":
            st.write("Update existing entries in the database")
            num_to_update = st.slider("Number of entries to update:", 1, 
                                    max(1, st.session_state.index.ntotal), 1)
            update_data = st.text_area(
                "Enter new packet data (one per line):",
                placeholder="10.10.10.1 10.10.10.2 HTTPS 443 443\n172.16.1.1 172.16.1.2 SSH 22 22",
                height=100
            )
            
            if st.button("Update Data"):
                if st.session_state.index.ntotal > 0:
                    try:
                        if update_data.strip():
                            new_texts = [line.strip() for line in update_data.split('\n') if line.strip()]
                        else:
                            # Generate random update data
                            new_texts = [f"10.10.{i//255}.{i%255} 172.16.{i//255}.{i%255} HTTPS {443+i} {9000+i}" 
                                       for i in range(num_to_update)]
                        
                        with st.spinner("Updating data..."):
                            updated_count = st.session_state.tracker.track_update(st.session_state.index, num_to_update, new_texts)
                        
                        st.success(f"Updated {updated_count} entries successfully!")
                        st.info(f"Total entries in database: {st.session_state.index.ntotal}")
                    except Exception as e:
                        st.error(f"Error during update: {str(e)}")
                else:
                    st.warning("No data to update!")
        
        elif operation == "Query":
            st.write("Search for similar packet data")
            query_text = st.text_input(
                "Enter query:",
                placeholder="192.168.1.1 192.168.1.2 TCP 80 443"
            )
            k = st.slider("Number of results (k):", 1, min(20, max(1, st.session_state.index.ntotal)), 5)
            
            if st.button("Search"):
                if query_text.strip() and st.session_state.index.ntotal > 0:
                    try:
                        with st.spinner("Searching..."):
                            distances, indices = st.session_state.tracker.track_query(
                                st.session_state.index, [query_text], k
                            )
                        
                        st.success(f"Query completed!")
                        
                        # Display results
                        if len(distances) > 0 and len(distances[0]) > 0:
                            st.write("**Search Results:**")
                            results_df = pd.DataFrame({
                                'Rank': range(1, len(distances[0]) + 1),
                                'Index': indices[0],
                                'Similarity Score': [f"{1-d:.4f}" for d in distances[0]],  # Convert L2 to similarity
                                'Data': [st.session_state.data_texts[i] if i < len(st.session_state.data_texts) 
                                       else f"Entry {i}" for i in indices[0]]
                            })
                            st.dataframe(results_df, use_container_width=True)
                        else:
                            st.warning("No results found!")
                    except Exception as e:
                        st.error(f"Error during query: {str(e)}")
                elif not query_text.strip():
                    st.warning("Please enter a query!")
                else:
                    st.warning("Database is empty!")
        
        elif operation == "View Data":
            st.write("Current database statistics")
            st.metric("Total Entries", st.session_state.index.ntotal)
            st.metric("Current Model", st.session_state.tracker.model_name)
            st.metric("Data Embedded", "Yes" if st.session_state.data_embedded else "No")
            
            # Performance metrics table
            st.write("**Performance Metrics Summary:**")
            metrics_df = st.session_state.tracker.get_metrics_summary()
            if metrics_df is not None:
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No performance metrics available yet. Perform some operations to see metrics!")
            
            if st.session_state.data_texts:
                st.write("**Sample Data:**")
                sample_size = min(10, len(st.session_state.data_texts))
                sample_df = pd.DataFrame({
                    'Index': range(1, sample_size + 1),
                    'Packet Data': st.session_state.data_texts[:sample_size]
                })
                st.dataframe(sample_df, use_container_width=True)
                
                if len(st.session_state.data_texts) > sample_size:
                    st.write(f"... and {len(st.session_state.data_texts) - sample_size} more entries")
                
                # Show data distribution
                if st.checkbox("Show Data Analysis"):
                    st.write("**Data Analysis:**")
                    
                    # Protocol distribution
                    protocols = []
                    for text in st.session_state.data_texts[:1000]:  # Sample first 1000 for performance
                        parts = text.split()
                        if len(parts) >= 3:
                            protocols.append(parts[2])  # Protocol is usually the 3rd element
                    
                    if protocols:
                        protocol_counts = pd.Series(protocols).value_counts()
                        st.write("**Top Protocols:**")
                        st.bar_chart(protocol_counts.head(10))
    
    with col2:
        st.subheader("Performance Metrics")
        
        # Check if we have any performance data
        has_data = (len(st.session_state.tracker.insertion_times) > 0 or 
                   len(st.session_state.tracker.deletion_times) > 0 or 
                   len(st.session_state.tracker.update_times) > 0 or 
                   len(st.session_state.tracker.query_times) > 0)
        
        if has_data:
            fig = st.session_state.tracker.create_performance_plot()
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary
            st.write("**Performance Summary:**")
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                if st.session_state.tracker.insertion_times:
                    avg_insert_time = np.mean(st.session_state.tracker.insertion_times)
                    st.metric("Avg Insert Time", f"{avg_insert_time:.4f}s")
                
                if st.session_state.tracker.query_times:
                    avg_query_time = np.mean(st.session_state.tracker.query_times)
                    st.metric("Avg Query Time", f"{avg_query_time:.4f}s")
            
            with metrics_col2:
                if st.session_state.tracker.deletion_times:
                    avg_delete_time = np.mean(st.session_state.tracker.deletion_times)
                    st.metric("Avg Delete Time", f"{avg_delete_time:.4f}s")
                
                if st.session_state.tracker.update_times:
                    avg_update_time = np.mean(st.session_state.tracker.update_times)
                    st.metric("Avg Update Time", f"{avg_update_time:.4f}s")
            
            
        else:
            st.info("Performance graphs will appear here after you perform operations")
            st.write("Try inserting, deleting, updating, or querying data to see performance metrics!")
            
            # Show model info while waiting
            st.write("**Current Configuration:**")
            st.write(f"*Model**: {st.session_state.tracker.model_name}")
            st.write(f"**Data Points**: {st.session_state.index.ntotal}")

    # Footer
    st.markdown("---")
    st.markdown("1. Perform operations with one model and note the performance")
    st.markdown("2. Switch to a different model using the sidebar")
    st.markdown("3. Perform similar operations and compare the results")
    st.markdown("4. Use the 'View Data' tab to see detailed performance metrics")

if __name__ == "__main__":
    main()
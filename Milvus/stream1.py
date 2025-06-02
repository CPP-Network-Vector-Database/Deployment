
import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import psutil
import threading
import gc
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BASE_COLLECTION_NAME = "st_network_crud" 
CHUNK_SIZE = 5000 
EMBEDDING_BATCH_SIZE = 32
MAX_ROWS_INITIAL_LOAD = 10000
ADD_ROWS_COUNT = 1000
DELETE_LAST_N_ROWS_DEFAULT = 500

CSV_FILE_PATH = "ip_flow_dataset.csv" 
COLUMNS_TO_EMBED = ['frame.number', 'frame.time', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', '_ws.col.protocol', 'frame.len']
NUM_QUERY_SAMPLES = 3
VECTOR_DIM = None 

# Available Models
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L12-v2": "sentence-transformers/paraphrase-MiniLM-L12-v2",
    "stsb-distilbert-base": "sentence-transformers/stsb-distilbert-base"
}

#Streamlit Page Config 
st.set_page_config(layout="wide", page_title="Milvus Interactive CRUD Benchmark")

# CPU and Memory Monitoring Setup 
cpu_usage_data = []
memory_usage_data = []
recording = True

def record_system_resources():
    global recording, cpu_usage_data, memory_usage_data
    process = psutil.Process()
    while recording:
        try:
            cpu_usage_data.append(psutil.cpu_percent(interval=0.01, percpu=False))
            memory_usage_data.append(process.memory_info().rss / (1024 * 1024)) 
            time.sleep(0.01)
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            recording = False; break

def start_resource_monitor():
    global recording, cpu_usage_data, memory_usage_data
    recording = True; cpu_usage_data = []; memory_usage_data = []
    monitor_thread = threading.Thread(target=record_system_resources, daemon=True); monitor_thread.start()

def stop_resource_monitor():
    global recording; recording = False; time.sleep(0.05)
    return {"cpu_usage": cpu_usage_data, "memory_usage": memory_usage_data}

# Milvus Connection 
@st.cache_resource
def get_milvus_connection():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=20)
        st.sidebar.success(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        st.sidebar.error(f"Milvus Connection Failed: {e}")
        return False

@st.cache_resource
def load_embedding_model(model_key):
    global VECTOR_DIM
    model_name = AVAILABLE_MODELS[model_key]
    status_text = st.sidebar.empty()
    status_text.info(f"Loading model: {model_key}...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(model_name, device=device)
        VECTOR_DIM = model.get_sentence_embedding_dimension()
        status_text.success(f"Model '{model_key}' loaded on {device}. Dim: {VECTOR_DIM}")
        return model
    except Exception as e:
        status_text.error(f"Failed to load model '{model_name}': {e}")
        return None

def get_sentence_embedding(text_list, model_instance):
    if not text_list or model_instance is None: return np.array([])
    return model_instance.encode(text_list, batch_size=EMBEDDING_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)


def get_dynamic_collection_name(model_key):
    safe_model_key = model_key.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"{BASE_COLLECTION_NAME}_{safe_model_key}"

def setup_collection(model_key, _model_instance):
    collection_name = get_dynamic_collection_name(model_key)
    if not VECTOR_DIM:
        st.error("Vector dimension not set. Cannot setup collection."); return None

    if utility.has_collection(collection_name):
        st.info(f"Using existing collection: '{collection_name}'")
        current_collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="frame_number", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]
        schema = CollectionSchema(fields, description=f"CRUD Benchmark: {model_key}")
        try:
            current_collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")
            st.success(f"Collection '{collection_name}' created.")
            index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
            current_collection.create_index("embedding", index_params)
            utility.wait_for_index_building_complete(collection_name)
            st.info("Default HNSW/COSINE index created.")
        except Exception as e:
            st.error(f"Failed to create/index collection '{collection_name}': {e}")
            return None
   
    
    if not current_collection.has_index():
        st.warning("Collection has no index. Creating default HNSW/COSINE index.")
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
        current_collection.create_index("embedding", index_params)
        utility.wait_for_index_building_complete(collection_name)
    try:
        current_collection.load()
        st.info(f"Collection '{collection_name}' loaded.")
    except Exception as e:
        st.error(f"Failed to load collection '{collection_name}': {e}")
    return current_collection

# BENCHMARKING FUNCTION 
def benchmark_operation(collection, query_vectors_list, query_identifiers_list, operation_description="Operation"):
    if not query_vectors_list or not query_identifiers_list:
        st.warning("No query vectors for benchmarking."); return {'latency': 0, 'cpu': 0, 'memory': 0}
   
    op_latencies, op_cpu_usages, op_peak_memories = [], [], []
    metric_type = "COSINE" 
   
    try:
        collection.load() 
        st.info(f"Benchmarking {len(query_vectors_list)} queries (Metric: {metric_type}) on '{collection.name}'...")
        if collection.is_empty: 
             st.warning(f"Collection '{collection.name}' is empty. Skipping benchmark queries."); return {'latency': 0, 'cpu': 0, 'memory': 0}

        bench_progress = st.progress(0)

        for i, query_vec_np in enumerate(query_vectors_list):
            query_id = query_identifiers_list[i]
            query_vec_list_of_list = [query_vec_np.tolist()]

            start_resource_monitor()
            query_start_time = time.time()
            try:
                search_params = {"metric_type": metric_type, "params": {"ef": 128}} # HNSW search param
                collection.search(query_vec_list_of_list, "embedding", search_params, limit=10)
            except Exception as e:
                st.error(f"Search failed for query {query_id}: {e}")
                _ = stop_resource_monitor(); continue
           
            query_latency = time.time() - query_start_time
            resource_data = stop_resource_monitor()

            op_latencies.append(query_latency)
            op_cpu_usages.append(np.mean(resource_data.get("cpu_usage", [0])))
            op_peak_memories.append(np.max(resource_data.get("memory_usage", [0])))
            bench_progress.progress((i + 1) / len(query_vectors_list))
       
        bench_progress.empty()

    except Exception as e:
        st.error(f"Error during benchmark operation setup: {e}")
        return {'latency': 0, 'cpu': 0, 'memory': 0}
    finally:
        if collection:
            try:
                collection.release() 
            except Exception as e_release:
                st.warning(f"Could not release collection: {e_release}")


    if not op_latencies: return {'latency': 0, 'cpu': 0, 'memory': 0}
    results = {
        'latency': np.mean(op_latencies), 'cpu': np.mean(op_cpu_usages), 'memory': np.mean(op_peak_memories)
    }
    st.subheader(f"--- Avg Performance: {operation_description} ---")
    st.metric("Avg Query Latency", f"{results['latency']:.4f}s")
    st.metric("Avg CPU Usage", f"{results['cpu']:.2f}%")
    st.metric("Avg Peak Memory", f"{results['memory']:.2f} MB")
    return results

# CRUD Operations 
def perform_initial_load(collection, model_instance, status_placeholder):
    collection.load() 
    if collection.num_entities > 0 :
        status_placeholder.info(f"Collection already has {collection.num_entities} entities. Skipping initial load.")
        collection.release()
        return True
    collection.release()

    try:
        with st.spinner(f"Performing initial load of up to {MAX_ROWS_INITIAL_LOAD} rows..."):
            df_initial = pd.read_csv(CSV_FILE_PATH, nrows=MAX_ROWS_INITIAL_LOAD, low_memory=False)
            st.session_state.csv_columns = df_initial.columns.tolist() 

            df_initial.dropna(subset=['frame.number'], inplace=True)
            df_initial['frame.number'] = pd.to_numeric(df_initial['frame.number'], errors='coerce')
            df_initial.dropna(subset=['frame.number'], inplace=True)
            df_initial.drop_duplicates(subset=['frame.number'], keep='first', inplace=True)

            if df_initial.empty:
                status_placeholder.warning("No valid data loaded from CSV for initial load."); return False

            frame_numbers = df_initial['frame.number'].astype(np.int64).tolist()
            valid_cols = [col for col in COLUMNS_TO_EMBED if col in df_initial.columns]
            df_initial['combined_text'] = df_initial[valid_cols].apply(lambda row: ' '.join(row.astype(str)), axis=1)
            embeddings = get_sentence_embedding(df_initial['combined_text'].tolist(), model_instance)
           
            collection.insert([frame_numbers, embeddings.tolist()])
            collection.flush()
            status_placeholder.success(f"Initial data loaded & flushed. Total rows: {collection.num_entities}")
            st.session_state.df_initial_data_cache = df_initial.copy() 
            st.session_state.df_initial_data_cache_columns = df_initial.columns.tolist() 
            return True
    except Exception as e:
        status_placeholder.error(f"Initial load failed: {e}")
        return False

def perform_add_data(collection, model_instance, status_placeholder):
    try:
        collection.load()
        current_entity_count = collection.num_entities
        collection.release() 

        
        column_names = st.session_state.get('csv_columns', None)
        if not column_names:
            try: 
                column_names = pd.read_csv(CSV_FILE_PATH, nrows=0).columns.tolist()
                st.session_state.csv_columns = column_names
            except Exception as e_read_header:
                status_placeholder.error(f"Critical: Could not read CSV header to determine columns: {e_read_header}")
                return False

        # Number of lines to skip from the top of the file:
        # 1 (for the original header) + current_entity_count (for already processed data rows)
        lines_to_skip_from_top = current_entity_count + 1
       
        df_add = pd.read_csv(CSV_FILE_PATH,
                             nrows=ADD_ROWS_COUNT,
                             skiprows=lines_to_skip_from_top,
                             header=None, 
                             names=column_names, 
                             low_memory=False)
       
        df_add.dropna(subset=['frame.number'], inplace=True)
        df_add['frame.number'] = pd.to_numeric(df_add['frame.number'], errors='coerce')
        df_add.dropna(subset=['frame.number'], inplace=True)
        df_add.drop_duplicates(subset=['frame.number'], keep='first', inplace=True)

        if df_add.empty:
            status_placeholder.warning("No new data found in CSV to add."); return False
       
        frame_numbers_add_orig = df_add['frame.number'].astype(np.int64).tolist()
       
    
        new_frame_numbers = frame_numbers_add_orig
       
        if not new_frame_numbers:
            status_placeholder.warning("No new unique rows to add after filtering (or all were duplicates)."); return False
       
        valid_cols = [col for col in COLUMNS_TO_EMBED if col in df_add.columns]
        df_add['combined_text'] = df_add[valid_cols].apply(lambda r: ' '.join(r.astype(str)), axis=1)
        embeddings_add = get_sentence_embedding(df_add['combined_text'].tolist(), model_instance)

        collection.insert([new_frame_numbers, embeddings_add.tolist()]) 
        collection.flush()
        status_placeholder.success(f"{len(new_frame_numbers)} rows added. Total rows: {collection.num_entities}")
        return True
    except Exception as e:
        status_placeholder.error(f"Data addition failed: {e}")
        return False

def perform_delete_data(collection, num_rows_to_delete, status_placeholder):
    try:
        collection.load()
        if collection.is_empty:
            status_placeholder.warning("Collection is empty. Nothing to delete.")
            return False

        status_placeholder.info(f"Fetching frame numbers to identify last {num_rows_to_delete} rows by highest frame_number...")
       
        all_pks_result = collection.query(expr="frame_number >= 0", output_fields=["frame_number"], limit=16384)
       
        if not all_pks_result:
            status_placeholder.warning("No frame numbers found in collection.")
            return False

        all_frame_numbers = sorted([item['frame_number'] for item in all_pks_result], reverse=True)
       
        ids_to_delete = all_frame_numbers[:num_rows_to_delete]

        if not ids_to_delete:
            status_placeholder.info("No rows identified for deletion (perhaps num_rows_to_delete > actual rows).")
            return False

        expr = f"frame_number in {ids_to_delete}"
        delete_result = collection.delete(expr)
        collection.flush()
        status_placeholder.success(f"{delete_result.delete_count} rows deleted. Total entities: {collection.num_entities}")
        return True
    except Exception as e:
        status_placeholder.error(f"Delete operation failed: {e}")
        return False
    finally:
        if collection: collection.release()

def perform_update_data(collection, model_instance, frame_number_to_update, new_ip_src, status_placeholder):
    try:
        original_row_df = None
        # Try to get the original row from the initial data cache
        if 'df_initial_data_cache' in st.session_state:
            initial_cache = st.session_state.df_initial_data_cache
            if isinstance(initial_cache, pd.DataFrame) and 'frame.number' in initial_cache.columns:
                 original_row_df = initial_cache[initial_cache['frame.number'] == frame_number_to_update].copy()

        if (original_row_df is None or original_row_df.empty):
            
            cached_csv_columns = st.session_state.get('csv_columns')
            if cached_csv_columns:
                
                try:
                    df_full_scan = pd.read_csv(CSV_FILE_PATH, low_memory=False) 
                    original_row_df = df_full_scan[df_full_scan['frame.number'] == frame_number_to_update].copy()
                except Exception as e_scan:
                    st.warning(f"Could not scan CSV for update source: {e_scan}")


        if original_row_df is None or original_row_df.empty:
            status_placeholder.warning(f"Data for frame {frame_number_to_update} not found. Creating placeholder for update.")
            placeholder_data = {col: 'placeholder' for col in COLUMNS_TO_EMBED if col != 'ip.src'}
            placeholder_data['frame.number'] = frame_number_to_update
            placeholder_data['ip.src'] = new_ip_src
            original_row_df = pd.DataFrame([placeholder_data])
        else:
            original_row_df['ip.src'] = new_ip_src 

        valid_cols_update = [col for col in COLUMNS_TO_EMBED if col in original_row_df.columns]
        if not valid_cols_update: 
             status_placeholder.error("Critical: No valid columns for embedding found in the row to update.")
             return False
        original_row_df['combined_text'] = original_row_df[valid_cols_update].apply(lambda row: ' '.join(row.astype(str)), axis=1)
       
        if original_row_df['combined_text'].empty:
            status_placeholder.error(f"Failed to generate combined text for frame {frame_number_to_update}.")
            return False

        updated_embedding_list = get_sentence_embedding(original_row_df['combined_text'].tolist(), model_instance)
        if updated_embedding_list.size == 0:
            status_placeholder.error(f"Failed to generate embedding for updated frame {frame_number_to_update}.")
            return False
        updated_embedding = updated_embedding_list[0]
       
        collection.upsert([[frame_number_to_update], [updated_embedding.tolist()]])
        collection.flush()
        status_placeholder.success(f"Frame {frame_number_to_update} updated (upserted). Total: {collection.num_entities}")
        return True
    except Exception as e:
        status_placeholder.error(f"Update failed: {e}")
        return False


def prepare_query_samples(model_instance):
    if 'query_vectors' in st.session_state and st.session_state.get('current_model_for_queries') == st.session_state.selected_model_key:
        return st.session_state.query_vectors, st.session_state.query_identifiers

    status_placeholder_qs = st.sidebar.empty()
    status_placeholder_qs.info("Preparing query samples...")
    try:
        df_qs = pd.read_csv(CSV_FILE_PATH, nrows=max(NUM_QUERY_SAMPLES,100), low_memory=False)
       
        
        if 'csv_columns' not in st.session_state:
            st.session_state.csv_columns = df_qs.columns.tolist()
        st.session_state.df_query_samples_cache_columns = df_qs.columns.tolist() 
       
        if 'df_initial_data_cache' not in st.session_state: 
             st.session_state.df_initial_data_cache = df_qs.copy()
             st.session_state.df_initial_data_cache_columns = df_qs.columns.tolist()


        df_query_samples_for_embed = df_qs.head(NUM_QUERY_SAMPLES).copy()
        st.session_state.df_query_samples_cache = df_query_samples_for_embed.copy()

        valid_cols_qs = [col for col in COLUMNS_TO_EMBED if col in df_query_samples_for_embed.columns]
        df_query_samples_for_embed['combined_text'] = df_query_samples_for_embed[valid_cols_qs].apply(lambda row: ' '.join(row.astype(str)), axis=1)
       
        q_embeddings_np = get_sentence_embedding(df_query_samples_for_embed['combined_text'].tolist(), model_instance)
       
        st.session_state.query_vectors = [emb for emb in q_embeddings_np]
        st.session_state.query_identifiers = df_query_samples_for_embed['frame.number'].astype(np.int64).tolist()
        st.session_state.current_model_for_queries = st.session_state.selected_model_key
        status_placeholder_qs.success(f"Prepared {NUM_QUERY_SAMPLES} query samples.")
        return st.session_state.query_vectors, st.session_state.query_identifiers
    except Exception as e:
        status_placeholder_qs.error(f"Failed to prepare query samples: {e}")
        return [], []

# Streamlit UI 
st.title("Milvus Interactive CRUD & Benchmark 📊")

if 'milvus_connected' not in st.session_state:
    st.session_state.milvus_connected = get_milvus_connection()
if not st.session_state.milvus_connected:
    st.error("Cannot proceed without Milvus connection."); st.stop()

st.sidebar.header("⚙️ Configuration")
selected_model_key = st.sidebar.selectbox("Choose Embedding Model:", list(AVAILABLE_MODELS.keys()), key='selected_model_key_selector')

if 'embedding_model' not in st.session_state or st.session_state.get('current_loaded_model_key') != selected_model_key:
    with st.spinner(f"Loading model '{selected_model_key}'... This may take a moment."):
        st.session_state.embedding_model = load_embedding_model(selected_model_key)
        st.session_state.current_loaded_model_key = selected_model_key
        
        if 'milvus_collection' in st.session_state: del st.session_state['milvus_collection']
        if 'query_vectors' in st.session_state: del st.session_state['query_vectors']
        if 'df_initial_data_cache' in st.session_state: del st.session_state['df_initial_data_cache']
        if 'df_initial_data_cache_columns' in st.session_state: del st.session_state['df_initial_data_cache_columns']
        if 'csv_columns' in st.session_state: del st.session_state['csv_columns'] 
        st.session_state.performance_history = {} 

model = st.session_state.embedding_model
if not model: st.error("Embedding model not loaded. Please check sidebar errors and select a model."); st.stop()

collection_name = get_dynamic_collection_name(selected_model_key)
if 'milvus_collection' not in st.session_state or st.session_state.milvus_collection.name != collection_name:
    with st.spinner(f"Setting up Milvus collection '{collection_name}'..."):
        st.session_state.milvus_collection = setup_collection(selected_model_key, model)

collection = st.session_state.milvus_collection
if not collection: st.error(f"Milvus collection '{collection_name}' could not be initialized."); st.stop()

query_vectors, query_identifiers = prepare_query_samples(model)

if 'performance_history' not in st.session_state: st.session_state.performance_history = {}

st.sidebar.header("🚀 Operations")
operation_choices = ["Idle", "Initial Load (10k rows)", "Add Data (1k rows)", "Delete Last N Rows", "Update Row (ip.src)"]
chosen_operation = st.sidebar.selectbox("Select Action:", operation_choices, index=0) # Default to Idle

status_placeholder = st.empty() 

if chosen_operation == "Initial Load (10k rows)":
    if st.sidebar.button("Execute Initial Load & Benchmark"):
        status_placeholder.empty();
        perform_initial_load(collection, model, status_placeholder)
        with st.spinner("Benchmarking after Initial Load..."):
            perf = benchmark_operation(collection, query_vectors, query_identifiers, "Initial Load")
            st.session_state.performance_history[f"{selected_model_key} - Initial Load"] = perf
        st.rerun()

elif chosen_operation == "Add Data (1k rows)":
    if st.sidebar.button("Execute Add Data & Benchmark"):
        status_placeholder.empty();
        perform_add_data(collection, model, status_placeholder)
        with st.spinner("Benchmarking after Adding Data..."):
            perf = benchmark_operation(collection, query_vectors, query_identifiers, "After Add")
            st.session_state.performance_history[f"{selected_model_key} - After Add"] = perf
        st.rerun()

elif chosen_operation == "Delete Last N Rows":
    st.sidebar.subheader("Delete Last N Rows")
    num_to_delete_input = st.sidebar.number_input(f"Number of Rows to Delete:", value=DELETE_LAST_N_ROWS_DEFAULT, min_value=1, step=10)
    if st.sidebar.button("Execute Delete & Benchmark"):
        status_placeholder.empty();
        perform_delete_data(collection, num_to_delete_input, status_placeholder)
        with st.spinner("Benchmarking after Delete..."):
            perf = benchmark_operation(collection, query_vectors, query_identifiers, "After Delete")
            st.session_state.performance_history[f"{selected_model_key} - After Delete"] = perf
        st.rerun()

elif chosen_operation == "Update Row (ip.src)":
    st.sidebar.subheader("Update Specific Row")
    default_update_id = int(query_identifiers[0]) if query_identifiers else 0
    frame_to_update = st.sidebar.number_input("Frame Number to Update:", value=default_update_id, step=1, min_value=0)
    new_ip = st.sidebar.text_input("New ip.src Value:", "11.22.33.44")
    if st.sidebar.button("Execute Update & Benchmark"):
        status_placeholder.empty();
        perform_update_data(collection, model, frame_to_update, new_ip, status_placeholder)
        with st.spinner("Benchmarking after Update..."):
            perf = benchmark_operation(collection, query_vectors, query_identifiers, "After Update")
            st.session_state.performance_history[f"{selected_model_key} - After Update"] = perf
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Collection:** `{collection.name if collection else 'N/A'}`")
if collection:
    try:
        collection.load() 
        st.sidebar.metric("Total Entities", collection.num_entities)
        collection.release()
    except Exception as e:
        st.sidebar.text(f"Entities: Error ({e})")


if st.sidebar.button("Clear Collection Data (for current model)"):
    if collection:
        try:
            collection_name_to_clear = collection.name
            utility.drop_collection(collection_name_to_clear)
            if 'milvus_collection' in st.session_state: del st.session_state['milvus_collection']
            if 'performance_history' in st.session_state: st.session_state.performance_history = {} 
            st.sidebar.success(f"Data cleared from '{collection_name_to_clear}'.")
            st.rerun()
        except Exception as e: st.sidebar.error(f"Failed to clear collection: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("⚠️ Drop ALL Benchmark Collections"):
    with st.spinner("Dropping all related benchmark collections..."):
        if not get_milvus_connection(): st.stop()
        all_cols = utility.list_collections()
        dropped_count = 0
        for col_name in all_cols:
            if col_name.startswith(BASE_COLLECTION_NAME):
                try: utility.drop_collection(col_name); st.sidebar.info(f"Dropped: {col_name}"); dropped_count +=1
                except Exception as e: st.sidebar.error(f"Failed to drop {col_name}: {e}")
        st.sidebar.success(f"Dropped {dropped_count} benchmark collections.")
        if 'milvus_collection' in st.session_state: del st.session_state['milvus_collection']
        if 'performance_history' in st.session_state: st.session_state.performance_history = {}
        st.rerun()

st.header("📊 Performance History")
if st.session_state.performance_history:
    df_perf = pd.DataFrame.from_dict(st.session_state.performance_history, orient='index')
    st.dataframe(df_perf.sort_index())

    if not df_perf.empty:
        current_model_perf_keys = [k for k in df_perf.index if k.startswith(selected_model_key)]
        if current_model_perf_keys:
            df_plot = df_perf.loc[current_model_perf_keys].copy()
           
            op_order_simple = ["Initial Load", "After Add", "After Delete", "After Update"]
            df_plot.index = df_plot.index.str.replace(f"{selected_model_key} - ", "")
            df_plot.index.name = "Operation Stage"
           
            
            ordered_categories = [op for op in op_order_simple if op in df_plot.index]
            if ordered_categories:
                df_plot.index = pd.Categorical(df_plot.index, categories=ordered_categories, ordered=True)
                df_plot.sort_index(inplace=True)

            st.subheader(f"Performance Graphs for '{selected_model_key}'")
            col1, col2, col3 = st.columns(3)
            plot_metrics = {'latency': 'Query Latency (s)', 'cpu': 'Avg CPU Usage (%)', 'memory': 'Avg Peak Memory (MB)'}
            plot_colors = {'latency': 'deepskyblue', 'cpu': 'mediumseagreen', 'memory': 'lightcoral'}
           
            for idx, (metric_key, metric_name) in enumerate(plot_metrics.items()):
                ax_col = [col1, col2, col3][idx % 3]
                with ax_col:
                    fig, ax = plt.subplots()
                    if metric_key in df_plot.columns and not df_plot[metric_key].dropna().empty:
                        df_plot[metric_key].plot(kind='line', marker='o', ax=ax, title=metric_name, color=plot_colors[metric_key])
                        ax.tick_params(axis='x', rotation=45)
                        plt.ylabel(metric_name)
                        st.pyplot(fig)
                    else: st.caption(f"No {metric_key} data to plot.")
        else:
            st.info(f"No performance data yet for model '{selected_model_key}' to plot.")
else:
    st.info("No performance data recorded yet.")
	

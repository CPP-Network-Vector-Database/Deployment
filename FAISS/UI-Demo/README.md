## FAISS Performance Tracker with Streamlit for UI

### Installation

#### Option 1: Automated Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/CPP-Network-Vector-Database/Deployment
   cd streamlit-network-analysis
   ```

2. Make the shell script executable:
   ```bash
   chmod +x run_streamlit_app.sh
   ```

3. Run the setup and execution script:
   ```bash
   ./run_streamlit_app.sh
   ```

#### Option 2: Docker Deployment

1. Make the Docker script executable:
   ```bash
   chmod +x run_docker_app.sh
   ```

2. Run the containerized setup:
   ```bash
   ./run_docker_app.sh
   ```

#### Option 3: Manual Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the dataset path in your application:
   ```python
   csv_path = "path/to/your/network_data.csv"
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Dataset Configuration

#### Custom Dataset Path

Set the `CSV_PATH` environment variable before running:
```bash
export CSV_PATH="/path/to/your/network_data.csv"
./run_streamlit_app.sh
```
*If no dataset is provided, the scripts automatically generate sample network data with in the same format

### Docker Config

#### Container Specifications
- Base Image: Python 3.9 slim
- Port Mapping: 8501 (Streamlit default)
- Volume Mounts: `/app/data` for persistent data storage
- Health Checks: Automatic container health monitoring

#### Docker Compose Support
docker-compose.yml handles:
- Service orchestration
- Volume management
- Network configuration
- Environment variable handling

### Issues faced and how to fix them

1. Port Already in Use:
   ```bash
   # Check for processes using port 8501
   lsof -i :8501
   # Kill the process if needed
   kill -9 <PID>
   ```

2. Dataset Loading Errors:
   - Verify CSV file format and column names
   - Check file permissions and path accessibility
   - Ensure sufficient disk space for processing

3. Memory Issues with Large Datasets:
   - Reduce dataset size or use sampling
   - Increase system memory allocation
   - Enable data chunking in the application

4. Docker Container Issues:
   ```bash
   # Check container status
   docker ps -a
   # View container logs
   docker logs <container_name>
   # Restart container
   docker restart <container_name>
   ```
5. If the app doesn't load:
        1. Verify the application is running on `http://localhost:8501`
        2. Check firewall settings for port 8501
        3. Ensure no proxy or VPN interference

### Configurations

#### Environment Variables

- `CSV_PATH`: Path to network data CSV file
- `STREAMLIT_PORT`: Custom port for the application (default: 8501)
- `DEBUG_MODE`: Enable detailed logging (true/false)
- `MAX_RECORDS`: Limit dataset size for performance

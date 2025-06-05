from langtrace_config import lt

@lt.trace(name="Query Operation")
def query_data():
    print("Querying data from the database...")
    # Simulate query
    return {"status": "success", "operation": "query", "data": ["item1", "item2"]}

if __name__ == "__main__":
    query_data()
    
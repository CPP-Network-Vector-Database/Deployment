from langtrace_config import lt

@lt.trace(name="Insert Operation")
def insert_data():
    print("Inserting data into the database...")
    return {"status": "success", "operation": "insert"}

if __name__ == "__main__":
    insert_data()

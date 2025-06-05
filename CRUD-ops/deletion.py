from langtrace_config import lt

@lt.trace(name="Delete Operation")
def delete_data():
    print("Deleting data from the database...")
    return {"status": "success", "operation": "delete"}

if __name__ == "__main__":
    delete_data()

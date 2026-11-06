import csv, json

def json_to_csv(data: csv, filename: str = "DAFinder_Apprenticeships.csv") -> str:
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data[0].keys())
        for row in data:
            writer.writerow(row.values())
    return f"Data successfully written to {filename}"
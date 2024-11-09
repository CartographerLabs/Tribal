import json
from tqdm import tqdm
from datetime import datetime

file_of_channel_posts = r"D:\merged_data.json"
file_of_usernames = r"D:\accounts.ndjson"

def load_user_map(file_of_usernames):
    user_map = {}
    with open(file_of_usernames, "r", encoding="utf8") as user_json_file:
        for line in user_json_file:
            user_entry = json.loads(line)
            user_map[user_entry["id"]] = user_entry["username"]
    print(f"Loaded {len(user_map)} usernames.")
    return user_map

def get_username_from_id(user_id, user_map):
    
    if user_id == None or user_id == 'None':
        return None

    if int(user_id) not in user_map:
        print(f"Username not found for ID: {user_id}")  # Debug: log missing usernames
    
    
    return user_map[int(user_id)]

def transform_entry(entry, user_map):
    #print(f"Processing entry: {entry}")  # Debug: show entry being processed
    user_id = entry.get("from_id")
    username = get_username_from_id(user_id, user_map)
    if not username:
        print(f"Skipping entry due to missing username for ID: {user_id}")
        return None
    
    dt = datetime.fromisoformat(entry.get("date"))

    # Format the datetime object to the desired format
    formatted_date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    transformed_entry = {
        "post_id": entry.get("post_id"),
        "post": entry.get("message"),
        "replying_to": entry.get("reply_to_msg_id"),
        "username": username,
        "time": formatted_date_str,
    }

    if entry.get("message") == 'None':
        return None

    print(f"Transformed entry: {transformed_entry}")  # Debug: show transformed entry
    return transformed_entry

def transform_data(input_file, output_file):
    user_map = load_user_map(file_of_usernames)
    transformed_data = []

    with open(input_file, 'r') as infile:
        data = json.load(infile)  # Load entire JSON file

    # Initialize tqdm with total number of entries
    for entry in tqdm(data, desc="Processing entries", unit="entry"):
        transformed_entry = transform_entry(entry, user_map)
        if transformed_entry:
            transformed_data.append(transformed_entry)

    print(f"Total transformed entries: {len(transformed_data)}")

    if transformed_data:
        with open(output_file, 'w') as outfile:
            json.dump(transformed_data, outfile, indent=4)
        print(f"Data successfully written to {output_file}")
    else:
        print("No data was transformed and written to the output file.")

# Example usage
output_file = r"D:\output.json"
transform_data(file_of_channel_posts, output_file)

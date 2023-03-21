import pandas as pd
from numpy import nan

# Read in the all_posts_metadata CSV file
all_posts_metadata = pd.read_csv("data/all_posts_metadata.csv")

# List containing all values to be added in a new column of image file paths
image_path = []

# Mapping of the username to the current number of photos matched for that user
# Image file names follow a counter-style pattern of 0 to n images
username_image_counter = {}

# Iterate through all the rows in the CSV
for index, row in all_posts_metadata.iterrows():
    username = row["username"]
    
    # Gets the current image count for the usernmae
    # Returns None if the username doesn't have a counter initialized yet 
    # (encountered for the first time while iterating)
    username_image_count = username_image_counter.get(username)

    if username_image_count is None:
        username_image_counter[username] = 0

    if row["type"] == "Photo":
        image_path.append(f"{username_image_counter[username]}.jpg")
        username_image_counter[username] += 1
    else:
        # No Videos have a file path
        image_path.append(nan)

# Appends a new column to the CSV
all_posts_metadata["image_path"] = image_path

all_posts_metadata.to_csv("data/all_posts_metadata_matched.csv", index=False)
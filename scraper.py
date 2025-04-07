import requests
from bs4 import BeautifulSoup
import os
import re
import json

def sanitize_filename(name, max_length=100):
    #reomove invalid chard
    name = re.sub(r'[\\/:*?"<>|]', '_', name) 
    #trim name if too long
    return name[:max_length]  # Trim filename if too long

def scrape_lastfm_album_art(tag, pages, dataset_folder):
    image_folder = os.path.join(dataset_folder, "images")
    metadata_file = os.path.join(dataset_folder, "metadata.jsonl")
    os.makedirs(image_folder, exist_ok=True)
    
    # load metadata if  file exists
    metadata = []
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = [json.loads(line) for line in f]
   
    #go over pages
    for page in range(pages):
        tag_url = f"https://www.last.fm/tag/{tag}/albums?page={page}"
        response = requests.get(tag_url)
        if response.status_code != 200:
            print(f"Failed to fetch {tag_url}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        albums = soup.find_all('div', class_='resource-list--release-list-item')
        
        for index, album in enumerate(albums):
            # get album name
            album_name_tag = album.find('h3', class_='resource-list--release-list-item-name')
            album_name = album_name_tag.text.strip() if album_name_tag else "Unknown"
            album_name_sanitized = sanitize_filename(album_name)
            
            # get artist name
            artist_tag = album.find('p', class_='resource-list--release-list-item-artist')
            artist_name = artist_tag.text.strip() if artist_tag else "Unknown"
            
            # get img url
            image_tag = album.find('img')
            image_url = image_tag['src'] if image_tag else "No Image"
            
            # if image
            if image_url != "No Image":
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    #save image
                    image_filename = f"{tag}_{index}_{album_name_sanitized}.jpg"
                    image_path = os.path.join(image_folder, image_filename)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_response.content)
                    
                    # save metadata
                    caption = f"An album cover for a {tag} album by {artist_name}, titled '{album_name}'."
                    metadata.append({"file_name": image_filename, "text": caption})
                    
                    print(f"Saved: {image_filename} - {caption}")
    
    # update metadata.jsonl
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Dataset updated at {dataset_folder}")

# list of genres
genres = ["rock", "pop", "hip-hop", "jazz", "metal", "blues", "country", "reggae", "classical", "electronic"]

dataset_folder = "album_dataset"

# number of pages per genre to scrape
pages = 300

for genre in genres:
    scrape_lastfm_album_art(genre, pages, dataset_folder)

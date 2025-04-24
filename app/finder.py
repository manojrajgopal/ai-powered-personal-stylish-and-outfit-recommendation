import os
import requests

class FashionFinder:
    def __init__(self, api_key=None, cx=None):
        self.api_key = api_key or os.getenv('G_API_KEY')
        self.cx = cx or os.getenv('G_CX')

    def search_google_api(self, query, filters, num_results=30):
        search_url = "https://www.googleapis.com/customsearch/v1"
        query += " " + " ".join(filters)
        
        items = []
        for start_index in range(1, num_results + 1, 10):
            params = {
                'q': query,
                'key': self.api_key,
                'cx': self.cx,
                'searchType': 'image',
                'num': 10,
                'start': start_index
            }
            try:
                response = requests.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
                for item in data.get('items', []):
                    items.append({
                        'title': item.get('title', 'Title not found'),
                        'link': item.get('link', 'Link not found'),
                        'description': item.get('snippet', 'Description not found'),
                        'image': item.get('link', 'Image link not found')
                    })
            except requests.exceptions.RequestException as e:
                break
        return items
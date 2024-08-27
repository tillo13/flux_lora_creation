import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def generate_direct_download_link(regular_link):
    file_id = None
    direct_download_url = None

    # Extract file ID from regular Google Drive link
    parsed_url = urlparse(regular_link)
    if 'drive.google.com' in parsed_url.netloc:
        if '/file/d/' in parsed_url.path:
            file_id = parsed_url.path.split('/file/d/')[1].split('/')[0]
        elif '/open' in parsed_url.path:
            file_id = parse_qs(parsed_url.query)['id'][0]

    if not file_id:
        return "Invalid Google Drive link provided."

    session = requests.Session()
    
    # Initial request to get confirmation token
    base_url = "https://drive.google.com/uc?export=download"
    initial_response = session.get(base_url, params={'id': file_id}, allow_redirects=True)

    if initial_response.status_code != 200:
        return "Failed to fetch the file information from Google Drive."

    token = get_confirm_token(initial_response)

    if not token:
        # Parse the HTML to find the necessary fields for confirmation
        soup = BeautifulSoup(initial_response.content, "html.parser")
        
        form = soup.select_one("form#download-form")
        if form:
            print("Form content:\n", form.prettify())
        
        confirm_token_input = soup.select_one("form#download-form input[name='confirm']")
        uuid_input = soup.select_one("form#download-form input[name='uuid']")
        
        at = None
        authuser = None

        if confirm_token_input:
            token = confirm_token_input.get('value')
        
        if uuid_input:
            uuid = uuid_input.get('value')
        
        at_input = soup.select_one("form#download-form input[name='at']")
        if at_input:
            at = at_input.get('value')
        
        authuser_input = soup.select_one("form#download-form input[name='authuser']")
        if authuser_input:
            authuser = authuser_input.get('value')
        
        # Construct final direct download URL
        direct_download_url = (f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                               f"&confirm={token}")
        if uuid:
            direct_download_url += f"&uuid={uuid}"
        if at:
            direct_download_url += f"&at={at}"
        if authuser:
            direct_download_url += f"&authuser={authuser}"
    else:
        direct_download_url = f"{base_url}&confirm={token}&id={file_id}"

    return direct_download_url

if __name__ == '__main__':
    GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1aZeBl8tVLXm9rGGgfXFzKZKKBTh7fJrM/view?usp=drive_link"
    direct_download_link = generate_direct_download_link(GOOGLE_DRIVE_LINK)
    print(f"Direct download link: {direct_download_link}")
import requests
from bs4 import BeautifulSoup
import csv

def extract_invalid_urls(num_pages):
    base_url = 'https://www.phishtank.com/phish_search.php?page={}&valid=n&Search=Search'
    invalid_urls = []

    for page in range(1, num_pages + 1):
        url = base_url.format(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table that contains the phishing URLs
        table = soup.find('table', {'class': 'data'})

        # Extract the URLs and validity labels from the table rows
        rows = table.find_all('tr')

        for row in rows[1:]:  # Skip the table header row
            columns = row.find_all('td')
            url = columns[1].text.strip()
            validity = columns[3].text.strip()

            if validity == 'INVALID':
                invalid_urls.append(url)

    return invalid_urls

def save_urls_to_csv(url_list, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'URL'])
        for url in url_list:
            writer.writerow(['HAM', url])

    print(f'{len(url_list)} URLs saved to {output_file}')

# Set the number of pages to extract
num_pages = 100

# Extract the invalid URLs from all pages of Phish Search
invalid_urls = extract_invalid_urls(num_pages)

# Save the extracted invalid URLs to a CSV file
output_file = 'benign_urls.csv'
save_urls_to_csv(invalid_urls, output_file)

def crawl_valid_phish(num_pages):
    base_url = 'https://www.phishtank.com/phish_search.php?valid=y&active=All&Search=Search'
    phishing_urls = []

    for page in range(1, num_pages + 1):
        url = f'{base_url}&page={page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table that contains the valid phishing URLs
        table = soup.find('table', {'class': 'data'})

        # Extract the URLs from the table rows and label them as SPAM
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip the table header row
            columns = row.find_all('td')
            url = columns[1].text.strip()
            phishing_urls.append((url, 'SPAM'))  # Append the URL with the label 'SPAM'

    return phishing_urls

def save_urls_to_csv(url_list, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'URL'])
        for url, label in url_list:
            writer.writerow([label, url])

    print(f'{len(url_list)} URLs saved to {output_file}')

# Example usage: Crawl Valid Phishing URLs from PhishTank and save them to a CSV file
num_pages = 100
output_file = 'valid_phish_urls.csv'
valid_phish_urls = crawl_valid_phish(num_pages)

save_urls_to_csv(valid_phish_urls, output_file)

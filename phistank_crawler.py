import requests
from bs4 import BeautifulSoup
import csv

def crawl_phishtank(num_pages):
    base_url = 'https://www.phishtank.com/phish_search.php?page='
    phishing_urls = []

    for page in range(1, num_pages + 1):
        url = base_url + str(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table that contains the phishing URLs
        table = soup.find('table', {'class': 'data'})

        # Extract the URLs from the table rows
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip the table header row
            columns = row.find_all('td')
            url = columns[1].text.strip()
            phishing_urls.append(url)

    return phishing_urls

def save_urls_to_csv(url_list, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'URL'])
        for url in url_list:
            writer.writerow(['spam', url])

    print(f'{len(url_list)} URLs saved to {output_file}')

# Example usage: Crawl 10 pages of PhishTank and save the URLs to a CSV file
num_pages = 10
output_file = 'phishing_urls.csv'
urls = crawl_phishtank(num_pages)
save_urls_to_csv(urls, output_file)

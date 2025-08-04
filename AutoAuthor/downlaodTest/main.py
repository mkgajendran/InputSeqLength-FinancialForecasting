from dotenv import load_dotenv
import os
load_dotenv()

import requests
import time
from bs4 import BeautifulSoup
from AutoAuthor.download_pdf.bib2pdf import load_soax_proxy_from_env

def get_final_redirected_url(url):
    """Follow redirects and return the final URL"""
    try:
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make request with allow_redirects=True to follow all redirects
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        
        # Return the final URL after all redirects
        return response.url
    except Exception as e:
        return f"Error: {str(e)}"

def get_page_content(url):
    """Get the HTML content of the webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the prettified HTML
        return soup.prettify()
    except Exception as e:
        return f"Error getting page content: {str(e)}"

def extract_pdf_links(html_content):
    """Extract all PDF links from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    pdf_links = []
    
    # Find all links that contain .pdf
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '.pdf' in href.lower():
            pdf_links.append(href)
    
    # Also check for PDF links in other elements
    for element in soup.find_all(['link', 'meta', 'script']):
        if element.get('href') and '.pdf' in element['href'].lower():
            pdf_links.append(element['href'])
        if element.get('content') and '.pdf' in element['content'].lower():
            pdf_links.append(element['content'])
    
    # Check for citation_pdf_url meta tag specifically
    citation_pdf = soup.find('meta', {'name': 'citation_pdf_url'})
    if citation_pdf and citation_pdf.get('content'):
        pdf_links.append(citation_pdf['content'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pdf_links = []
    for link in pdf_links:
        if link not in seen:
            seen.add(link)
            unique_pdf_links.append(link)
    
    return unique_pdf_links

def nature_download(original_url, final_url, index):
    """Handle Nature article downloads specifically"""
    
    proxy_config = load_soax_proxy_from_env()
    # Always ensure the proxy URL starts with http://
    proxy_server = proxy_config["server"]
    if not proxy_server.startswith("http://") and not proxy_server.startswith("https://"):
        proxy_server = f"http://{proxy_server}"
    proxies = {
        "http": proxy_server,
        "https": proxy_server
    }

    def download_pdf(pdf_url, base_url, filename):
        """Download a PDF file for Nature articles"""
        try:
            # Make URL absolute if it's relative
            if pdf_url.startswith('/'):
                pdf_url = base_url + pdf_url
            elif not pdf_url.startswith('http'):
                pdf_url = base_url + '/' + pdf_url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            print(f"   Downloading: {pdf_url}")
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.get(pdf_url, timeout=60, proxies=proxies, allow_redirects=True)
                response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"   Saved: {filename} ({len(response.content)} bytes)")
            return True
        except Exception as e:
            print(f"   Error downloading {pdf_url}: {str(e)}")
            return False
    
    print(f"   Processing Nature article...")
    
    # Get and save page content
    print(f"   Saving page content to file...")
    page_content = get_page_content(final_url)
    
    # Save to file in same directory as main.py
    filename = f"webpage_content_{index}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(page_content)
    print(f"   Saved to: {filename}")
    print(f"   Content length: {len(page_content)} characters")
    
    # Extract and download PDF links
    print(f"   Extracting PDF links...")
    pdf_links = extract_pdf_links(page_content)
    print(f"   Found {len(pdf_links)} PDF links:")
    for j, link in enumerate(pdf_links, 1):
        print(f"     {j}. {link}")
    
    if pdf_links:
        # Create PDFs directory if it doesn't exist
        import os
        pdf_dir = f"pdfs_{index}"
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Download each PDF
        for j, pdf_link in enumerate(pdf_links, 1):
            # Extract filename from URL
            pdf_filename = pdf_link.split('/')[-1]
            if not pdf_filename.endswith('.pdf'):
                pdf_filename = f"pdf_{j}.pdf"
            
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            download_pdf(pdf_link, final_url, pdf_path)
    
    print("-" * 50)

def main():
    # Read URLs from all.bib file
    try:
        with open('all.bib', 'r') as file:
            urls = [line.strip() for line in file if line.strip() and line.strip().startswith('http')]
    except FileNotFoundError:
        print("Error: all.bib file not found")
        return
    
    # Print the proxy key being used
    proxy_config = load_soax_proxy_from_env()
    proxy_server = proxy_config["server"]
    if not proxy_server.startswith("http://") and not proxy_server.startswith("https://"):
        proxy_server = f"http://{proxy_server}"
    print(f"[DEBUG] Using proxy server: {proxy_server}")

    print(f"Found {len(urls)} URLs in all.bib")
    print("=" * 50)
    
    # Process each URL
    for i, original_url in enumerate(urls, 1):
        print(f"{i}. Original URL: {original_url}")
        
        # Get final redirected URL
        final_url = get_final_redirected_url(original_url)
        print(f"   Final URL: {final_url}")
        
        # Check if it's a Nature article and handle accordingly
        if 'www.nature.com' in final_url:
            nature_download(original_url, final_url, i)
        else:
            print(f"   Not a Nature article, skipping...")
            print("-" * 50)
        
        # Small delay to be respectful to servers
        time.sleep(1)

if __name__ == "__main__":
    main()

from bs4 import BeautifulSoup

def parse_markdown_content(text):
    
    # Parses the content between <markdown> and </markdown> tags from the given text.
    
    soup = BeautifulSoup(text, 'html.parser')
    markdown_tag = soup.find('markdown')
    
    if markdown_tag:
        return markdown_tag.get_text()
    else:
        return ''
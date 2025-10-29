import bibtexparser


def search_bibtex(bib_database='default', author=None, year=None, journal=None, title_words=None):
    if bib_database == 'default':
        with open('references.bib') as bibfile:
            bib_database = bibtexparser.load(bibfile)

    results = []
    for entry in bib_database.entries:
        match = True
        if author and author.lower() not in entry.get('author', '').lower():
            match = False
        if year and year != entry.get('year', ''):
            match = False
        if journal and journal.lower() not in entry.get('journal', '').lower():
            match = False
        if title_words and not all(word.lower() in entry.get('title', '').lower() for word in title_words):
            match = False
        if match:
            results.append(entry)
    return results



def fetch_all_zotero_json(user_id, api_key, limit=100):
    """
    Fetch all Zotero items in JSON format, handling pagination.
    Returns a list of items (dicts).
    """
    all_items = []
    start = 0
    while True:
        url = f'https://api.zotero.org/users/{user_id}/items?format=json&limit={limit}&start={start}'
        headers = {'Zotero-API-Key': api_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        items = response.json()
        if not items:
            break
        all_items.extend(items)
        start += limit
    return all_items

def zotero_json_to_bibtex(items):
    """
    Convert Zotero JSON items to BibTeX entries using bibtexparser.
    Maps Zotero types to standard BibTeX types.
    Returns BibTeX string.
    """
    type_map = {
        'journalArticle': 'article',
        'book': 'book',
        'conferencePaper': 'inproceedings',
        'thesis': 'phdthesis',
        'webpage': 'misc',
        'report': 'techreport',
        # Add more mappings as needed
    }
    bib_entries = []
    for item in items:
        data = item.get('data', {})
        if not data.get('title'):
            continue
        ztype = data.get('itemType', 'misc')
        entry_type = type_map.get(ztype, 'misc')
        entry = {
            'ENTRYTYPE': entry_type,
            'ID': data.get('key', ''),
            'title': data.get('title', ''),
            'year': data.get('date', '')[:4],
            'journal': data.get('publicationTitle', ''),
            'author': ' and '.join([
                f"{c.get('lastName', '')}, {c.get('firstName', '')}"
                for c in data.get('creators', []) if 'lastName' in c and 'firstName' in c
            ]),
        }
        bib_entries.append(entry)
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = bib_entries
    return bibtexparser.dumps(db)

def fetch_and_save_zotero_bibtex(user_id, api_key, save_path):
    """
    Fetch all Zotero items in JSON, convert to BibTeX, and save to a .bib file.
    """
    items = fetch_all_zotero_json(user_id, api_key)
    bibtex_str = zotero_json_to_bibtex(items)
    with open(save_path, 'w') as f:
        f.write(bibtex_str)
import json
import requests
from idd_manuscripts import constants as rfc


zotero_config = rfc.zotero_config


def fetch_zotero_bibtex(user_id, api_key, save_path, limit=100):
    """
    Fetch Zotero items in BibTeX format in batches and save to a .bib file.
    Handles large libraries by paginating requests.
    """
    start = 0
    all_bibtex = ""
    while True:
        url = f'https://api.zotero.org/users/{user_id}/items?format=bib&limit={limit}&start={start}'
        headers = {'Zotero-API-Key': api_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        bibtex_batch = response.text.strip()
        if not bibtex_batch:
            break
        all_bibtex += bibtex_batch + "\n\n"
        # If fewer than 'limit' entries returned, we're done
        if bibtex_batch.count('@') < limit:
            break
        start += limit
    with open(save_path, 'w') as f:
        f.write(all_bibtex)

def fetch_and_save_zotero_items(user_id, api_key, save_path, limit=100):
    """
    Fetch all Zotero items (excluding attachments) and save to a JSON file.
    Returns the list of items.
    """
    all_items = []
    start = 0
    while True:
        url = f'https://api.zotero.org/users/{user_id}/items?format=json&limit={limit}&start={start}'
        headers = {'Zotero-API-Key': api_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        items = response.json()
        # Filter out attachments (e.g., PDFs)
        items = [item for item in items if item.get('data', {}).get('title')]
        if not items:
            break
        all_items.extend(items)
        start += limit
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(all_items, f)
    return all_items

def load_zotero_items_from_file(load_path):
    """
    Load Zotero items from a previously saved JSON file.
    Returns the list of items.
    """
    with open(load_path, 'r') as f:
        return json.load(f)

def load_zotero_config(config_source):
    """
    Load Zotero username and API key from a YAML config file or Python dict.
    Returns (username, api_key)
    """
    if isinstance(config_source, dict):
        username = config_source.get('username')
        api_key = config_source.get('api_key')
        return username, api_key
    else:
        import yaml
        with open(config_source, 'r') as f:
            config = yaml.safe_load(f)
        username = config['zotero']['username']
        api_key = config['zotero']['api_key']
        return username, api_key

def fetch_zotero_items(user_id, api_key, limit=100):
    """
    Fetch all Zotero items using the API, handling pagination.
    Returns a list of items (dicts).
    """
    all_items = []
    start = 0
    while True:
        url = f'https://api.zotero.org/users/{user_id}/items?format=json&limit={limit}&start={start}'
        headers = {'Zotero-API-Key': api_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        items = response.json()
        if not items:
            break
        all_items.extend(items)
        start += limit
    return all_items



# Example usage for default_items:
# user_id, api_key = load_zotero_config(rfc.zotero_config)
# default_items = fetch_zotero_items(user_id, api_key)

def search_bibliography(items=None, author=None, year=None, year_range=None, journal=None, title_words=None):
    """
    Filter Zotero items by author last name, year, year range, journal, or words in title.
    At least one filter must be provided.
    """

    # Use default_items if items is None
    if items is None:
        user_id, api_key = load_zotero_config(rfc.zotero_config)
        items = fetch_zotero_items(user_id, api_key)

    if not any([author, year, year_range, journal, title_words]):
        raise ValueError("Provide at least one filter: author, year, year_range, journal, or title_words.")

    results = []
    for item in items:
        data = item.get('data', {})
        if not data.get('title'):
            continue

        match = True

        # Author filter
        if author:
            authors = [c.get('lastName', '').lower() for c in data.get('creators', []) if 'lastName' in c]
            if not any(author.lower() in a for a in authors):
                match = False

        # Year filter
        if year:
            item_year = data.get('date', '')[:4]
            if not (item_year.isdigit() and int(item_year) == int(year)):
                match = False

        # Year range filter
        if year_range:
            item_year = data.get('date', '')[:4]
            if not (item_year.isdigit() and int(year_range[0]) <= int(item_year) <= int(year_range[1])):
                match = False

        # Journal filter
        if journal:
            journal_name = data.get('publicationTitle', '').lower()
            if journal.lower() not in journal_name:
                match = False

        # Title words filter
        if title_words:
            title = data.get('title', '').lower()
            if not all(word.lower() in title for word in title_words):
                match = False

        if match:
            results.append({
                'key': data.get('key'),
                'title': data.get('title'),
                'authors': ', '.join([f"{c.get('lastName', '')}" for c in data.get('creators', []) if 'lastName' in c]),
                'year': data.get('date', '')[:4],
                'journal': data.get('publicationTitle', '')
            })

    # Print pretty answers
    if results:
        for r in results:
            print(f"{r['key']}: {r['title']}\n  Authors: {r['authors']}\n  Year: {r['year']}\n  Journal: {r['journal']}\n")
    else:
        print("No matching items found.")
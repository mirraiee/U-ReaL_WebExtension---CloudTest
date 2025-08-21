import re
import csv
from urllib.parse import urlparse, urlunparse

class URLFeatureExtractor:
    WHITELIST = set()
    MAX_WHITELIST = 30000
    SUSPICIOUS_KEYWORDS = ['login', 'secure', 'account', 'bank', 'confirm', 'signin', 'money', 'free']

    @staticmethod
    def normalize_domain(domain_or_url):
        parsed = urlparse(domain_or_url)
        netloc = parsed.netloc if parsed.netloc else parsed.path
        netloc = netloc.lower().split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return netloc

    @staticmethod
    def normalize_url(url):
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        path = parsed.path or "/"   # normalize empty path
        if path == "/":             # canonicalize root slash
            path = ""               
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))


    @staticmethod
    def load_whitelist(csv_path):
        whitelist = set()
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= URLFeatureExtractor.MAX_WHITELIST:
                    break
                if len(row) < 2:
                    continue
                url = row[1].strip()
                if url:
                    domain = URLFeatureExtractor.normalize_domain(url)
                    whitelist.add(domain)
        URLFeatureExtractor.WHITELIST = whitelist

    def is_whitelisted(self):
        # Exact domain match
        if self.domain in URLFeatureExtractor.WHITELIST:
            return True

        # Subdomain match: check if whitelist entry is a suffix of the domain
        for w in URLFeatureExtractor.WHITELIST:
            if self.domain == w or self.domain.endswith("." + w):
                return True
        return False


    def __init__(self, url):
        self.url = self.normalize_url(url)
        self.parsed = urlparse(self.url)
        self.domain = self.normalize_domain(self.url)
        self.path = self.parsed.path
        self.tokens_url = re.split(r'\W+', self.url)
        self.tokens_domain = re.split(r'\W+', self.domain)
        self.tokens_path = re.split(r'\W+', self.path)
        self._url_lower = self.url.lower()

    # ===== Token Stats Helper =====
    @staticmethod
    def token_stats(tokens):
        tokens = [t for t in tokens if t]
        if not tokens:
            return 0, 0, 0
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        largest = max(len(t) for t in tokens)
        count = len(tokens)
        return avg_len, count, largest

    # ===== Individual Feature Methods =====
    def URL_length(self):
        return len(self.url)

    def Domain_length(self):
        return len(self.domain)

    def No_of_dots(self):
        return self.url.count('.')
    
    def hyphen_count_url(self):
        return self.url.count('-')

    # Token features
    def avg_token_length(self):
        return self.token_stats(self.tokens_url)[0]

    def token_count(self):
        return self.token_stats(self.tokens_url)[1]

    def largest_token(self):
        return self.token_stats(self.tokens_url)[2]

    def avg_domain_token_length(self):
        return self.token_stats(self.tokens_domain)[0]

    def domain_token_count(self):
        return self.token_stats(self.tokens_domain)[1]

    def largest_domain(self):
        return self.token_stats(self.tokens_domain)[2]

    def avg_path_token(self):
        return self.token_stats(self.tokens_path)[0]

    def path_token_count(self):
        return self.token_stats(self.tokens_path)[1]

    def largest_path(self):
        return self.token_stats(self.tokens_path)[2]

    # Security features
    def sec_sen_word_cnt(self):
        return sum(self._url_lower.count(word) for word in self.SUSPICIOUS_KEYWORDS)

    def IPaddress_presence(self):
        return int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', self.url)))

    def exe_in_url(self):
        return int('.exe' in self._url_lower)

    # Convenience method to extract all features in dictionary
    def extract_features(self):
        return {
            'URL_length': self.URL_length(),
            'Domain_length': self.Domain_length(),
            'No_of_dots': self.No_of_dots(),
            'avg_token_length': self.avg_token_length(),
            'token_count': self.token_count(),
            'largest_token': self.largest_token(),
            'avg_domain_token_length': self.avg_domain_token_length(),
            'domain_token_count': self.domain_token_count(),
            'largest_domain': self.largest_domain(),
            'avg_path_token': self.avg_path_token(),
            'path_token_count': self.path_token_count(),
            'largest_path': self.largest_path(),
            'sec_sen_word_cnt': self.sec_sen_word_cnt(),
            'IPaddress_presence': self.IPaddress_presence(),
            'exe_in_url': self.exe_in_url(),
            'hyphen_count_url': self.hyphen_count_url()
        }

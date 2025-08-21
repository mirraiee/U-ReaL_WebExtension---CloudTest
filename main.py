import pandas as pd
import xgboost as xgb
from feature_extractor import URLFeatureExtractor

# -----------------------------
# Load Trained Model
# -----------------------------
model = xgb.XGBClassifier()
model.load_model("url_xgb_model_v2.json")

# -----------------------------
# Load Whitelist
# -----------------------------
URLFeatureExtractor.load_whitelist("raw_datasets/benign-urls.csv")

# -----------------------------
# Test URLs
# -----------------------------
test_urls = [
    "http://secure-login-update.com",
    "https://www.google.com",
    "http://free-money-now.ru",
    "https://bankofamerica.com",
    "https://www.prydwen.gg/",
    "https://shopee.ph/",
    "https://funky-article-722279.framer.app/",
    "https://special-offers-signup.att.com/#/237/HRApplicationOffer",
    "https://www.geeksforgeeks.org/",
    "https://docs.google.com/document/d/1R-tJAzhftiZeIcn-mMwCbIOOxkjupdfDFvP_dLG1u1g/edit?tab=t.0#heading=h.3718mz9rw9am",
    "https://housitba5.firebaseapp.com/",
    "http://slatteryauctions.com.au",
    "https://www.amazon.com/",
    "https://www.facebook.com/",
    "https://fribbels.github.io/hsr-optimizer#showcase?id=802748532",
    "https://www.paypal.com/ph/home",
    "https://bit.ly/3xyzAbC",
    "https://tinyurl.com/abcd123",
    "https://dappssolver.pages.dev/app/",
    "http://allegro.pl-oferta20382047420.icu",
    "https://www.youtube.com/watch?v=yu9lEPDVn1A",
    "https://www.youtube.com/",
    "https://film.kace.dev",
    "https://dibati.com",
    "https://ln.run/JrHuK",
    "http://192.227.138.203/",
    "https://fixnewupdate.com/down/app/index.php?view=index&amp;id=51caa06880986ef8a58eb492b891de47"
]

# -----------------------------
# Feature order must match training
# -----------------------------
FEATURE_ORDER = [
    'URL_length',
    'Domain_length',
    'No_of_dots',
    'avg_token_length',
    'token_count',
    'largest_token',
    'avg_domain_token_length',
    'domain_token_count',
    'largest_domain',
    'avg_path_token',
    'path_token_count',
    'largest_path',
    'sec_sen_word_cnt',
    'IPaddress_presence',
    'exe_in_url',
    'hyphen_count_url'
]

extractor = URLFeatureExtractor(test_urls[2])
feat_dict = extractor.extract_features()
print(feat_dict)
print("\n🔎 Predictions:")
for url in test_urls:
    extractor = URLFeatureExtractor(url)

    # ✅ If whitelisted, skip prediction and label as benign
    if extractor.is_whitelisted():
        print(f"{url} → 🟢 Benign (whitelisted)")
        continue

    feat_dict = extractor.extract_features()
    if feat_dict is None:
        print(f"{url} → ❌ Feature extraction failed")
        continue

    df = pd.DataFrame([[feat_dict[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    if df.isnull().any().any():
        print(f"{url} → ❌ Found NaNs in features")
        continue

    proba = model.predict_proba(df)[0][1]
    label = "🔴 Malicious" if proba >= 0.3 else "🟢 Benign"
    print(f"{url} → {label} ({proba * 100:.2f}% confidence)")

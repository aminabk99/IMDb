import csv
from pathlib import Path

def load_lexicon(vocab_path: Path, er_path: Path, threshold=0.15):
    lexicon = {}
    with vocab_path.open("r", encoding="utf-8") as v_file, er_path.open("r", encoding="utf-8") as er_file:
        for word, rating_str in zip(v_file, er_file):
            word = word.strip()
            rating = float(rating_str.strip())
            

            if abs(rating) >= threshold: 
                lexicon[word] = rating
    return lexicon

def is_negative_review(text: str, lexicon: dict) -> int:
    """
    Applies the weighted logic: Sums the expected ratings of all words.
    If the total score < 0, flag as negative (0).
    Otherwise, flag as positive (1).
    """
    words = text.split()
    
    total_score = sum(lexicon.get(w, 0.0) for w in words)
    
    if total_score < -0.5:
        return 0 # Negative label
    else:
        return 1 # Positive label

def main():
    dataset_dir = Path("aclImdb")
    vocab_file = dataset_dir / "imdb.vocab"
    er_file = dataset_dir / "imdbEr.txt"
    test_csv = Path("processed") / "test.csv"
    
    print("Loading weighted sentiment lexicon...")
    lexicon = load_lexicon(vocab_file, er_file)
    
    print(f"Loaded {len(lexicon)} words with expected ratings.")
    print("Evaluating test set...")
    
    correct_predictions = 0
    total_reviews = 0
    
    with test_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            if not row or len(row) != 2:
                continue
                
            text, actual_label = row[0], int(row[1])
            predicted_label = is_negative_review(text, lexicon)
            
            if predicted_label == actual_label:
                correct_predictions += 1
            total_reviews += 1
            
    if total_reviews > 0:
        accuracy = (correct_predictions / total_reviews) * 100
        print(f"\nResults:")
        print(f"Total Reviews Evaluated: {total_reviews}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No reviews found in test.csv. Make sure to run reviewer.py first!")

if __name__ == "__main__":
    main()
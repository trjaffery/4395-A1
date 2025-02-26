"""
N-gram Language Model with Preprocessing, Unknown Word Handling,
Smoothing, and Perplexity Calculation (including Interpolation).

Author: Zayd Kazi (zak210006), Saad Makda (sxm210191), Tanzeel Jaffery (txj220028), Maaz Faisal (mxf220053)
Date: 02/26/2025

Instructions:
    1. Place 'train.txt' and 'val.txt' in the same directory as this file.
    2. Run this script (e.g. `python cs4395_assignment01_code_group07.py`).
       - It will read 'train.txt' to build the models.
       - It will then compute perplexity on 'val.txt'
    3. This code demonstrates:
       - Additional text preprocessing (lowercasing, optional handling of punctuation/numbers).
       - Unigram and Bigram models (unsmoothed).
       - Unknown word handling (words occurring <= a threshold replaced by <UNK>).
       - Multiple smoothing strategies (Laplace/Add-1, Add-k).
       - Optional interpolation between unigram and bigram probabilities.
       - Perplexity computation for all the above.

Note:
    - We assume each line in train.txt / val.txt is one "review".
    - Each line is already tokenized by spaces, but we now add a bit more
      robust preprocessing (lowercasing, optional punctuation removal, etc.).
    - For bigram modeling, we insert special start <s> and end </s> tokens
      around each line to capture sentence boundaries.
"""

import math
import sys
import re
from collections import Counter

###############################################
# 1. Reading and Preprocessing
###############################################

def preprocess_text(line):
    """
    Applies simple text preprocessing to a single line:
      1. Lowercasing.
      2. Optionally remove or replace punctuation.
      3. Optionally replace numeric digits with a <NUM> token.

    Arguments:
        line: A string (one line from the corpus).

    Returns:
        A processed string suitable for tokenization.
    """
    # Convert text to lowercase.
    line = line.lower()

    # Replace any sequence of digits with <NUM>
    # (uncomment if you want to treat all digits as a single token)
    line = re.sub(r"\d+", "<NUM>", line)

    # Remove punctuation or replace with space.
    # (Here, we remove them for simplicity; you could also keep punctuation as separate tokens.)
    line = re.sub(r"[^\w\s<UNK><NUM>]", "", line)

    return line.strip()

def read_corpus(filename):
    """
    Reads a file where each line is assumed to be tokenized by spaces.
    Applies some optional further preprocessing (see preprocess_text).
    Returns a list of lists, where each sublist is a tokenized and preprocessed line.

    Arguments:
        filename: Path to the file containing the corpus.

    Returns:
        all_sentences: A list of tokenized lines (list of list of tokens).
    """
    all_sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # First do extra preprocessing
            line = preprocess_text(line)
            # Now tokenize by whitespace
            tokens = line.split()
            if tokens:  # If not empty after cleaning
                all_sentences.append(tokens)
    return all_sentences

def build_vocab(train_sentences, unk_threshold=2):
    """
    Given a list of tokenized sentences (train_sentences),
    build a vocabulary (word -> count) and replace words occurring
    <= unk_threshold times with a special <UNK> token.

    Arguments:
        train_sentences: List of tokenized sentences (each a list of strings).
        unk_threshold  : Words with count <= this threshold get turned into <UNK>.

    Returns:
        vocab      : dict mapping word -> count (for words above threshold)
        word_counts: dict mapping word -> raw count (before <UNK> replacement)
    """
    # Count all word frequencies in the training data
    word_counts = Counter()
    for sentence in train_sentences:
        for w in sentence:
            word_counts[w] += 1

    # Build final vocab: only words with count > unk_threshold remain
    vocab = {}
    for w, c in word_counts.items():
        if c > unk_threshold:
            vocab[w] = c

    return vocab, word_counts

def replace_rare_words(sentences, vocab):
    """
    Replace tokens not in vocab with <UNK> in all sentences.

    Arguments:
        sentences: A list of lists of tokens.
        vocab    : A dictionary containing the known vocabulary (words -> counts).

    Returns:
        new_sentences: A list of lists of tokens after <UNK> replacement.
    """
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for w in sentence:
            if w in vocab:
                new_sentence.append(w)
            else:
                new_sentence.append("<UNK>")
        new_sentences.append(new_sentence)
    return new_sentences

###############################################
# 2. Building N-gram Counts
###############################################

def build_unigram_counts(sentences):
    """
    Build unigram counts from a list of tokenized sentences.

    Arguments:
        sentences: A list of lists of tokens.

    Returns:
        unigram_counts: dict[word] = count of that word
        total_unigrams: int, total number of tokens
    """
    unigram_counts = Counter()
    total_unigrams = 0
    for sentence in sentences:
        for w in sentence:
            unigram_counts[w] += 1
            total_unigrams += 1
    return unigram_counts, total_unigrams

def build_bigram_counts(sentences):
    """
    Build bigram counts from a list of tokenized sentences.
    We add <s> at the start and </s> at the end of each sentence to
    capture sentence boundaries.

    Arguments:
        sentences: A list of lists of tokens.

    Returns:
        bigram_counts: dict of (w1, w2) -> count
        context_counts: dict of w1 -> count (i.e., how many times w1 appears as the first word in a bigram)
    """
    bigram_counts = Counter()
    context_counts = Counter()  # For P(w2|w1), we need counts of w1

    for sentence in sentences:
        # Insert start/end tokens
        mod_sentence = ["<s>"] + sentence + ["</s>"]
        for i in range(len(mod_sentence) - 1):
            w1 = mod_sentence[i]
            w2 = mod_sentence[i+1]
            bigram_counts[(w1, w2)] += 1
            context_counts[w1] += 1

    return bigram_counts, context_counts

###############################################
# 3. Probability Calculation (Smoothing)
###############################################

def unigram_prob(word, unigram_counts, total_unigrams, vocab_size, k=0.0):
    """
    Returns P(word) with add-k smoothing for unigrams.

    Arguments:
        word            : token
        unigram_counts  : dict of word->count
        total_unigrams  : total number of tokens (in training)
        vocab_size      : size of vocabulary (including <UNK>)
        k               : smoothing parameter (0 means none, 1 means Laplace, etc)

    Notes:
        - If k=0, it's the unsmoothed MLE: count(word)/total_unigrams.
        - If k>0, it's add-k smoothing: (count(word)+k)/(total_unigrams + k*vocab_size).
    """
    count_w = unigram_counts.get(word, 0)
    numerator = count_w + k
    denominator = total_unigrams + (k * vocab_size)
    return float(numerator) / float(denominator)

def bigram_prob(w1, w2, bigram_counts, context_counts, vocab_size, k=0.0):
    """
    Returns P(w2|w1) with add-k smoothing for bigrams.

    Arguments:
        w1, w2          : tokens
        bigram_counts   : dict of (w1,w2)->count
        context_counts  : dict of w1->count
        vocab_size      : size of vocabulary (including <UNK>)
        k               : smoothing parameter (0 means none, 1 means Laplace, etc)

    Notes:
        - If k=0, it's the unsmoothed MLE: count(w1,w2)/count(w1).
        - If k>0, it's add-k smoothing: (count(w1,w2)+k)/(count(w1) + k*vocab_size).
    """
    count_w1w2 = bigram_counts.get((w1, w2), 0)
    count_w1 = context_counts.get(w1, 0)
    numerator = count_w1w2 + k
    denominator = count_w1 + (k * vocab_size)
    return float(numerator) / float(denominator)

def bigram_prob_interpolated(
    w1, w2, bigram_counts, context_counts,
    unigram_counts, total_unigrams, vocab_size,
    alpha=0.7, k=0.0):
    """
    Returns an interpolated probability:
        P_interpolated(w2 | w1) = alpha * P_bigram(w2 | w1) + (1-alpha)*P_unigram(w2)

    where:
      P_bigram(w2|w1) uses add-k smoothing,
      P_unigram(w2) uses add-k smoothing,
      alpha is a weight between 0 and 1.

    This helps mitigate data sparsity by incorporating both
    the bigram and unigram probabilities.

    Arguments:
        w1, w2          : tokens
        bigram_counts   : dictionary of (w1,w2) -> bigram count
        context_counts  : dictionary of w1 -> context count
        unigram_counts  : dictionary of w -> unigram count
        total_unigrams  : total number of tokens
        vocab_size      : vocabulary size (including <UNK>)
        alpha           : interpolation weight. alpha=0.7 means 70% bigram, 30% unigram
        k               : add-k smoothing parameter for both bigram and unigram
    """
    # Get the bigram probability with add-k smoothing
    p_bigram = bigram_prob(w1, w2, bigram_counts, context_counts, vocab_size, k)
    # Get the unigram probability with add-k smoothing
    p_unigram = unigram_prob(w2, unigram_counts, total_unigrams, vocab_size, k)

    # Weighted sum for interpolation
    p_interpolated = alpha * p_bigram + (1.0 - alpha) * p_unigram
    return p_interpolated

###############################################
# 4. Perplexity Calculation
###############################################

def compute_perplexity_unigram(sentences, unigram_counts, total_unigrams, vocab_size, k=0.0):
    """
    Compute perplexity for a set of sentences under a (smoothed) unigram model.

    Perplexity = exp( - (1/N) * sum_{all tokens} log P(token) )

    Arguments:
        sentences       : list of token lists
        unigram_counts  : dictionary of unigrams -> count
        total_unigrams  : total number of training tokens
        vocab_size      : vocabulary size (including <UNK>)
        k               : add-k smoothing parameter (0 => unsmoothed, >0 => smoothing)
    """
    log_prob_sum = 0.0
    total_tokens = 0

    for sentence in sentences:
        for w in sentence:
            p = unigram_prob(w, unigram_counts, total_unigrams, vocab_size, k)
            # Add a small constant (1e-15) before log to avoid log(0)
            log_prob_sum += math.log(p + 1e-15)
            total_tokens += 1

    if total_tokens == 0:
        # If the validation file is empty or something went wrong
        return float('inf')

    avg_log_prob = log_prob_sum / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def compute_perplexity_bigram(sentences, bigram_counts, context_counts, vocab_size, k=0.0):
    """
    Compute perplexity for a set of sentences under a (smoothed) bigram model.

    Perplexity = exp( - (1/N) * sum_{all tokens} log P(token|prev_token) )
    Here N is total number of tokens across all sentences (including <s> if desired).

    We insert <s> and </s> around each sentence for evaluation, matching
    how we built the bigram counts.

    Arguments:
        sentences       : list of token lists
        bigram_counts   : dictionary of (w1,w2) -> count
        context_counts  : dictionary of w1 -> count
        vocab_size      : vocabulary size for smoothing
        k               : add-k smoothing parameter (0 => unsmoothed, >0 => smoothing)
    """
    log_prob_sum = 0.0
    total_tokens = 0

    for sentence in sentences:
        mod_sentence = ["<s>"] + sentence + ["</s>"]
        for i in range(len(mod_sentence) - 1):
            w1 = mod_sentence[i]
            w2 = mod_sentence[i+1]
            p = bigram_prob(w1, w2, bigram_counts, context_counts, vocab_size, k)
            log_prob_sum += math.log(p + 1e-15)
            total_tokens += 1

    if total_tokens == 0:
        return float('inf')

    avg_log_prob = log_prob_sum / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def compute_perplexity_bigram_interpolated(
    sentences, bigram_counts, context_counts,
    unigram_counts, total_unigrams, vocab_size,
    alpha=0.7, k=0.0):
    """
    Compute perplexity for a set of sentences under an interpolated
    bigram + unigram model:
        P_interpolated(w2 | w1) = alpha * P_bigram(w2|w1) + (1-alpha)*P_unigram(w2)

    Both bigram and unigram probabilities can have add-k smoothing.

    Perplexity = exp( - (1/N) * sum_{all tokens} log P_interpolated(token|prev_token) )

    Arguments:
        sentences       : list of token lists
        bigram_counts   : dict of (w1,w2)->count
        context_counts  : dict of w1->count
        unigram_counts  : dict of w->count
        total_unigrams  : total number of training tokens
        vocab_size      : vocabulary size (including <UNK>)
        alpha           : interpolation weight for bigram vs. unigram
        k               : add-k smoothing parameter
    """
    log_prob_sum = 0.0
    total_tokens = 0

    for sentence in sentences:
        mod_sentence = ["<s>"] + sentence + ["</s>"]
        for i in range(len(mod_sentence) - 1):
            w1 = mod_sentence[i]
            w2 = mod_sentence[i+1]
            p = bigram_prob_interpolated(
                w1, w2,
                bigram_counts, context_counts,
                unigram_counts, total_unigrams,
                vocab_size, alpha, k
            )
            log_prob_sum += math.log(p + 1e-15)  # protect log(0)
            total_tokens += 1

    if total_tokens == 0:
        return float('inf')

    avg_log_prob = log_prob_sum / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity

###############################################
# 5. Main Flow
###############################################

def main():
    # File names can be switched if using different files
    train_file = "train.txt"
    valid_file = "val.txt"

    print("Reading training data...")
    train_sentences_orig = read_corpus(train_file)

    # -----------------------------------------------------
    # Step A: Build vocabulary and handle <UNK> for training data
    # -----------------------------------------------------
    # Let's use a threshold of <=2 for <UNK> assignment to reduce data sparsity
    # (words appearing <=2 times become <UNK>).
    print("Building vocab and handling <UNK> for training data...")
    vocab, raw_counts = build_vocab(train_sentences_orig, unk_threshold=2)

    # Replace rare words in training data with <UNK>
    train_sentences = replace_rare_words(train_sentences_orig, vocab)

    # -----------------------------------------------------
    # Step B: Build n-gram counts for unigrams & bigrams
    # -----------------------------------------------------
    print("Building unigram and bigram counts...")
    unigram_counts, total_unigrams = build_unigram_counts(train_sentences)
    bigram_counts, context_counts = build_bigram_counts(train_sentences)

    # Our final vocabulary size (including <UNK>).
    # By this point, <UNK> is definitely in the vocabulary.
    vocab_size = len(unigram_counts)

    # -----------------------------------------------------
    # Step C: Read validation data and replace OOV words with <UNK>
    # -----------------------------------------------------
    print("Reading validation data and replacing OOV with <UNK>...")
    valid_sentences_orig = read_corpus(valid_file)
    valid_sentences = replace_rare_words(valid_sentences_orig, vocab)

    # -----------------------------------------------------
    # Step D: Evaluate Perplexities (Different Methods)
    # -----------------------------------------------------
    print("\n==== Perplexities ====")
    # 1) Unsmoothed Unigram
    pp_unigram_unsmoothed = compute_perplexity_unigram(
        valid_sentences, unigram_counts, total_unigrams, vocab_size, k=0.0
    )
    print(f"Unigram (Unsmoothed) Perplexity: {pp_unigram_unsmoothed:.4f}")

    # 2) Unsmoothed Bigram
    pp_bigram_unsmoothed = compute_perplexity_bigram(
        valid_sentences, bigram_counts, context_counts, vocab_size, k=0.0
    )
    print(f"Bigram (Unsmoothed) Perplexity : {pp_bigram_unsmoothed:.4f}")

    # 3) Laplace Smoothing (Add-1) for Unigram
    pp_unigram_laplace = compute_perplexity_unigram(
        valid_sentences, unigram_counts, total_unigrams, vocab_size, k=1.0
    )
    print(f"Unigram (Laplace, k=1) Perplexity: {pp_unigram_laplace:.4f}")

    # 4) Laplace Smoothing (Add-1) for Bigram
    pp_bigram_laplace = compute_perplexity_bigram(
        valid_sentences, bigram_counts, context_counts, vocab_size, k=1.0
    )
    print(f"Bigram (Laplace, k=1) Perplexity : {pp_bigram_laplace:.4f}")

    # 5) Add-k Smoothing with some k != 1 (say k=0.5) for Unigram
    pp_unigram_addk = compute_perplexity_unigram(
        valid_sentences, unigram_counts, total_unigrams, vocab_size, k=0.5
    )
    print(f"Unigram (Add-0.5) Perplexity   : {pp_unigram_addk:.4f}")

    # 6) Add-k Smoothing with some k != 1 (say k=0.5) for Bigram
    pp_bigram_addk = compute_perplexity_bigram(
        valid_sentences, bigram_counts, context_counts, vocab_size, k=0.5
    )
    print(f"Bigram (Add-0.5) Perplexity    : {pp_bigram_addk:.4f}")

    # 7) Interpolation (Bigram+Unigram) with alpha=0.7 and add-0.5
    #    We can tweak alpha and k to see we you get better results.
    alpha = 0.7
    k_smoothing = 0.5
    pp_bigram_interpolated = compute_perplexity_bigram_interpolated(
        valid_sentences,
        bigram_counts,
        context_counts,
        unigram_counts,
        total_unigrams,
        vocab_size,
        alpha=alpha,
        k=k_smoothing
    )
    print(f"Bigram+Unigram Interpolation (alpha={alpha}, k={k_smoothing}) Perplexity: {pp_bigram_interpolated:.4f}")

    print("\nDone.")

if __name__ == "__main__":
    # You can pass arguments via command line if desired,
    # but here we'll just call main() directly.
    main()

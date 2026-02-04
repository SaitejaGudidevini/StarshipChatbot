#!/usr/bin/env python3
"""
Experimental Search Algorithms
------------------------------
New algorithms for testing - NOT integrated into main codebase.
These are alternatives to diagnose search issues like "cofounder" vs "co-founder".

Run via: python menu.py -> Option X (Experimental)
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


class ExperimentalSearch:
    """Experimental search algorithms for debugging"""

    def __init__(self, qa_pairs: List[Dict]):
        self.qa_pairs = qa_pairs
        # Handle both dict and dataclass QAPair objects
        self.questions = [qa.question if hasattr(qa, 'question') else qa.get('question', '') for qa in qa_pairs]
        self.answers = [qa.answer if hasattr(qa, 'answer') else qa.get('answer', '') for qa in qa_pairs]

        # Pre-compute normalized versions
        self.normalized_questions = [self.normalize_text(q) for q in self.questions]
        self.normalized_answers = [self.normalize_text(a) for a in self.answers]

        # Combined Q+A for retrieval
        self.combined_qa = [f"{q} {a}" for q, a in zip(self.questions, self.answers)]
        self.normalized_combined = [self.normalize_text(c) for c in self.combined_qa]

        # Try to load optional libraries
        self._load_optional_libs()

        # Pre-compute lemmas for all answers (expensive but only done once)
        self._precompute_lemmas()

        # Compute IDF (inverse document frequency) for keywords
        self._compute_idf()

    def _compute_idf(self):
        """
        Compute IDF (Inverse Document Frequency) for all stems.
        Rare words get higher IDF, common words get lower IDF.
        This fixes "syra health" matching everything.
        """
        import math

        # Count document frequency for each stem
        self.stem_doc_freq = Counter()
        total_docs = len(self.answers)

        for answer in self.answers:
            # Get unique stems in this document
            answer_norm = self.normalize_text(answer)
            if self.has_nltk:
                stems = set(self.porter.stem(w) for w in answer_norm.split())
            else:
                stems = set(answer_norm.split())

            for stem in stems:
                self.stem_doc_freq[stem] += 1

        # Compute IDF: log(total_docs / (1 + doc_freq))
        self.stem_idf = {}
        for stem, freq in self.stem_doc_freq.items():
            self.stem_idf[stem] = math.log(total_docs / (1 + freq))

        # Also track max IDF for normalization
        self.max_idf = max(self.stem_idf.values()) if self.stem_idf else 1.0

    def get_keyword_idf(self, stem: str) -> float:
        """Get normalized IDF for a keyword stem (0-1 range)."""
        idf = self.stem_idf.get(stem, self.max_idf)  # Unknown words get max IDF
        return idf / self.max_idf if self.max_idf > 0 else 0.5

    def _precompute_lemmas(self):
        """
        Pre-compute lemmas for all answers to avoid calling spaCy in the hot loop.
        This makes fuzzy_substring_match MUCH faster.
        """
        self.answer_lemmas = []
        self.combined_lemmas = []

        if not self.has_spacy:
            print("  ⏭️  Skipping lemma precomputation (no spaCy)")
            return

        print(f"  🔄 Pre-computing lemmas for {len(self.answers)} answers...", end="", flush=True)
        for i, answer in enumerate(self.answers):
            # Truncate to first 2000 chars for speed
            text = answer.lower()[:2000]
            doc = self.nlp(text)
            lemmas = set(token.lemma_ for token in doc if len(token.lemma_) > 2)
            self.answer_lemmas.append(lemmas)

            # Also for combined Q+A
            combined = self.combined_qa[i].lower()[:2000]
            doc = self.nlp(combined)
            c_lemmas = set(token.lemma_ for token in doc if len(token.lemma_) > 2)
            self.combined_lemmas.append(c_lemmas)

            # Progress indicator every 200 items
            if (i + 1) % 200 == 0:
                print(f" {i+1}", end="", flush=True)

        print(f" Done!")

    def _load_optional_libs(self):
        """Load optional libraries for advanced algorithms"""
        # Stemming
        try:
            from nltk.stem import PorterStemmer, SnowballStemmer
            from nltk.tokenize import word_tokenize
            self.porter = PorterStemmer()
            self.snowball = SnowballStemmer('english')
            self.word_tokenize = word_tokenize
            self.has_nltk = True
            print("  ✅ NLTK stemming available")
        except ImportError:
            self.has_nltk = False
            print("  ⚠️  NLTK not installed (pip install nltk)")

        # Fuzzy matching
        try:
            from rapidfuzz import fuzz, process
            self.fuzz = fuzz
            self.fuzz_process = process
            self.has_fuzzy = True
            print("  ✅ RapidFuzz available")
        except ImportError:
            try:
                from fuzzywuzzy import fuzz, process
                self.fuzz = fuzz
                self.fuzz_process = process
                self.has_fuzzy = True
                print("  ✅ FuzzyWuzzy available")
            except ImportError:
                self.has_fuzzy = False
                print("  ⚠️  Fuzzy matching not installed (pip install rapidfuzz)")

        # Lemmatization
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
            self.has_spacy = True
            print("  ✅ spaCy lemmatization available")
        except:
            self.has_spacy = False
            print("  ⚠️  spaCy not available")

    def normalize_text(self, text: str) -> str:
        """
        Normalize text to handle variations like:
        - cofounder / co-founder / co founder
        - CEO / ceo
        - healthcare / health care / health-care
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove hyphens (co-founder -> cofounder)
        text = text.replace('-', '')

        # Normalize whitespace
        text = ' '.join(text.split())

        # Remove common punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())

        return text

    def stem_text(self, text: str, stemmer='porter') -> str:
        """Stem text using NLTK"""
        if not self.has_nltk:
            return text

        tokens = self.word_tokenize(text.lower())
        if stemmer == 'porter':
            stemmed = [self.porter.stem(t) for t in tokens]
        else:
            stemmed = [self.snowball.stem(t) for t in tokens]
        return ' '.join(stemmed)

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using spaCy"""
        if not self.has_spacy:
            return text

        doc = self.nlp(text.lower())
        return ' '.join([token.lemma_ for token in doc])

    def char_ngrams(self, text: str, n: int = 3) -> set:
        """Generate character n-grams"""
        text = text.lower().replace(' ', '')
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # SEARCH ALGORITHMS
    # =========================================================================

    def normalized_search(self, query: str, top_k: int = 10) -> List[Tuple[float, int, str]]:
        """
        Search using normalized text (removes hyphens, lowercases, etc.)
        This should fix "cofounder" vs "co-founder" issues.
        """
        norm_query = self.normalize_text(query)
        query_tokens = set(norm_query.split())

        scores = []
        for i, norm_q in enumerate(self.normalized_questions):
            q_tokens = set(norm_q.split())

            # Jaccard similarity on tokens
            jaccard = self.jaccard_similarity(query_tokens, q_tokens)

            # Substring match bonus
            substring_bonus = 0.2 if norm_query in norm_q or norm_q in norm_query else 0

            # Exact token overlap ratio
            overlap = len(query_tokens & q_tokens) / max(len(query_tokens), 1)

            score = 0.4 * jaccard + 0.4 * overlap + 0.2 * substring_bonus
            scores.append((score, i, self.questions[i]))

        scores.sort(reverse=True)
        return scores[:top_k]

    def stemmed_search(self, query: str, top_k: int = 10) -> List[Tuple[float, int, str]]:
        """
        Search using stemmed text (founder, founded, founding -> found)
        """
        if not self.has_nltk:
            print("  NLTK not available - using normalized search instead")
            return self.normalized_search(query, top_k)

        stemmed_query = self.stem_text(query)
        query_stems = set(stemmed_query.split())

        scores = []
        for i, q in enumerate(self.questions):
            stemmed_q = self.stem_text(q)
            q_stems = set(stemmed_q.split())

            jaccard = self.jaccard_similarity(query_stems, q_stems)
            overlap = len(query_stems & q_stems) / max(len(query_stems), 1)

            score = 0.5 * jaccard + 0.5 * overlap
            scores.append((score, i, q))

        scores.sort(reverse=True)
        return scores[:top_k]

    def fuzzy_search(self, query: str, top_k: int = 10) -> List[Tuple[float, int, str]]:
        """
        Fuzzy string matching (handles typos, slight variations)
        """
        if not self.has_fuzzy:
            print("  Fuzzy matching not available")
            return []

        # Use process.extract for efficient batch matching
        results = self.fuzz_process.extract(
            query,
            self.questions,
            scorer=self.fuzz.token_set_ratio,
            limit=top_k
        )

        return [(score/100, self.questions.index(match), match) for match, score, _ in results]

    def char_ngram_search(self, query: str, n: int = 3, top_k: int = 10) -> List[Tuple[float, int, str]]:
        """
        Character n-gram similarity (robust to small variations)
        """
        query_ngrams = self.char_ngrams(query, n)

        scores = []
        for i, q in enumerate(self.questions):
            q_ngrams = self.char_ngrams(q, n)
            score = self.jaccard_similarity(query_ngrams, q_ngrams)
            scores.append((score, i, q))

        scores.sort(reverse=True)
        return scores[:top_k]

    def combined_experimental_search(self, query: str, top_k: int = 10, search_engine=None) -> List[Dict]:
        """
        Combine ALL methods: experimental + cosine + BM25.
        Final score = average of non-zero scores (zeros don't drag down).
        """
        print(f"\n  Running all algorithms...")

        # Experimental methods
        print(f"    [1/7] Normalized search...")
        normalized = {idx: score for score, idx, _ in self.normalized_search(query, 100)}

        print(f"    [2/7] Stemmed search...")
        stemmed = {idx: score for score, idx, _ in self.stemmed_search(query, 100)} if self.has_nltk else {}

        print(f"    [3/7] Fuzzy search...")
        fuzzy = {idx: score for score, idx, _ in self.fuzzy_search(query, 100)} if self.has_fuzzy else {}

        print(f"    [4/7] N-gram search...")
        ngram = {idx: score for score, idx, _ in self.char_ngram_search(query, 3, 100)}

        print(f"    [5/7] Retrieve-then-rerank...")
        rtr_results = self.retrieve_then_rerank(query, top_k=2000)  # Get all matches
        retrieve_rerank = {r['idx']: r['final_score'] for r in rtr_results}

        # Cosine similarity (semantic) and BM25 from main search engine
        cosine = {}
        bm25 = {}
        if search_engine:
            print(f"    [6/7] Cosine similarity (semantic)...")
            try:
                sem_result = search_engine.similarity_search(query, top_k=100)
                if sem_result.get('candidates'):
                    scores_list = sem_result.get('scores', [])
                    for i, item in enumerate(sem_result['candidates']):
                        # Find index in our qa_pairs
                        q = item.question if hasattr(item, 'question') else item.get('question', '')
                        for idx, our_q in enumerate(self.questions):
                            if our_q == q:
                                cosine[idx] = scores_list[i] if i < len(scores_list) else 0
                                break
            except Exception as e:
                print(f"      (cosine failed: {e})")

            print(f"    [7/7] BM25 keyword search...")
            try:
                if hasattr(search_engine, 'bm25') and search_engine.bm25:
                    bm25_scores = search_engine.bm25.get_scores(query.lower().split())
                    # Normalize BM25 scores to 0-1
                    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
                    for idx, score in enumerate(bm25_scores):
                        if score > 0:
                            bm25[idx] = score / max_bm25
            except Exception as e:
                print(f"      (BM25 failed: {e})")
        else:
            print(f"    [6/7] Cosine - skipped (no search_engine)")
            print(f"    [7/7] BM25 - skipped (no search_engine)")

        # Combine all indices
        all_indices = (set(normalized.keys()) | set(stemmed.keys()) |
                       set(fuzzy.keys()) | set(ngram.keys()) |
                       set(retrieve_rerank.keys()) | set(cosine.keys()) | set(bm25.keys()))

        combined = []
        for idx in all_indices:
            scores = {
                'norm': normalized.get(idx, 0),
                'stem': stemmed.get(idx, 0),
                'fuzz': fuzzy.get(idx, 0),
                'ngrm': ngram.get(idx, 0),
                'rtr': retrieve_rerank.get(idx, 0),
                'cos': cosine.get(idx, 0),
                'bm25': bm25.get(idx, 0),
            }

            # Final score = average of ALL scores (zeros count against!)
            all_scores = list(scores.values())
            num_algos = len([v for v in all_scores if v > 0])

            # Require at least 2 algorithms to agree
            if num_algos < 2:
                continue

            # Average ALL 7 scores - zeros drag down the average
            final_score = sum(all_scores) / len(all_scores)

            combined.append({
                'idx': idx,
                'question': self.questions[idx],
                'answer': self.answers[idx][:200] + '...' if len(self.answers[idx]) > 200 else self.answers[idx],
                'final_score': final_score,
                'num_algos': num_algos,
                **scores
            })

        combined.sort(key=lambda x: x['final_score'], reverse=True)
        return combined[:top_k]

    # =========================================================================
    # RETRIEVE-THEN-RERANK (THE BIG FIX)
    # =========================================================================

    def get_stopwords(self) -> set:
        """
        Get stopwords: NLTK base + custom additions for Q&A search.
        """
        # Custom stopwords for question-answering (always included)
        custom_stopwords = {
            # Question words
            'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose',
            # Common verbs in questions
            'tell', 'know', 'explain', 'describe', 'discuss', 'give', 'show',
            'please', 'help', 'need', 'want', 'looking', 'find', 'get', 'got',
            # Pronouns and determiners
            'me', 'about', 'some', 'any', 'many', 'much', 'more', 'most',
            'other', 'another', 'such', 'own', 'same', 'different',
            # Common filler words
            'really', 'actually', 'basically', 'simply', 'just', 'even',
            'also', 'too', 'very', 'quite', 'rather', 'well', 'still',
        }

        # Try to get NLTK stopwords and merge
        nltk_stopwords = set()
        if self.has_nltk:
            try:
                from nltk.corpus import stopwords
                nltk_stopwords = set(stopwords.words('english'))
            except:
                try:
                    import nltk
                    nltk.download('stopwords', quiet=True)
                    from nltk.corpus import stopwords
                    nltk_stopwords = set(stopwords.words('english'))
                except:
                    pass

        # Merge both sets
        return nltk_stopwords | custom_stopwords

    def extract_keywords(self, query: str, use_stemming: bool = True) -> List[Tuple[str, List[str]]]:
        """
        Extract meaningful keywords from query.
        Returns list of (original_word, [list_of_stems]) tuples.

        For words like "founder", expands to multiple related stems:
        - "founder" -> ["founder", "cofound", "found"] to match co-founder, founded, etc.
        """
        stopwords = self.get_stopwords()

        # Normalize and split
        normalized = self.normalize_text(query)
        words = normalized.split()

        # Filter out stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Return (original, [stems]) pairs
        result = []
        for kw in keywords:
            if use_stemming and self.has_nltk:
                primary_stem = self.porter.stem(kw)
                stems = [primary_stem]

                # Expand to related stems for common variations
                related_stems = self._get_related_stems(kw, primary_stem)
                stems.extend(related_stems)
                stems = list(set(stems))  # Dedupe
            else:
                stems = [kw]
            result.append((kw, stems))

        return result

    def _get_related_stems(self, word: str, primary_stem: str) -> List[str]:
        """
        Get related stems for a word - CONSERVATIVE approach.
        Only add stems that are clearly the same concept.
        DO NOT add ambiguous stems like "found" which matches "find" past tense.
        """
        related = []

        # Only expand for hyphenated variants - NOTHING ELSE
        # cofounder <-> co-founder (same word, different spelling)
        word_lower = word.lower()

        # Handle co-/coXXX variants only
        if word_lower.startswith('co') and len(word_lower) > 4:
            # cofounder -> also match cofound (the stem of co-founder)
            base = word_lower[2:]  # Remove 'co'
            if self.has_nltk:
                related.append('co' + self.porter.stem(base))  # cofound
            # DO NOT add base stem alone - too risky (founder != found)

        return related

    def generate_keyword_variants(self, keyword: str) -> List[str]:
        """
        Generate variants of a keyword to handle hyphenation and common variations.
        E.g., "cofounder" -> ["cofounder", "co-founder", "co founder", "founder"]
        """
        variants = [keyword]

        # If word could be hyphenated, generate variants
        # Common prefixes that might be hyphenated
        prefixes = ['co', 'pre', 'post', 'non', 'self', 'ex', 'anti', 'semi', 'multi', 'sub']

        for prefix in prefixes:
            if keyword.startswith(prefix) and len(keyword) > len(prefix):
                base = keyword[len(prefix):]
                # Add hyphenated version: cofounder -> co-founder
                variants.append(f"{prefix}-{base}")
                # Add spaced version: cofounder -> co founder
                variants.append(f"{prefix} {base}")
                # Add just the base: cofounder -> founder
                if len(base) > 3:
                    variants.append(base)

        # If keyword contains hyphen, add without hyphen
        if '-' in keyword:
            variants.append(keyword.replace('-', ''))
            variants.append(keyword.replace('-', ' '))

        return list(set(variants))

    def stem_match(self, keyword_stem: str, text: str) -> bool:
        """
        Check if any word in text shares the same stem as keyword.
        This is the key insight: "founder", "co-founder", "founded" all stem to "found".
        """
        if not self.has_nltk:
            return keyword_stem.lower() in text.lower()

        # Stem all words in text and check for match
        text_normalized = self.normalize_text(text)
        text_words = text_normalized.split()
        text_stems = [self.porter.stem(w) for w in text_words]

        return keyword_stem in text_stems

    def fuzzy_substring_match(self, keyword: str, keyword_stem: str, text: str, threshold: int = 80, text_idx: int = None, use_combined: bool = False) -> float:
        """
        Check if keyword appears in text using TIERED strategies.
        Returns a SCORE (0.0 to 1.0) not just True/False.

        Tier 1: EXACT substring match (cofounder, co-founder, founder) → 1.0
        Tier 2: Lemmatization match (spaCy) → 0.8  (uses pre-computed lemmas if text_idx provided)
        Tier 3: Fuzzy substring match → 0.6 * (fuzzy_score/100)
        Tier 4: Stem match (risky) → 0.3

        Args:
            text_idx: If provided, use pre-computed lemmas for that index (MUCH faster)
            use_combined: If True and text_idx provided, use combined Q+A lemmas
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()

        # TIER 1: Exact substring match - HIGHEST PRIORITY
        variants = self.generate_keyword_variants(keyword)
        for variant in variants:
            if variant.lower() in text_lower:
                return 1.0  # Exact match!

        # TIER 2: Lemmatization match using spaCy
        if self.has_spacy:
            keyword_doc = self.nlp(keyword_lower)
            keyword_lemmas = set(token.lemma_ for token in keyword_doc if len(token.lemma_) > 2)

            # Use pre-computed lemmas if available (MUCH faster!)
            if text_idx is not None and hasattr(self, 'answer_lemmas') and self.answer_lemmas:
                if use_combined and self.combined_lemmas:
                    text_lemmas = self.combined_lemmas[text_idx]
                else:
                    text_lemmas = self.answer_lemmas[text_idx]
            else:
                # Fallback: compute on the fly (slow)
                text_doc = self.nlp(text_lower[:2000])
                text_lemmas = set(token.lemma_ for token in text_doc if len(token.lemma_) > 2)

            # Check if keyword lemmas appear in text lemmas
            if keyword_lemmas & text_lemmas:
                return 0.8  # Lemma match

        # TIER 3: Fuzzy substring match
        if self.has_fuzzy:
            best_fuzzy = 0
            for variant in variants:
                score = self.fuzz.partial_ratio(variant.lower(), text_lower)
                best_fuzzy = max(best_fuzzy, score)
            if best_fuzzy >= threshold:
                return 0.6 * (best_fuzzy / 100)  # Scale fuzzy score

        # TIER 4: Stem match - LOWEST PRIORITY (risky, can match wrong things)
        if self.stem_match(keyword_stem, text):
            return 0.3  # Low score for stem-only match

        return 0.0  # No match

    def answer_keyword_retrieval(self, query: str, min_keywords_match: int = 1) -> List[Dict]:
        """
        STAGE 1: Retrieve all Q&As where the ANSWER contains query keywords.
        This is the "cast wide net" phase - find everything potentially relevant.

        Uses stemming so "cofounder" matches "Co-Founder", "founded", etc.
        Uses IDF weighting so rare keywords (like "founder") matter more than common ones (like "syra").

        Returns list of dicts with idx, question, answer, matched_keywords, match_count, idf_score
        """
        keywords = self.extract_keywords(query, use_stemming=True)

        if not keywords:
            # If no keywords extracted, fall back to normalized query
            norm_q = self.normalize_text(query)
            keywords = [(norm_q, [self.porter.stem(norm_q) if self.has_nltk else norm_q])]

        # Compute IDF for each keyword (use best IDF from expanded stems)
        keyword_idfs = {}
        for orig, stems in keywords:
            # Get best (lowest doc freq = highest IDF) among all stems
            best_idf = max(self.get_keyword_idf(s) for s in stems)
            keyword_idfs[orig] = best_idf

        # Sort keywords by IDF (rarest first) for display
        sorted_kws = sorted(keywords, key=lambda x: keyword_idfs[x[0]], reverse=True)
        print(f"\n  [Retrieval] Keywords (sorted by rarity):")
        for orig, stems in sorted_kws:
            idf = keyword_idfs[orig]
            # Show doc freq for each stem
            stem_info = ', '.join(f"{s}({self.stem_doc_freq.get(s, 0)})" for s in stems[:3])
            print(f"    '{orig}' -> stems: [{stem_info}] - IDF: {idf:.3f}")

        matches = []
        total_answers = len(self.answers)
        for idx, answer in enumerate(self.answers):
            # Progress indicator every 500 answers
            if idx % 500 == 0 and idx > 0:
                print(f"    ... processing {idx}/{total_answers} answers", flush=True)

            matched_keywords = []
            matched_idfs = []
            matched_scores = []  # Track HOW WELL each keyword matched
            combined = self.combined_qa[idx]

            for orig_kw, stems in keywords:
                # Check if ANY of the stems match - get best score
                best_score = 0.0
                for stem in stems:
                    # Pass text_idx for pre-computed lemmas (MUCH faster!)
                    score = self.fuzzy_substring_match(orig_kw, stem, answer, threshold=75, text_idx=idx, use_combined=False)
                    if score > best_score:
                        best_score = score
                    if score == 0:  # Try combined if answer didn't match
                        score = self.fuzzy_substring_match(orig_kw, stem, combined, threshold=75, text_idx=idx, use_combined=True)
                        if score > best_score:
                            best_score = score

                if best_score > 0:
                    matched_keywords.append(orig_kw)
                    matched_idfs.append(keyword_idfs[orig_kw])
                    matched_scores.append(best_score)  # Store match quality

            if len(matched_keywords) >= min_keywords_match:
                # Calculate IDF-weighted coverage
                total_idf = sum(keyword_idfs.values())
                matched_idf = sum(matched_idfs)
                idf_weighted_coverage = matched_idf / total_idf if total_idf > 0 else 0

                # Match quality: average of how well each keyword matched (1.0=exact, 0.3=stem only)
                match_quality = sum(matched_scores) / len(matched_scores) if matched_scores else 0

                matches.append({
                    'idx': idx,
                    'question': self.questions[idx],
                    'answer': self.answers[idx][:200] + '...' if len(self.answers[idx]) > 200 else self.answers[idx],
                    'full_answer': self.answers[idx],
                    'matched_keywords': matched_keywords,
                    'matched_idfs': matched_idfs,
                    'matched_scores': matched_scores,
                    'match_count': len(matched_keywords),
                    'keyword_coverage': len(matched_keywords) / len(keywords) if keywords else 0,
                    'idf_score': idf_weighted_coverage,
                    'match_quality': match_quality  # NEW: 1.0=exact, 0.8=lemma, 0.6=fuzzy, 0.3=stem
                })

        # Sort by match quality FIRST, then IDF (exact matches beat stem matches)
        matches.sort(key=lambda x: (x['match_quality'], x['idf_score']), reverse=True)

        print(f"  [Retrieval] Found {len(matches)} candidates matching keywords")
        return matches

    def retrieve_then_rerank(self, query: str, top_k: int = 10, semantic_encoder=None) -> List[Dict]:
        """
        THE BIG FIX: Retrieve-Then-Rerank Algorithm

        Stage 1: RETRIEVE - Cast wide net using keyword matching in answers (with stemming!)
        Stage 2: RERANK - Use multiple signals to score candidates

        This fixes the "cofounder" problem because:
        1. Stemming: "cofounder" stems to "cofound", matches "Co-Founder"
        2. We search in ANSWERS, not just questions
        3. Multiple ranking signals avoid semantic embedding failures
        """
        print(f"\n{'='*60}")
        print(f"RETRIEVE-THEN-RERANK: '{query}'")
        print(f"{'='*60}")

        # STAGE 1: RETRIEVE
        candidates = self.answer_keyword_retrieval(query, min_keywords_match=1)

        if not candidates:
            print("  [Warning] No keyword matches found - falling back to combined search")
            return self.combined_experimental_search(query, top_k)

        # STAGE 2: RERANK with multiple signals
        print(f"\n  [Rerank] Scoring {len(candidates)} candidates...")

        # Get the stemmed keywords for checking where matches occur
        query_keyword_stems = set()
        keywords = self.extract_keywords(query, use_stemming=True)
        for orig, stems in keywords:
            query_keyword_stems.update(stems)  # Add all stems

        reranked = []
        for c in candidates:
            idx = c['idx']
            question = self.questions[idx]
            answer = c['full_answer']

            # Signal 1: Keyword coverage (0-1)
            keyword_score = c['keyword_coverage']

            # Signal 2: WHERE do keywords appear? Answer > Question
            # This is THE KEY FIX: "cofounder" in "Co-Founder of Syra Health" should rank high
            answer_keyword_hit = 0
            question_keyword_hit = 0
            answer_norm = self.normalize_text(answer)
            question_norm = self.normalized_questions[idx]

            if self.has_nltk:
                answer_stems = set(self.porter.stem(w) for w in answer_norm.split())
                question_stems = set(self.porter.stem(w) for w in question_norm.split())
            else:
                answer_stems = set(answer_norm.split())
                question_stems = set(question_norm.split())

            # Check how many query keywords appear in answer vs question
            for stem in query_keyword_stems:
                if stem in answer_stems:
                    answer_keyword_hit += 1
                if stem in question_stems:
                    question_keyword_hit += 1

            # Normalize by keyword count
            num_keywords = len(query_keyword_stems) or 1
            answer_keyword_score = answer_keyword_hit / num_keywords
            question_keyword_score = question_keyword_hit / num_keywords

            # Signal 3: Question similarity using stemmed tokens (all query words, not just keywords)
            query_stems = set()
            if self.has_nltk:
                for w in self.normalize_text(query).split():
                    query_stems.add(self.porter.stem(w))
            else:
                query_stems = set(self.normalize_text(query).split())

            question_sim = self.jaccard_similarity(query_stems, question_stems)

            # Signal 4: Answer relevance - Jaccard of all query stems with answer stems
            answer_relevance = self.jaccard_similarity(query_stems, answer_stems)

            # Signal 5: Keyword density in answer - how prominent is the keyword?
            # Count occurrences of keyword stems in answer
            keyword_density = 0
            for stem in query_keyword_stems:
                keyword_density += sum(1 for w in answer_norm.split() if (self.porter.stem(w) if self.has_nltk else w) == stem)
            # Normalize by answer length (longer answers naturally have more occurrences)
            answer_len = len(answer_norm.split()) or 1
            keyword_prominence = min(1.0, keyword_density / (answer_len * 0.05))  # Expect ~5% to get 1.0

            # Signal 6: N-gram similarity for fuzzy matching (reduced weight)
            query_ngrams = self.char_ngrams(query, 3)
            q_ngrams = self.char_ngrams(question, 3)
            a_ngrams = self.char_ngrams(answer[:500], 3)
            ngram_q_score = self.jaccard_similarity(query_ngrams, q_ngrams)
            ngram_a_score = self.jaccard_similarity(query_ngrams, a_ngrams)
            ngram_score = max(ngram_q_score, ngram_a_score)

            # Signal 7: Fuzzy string similarity (reduced weight - was causing false positives)
            fuzzy_score = 0
            if self.has_fuzzy:
                fuzzy_q = self.fuzz.token_set_ratio(query, question) / 100
                fuzzy_a = self.fuzz.token_set_ratio(query, answer[:500]) / 100
                fuzzy_score = max(fuzzy_q, fuzzy_a)

            # Signal 8: Semantic similarity (if encoder provided)
            semantic_score = 0
            if semantic_encoder:
                try:
                    q_emb = semantic_encoder.encode([query])[0]
                    doc_emb = semantic_encoder.encode([question + " " + answer[:500]])[0]
                    semantic_score = float(np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)))
                except:
                    pass

            # Signal 9: IDF-weighted keyword score (rare keywords matter more!)
            idf_score = c.get('idf_score', keyword_score)

            # Combined score with weights
            # No special bonus for Q+A both matching - that causes false positives
            # (e.g., "found" in question matching "founder" in answer)
            final_score = (
                0.30 * idf_score +               # Rare keyword matches matter most
                0.25 * answer_keyword_score +    # Keyword in answer
                0.20 * keyword_prominence +      # How prominent is keyword in answer?
                0.10 * question_sim +            # Question structure similarity
                0.05 * answer_relevance +        # Overall answer relevance
                0.05 * ngram_score +             # Character-level
                0.05 * fuzzy_score               # Fuzzy match
                # NOTE: removed question_keyword_score - was causing false positives
            )

            reranked.append({
                'idx': idx,
                'question': question,
                'answer': c['answer'],
                'final_score': final_score,
                'idf_score': idf_score,
                'keyword_score': keyword_score,
                'answer_keyword_score': answer_keyword_score,
                'keyword_prominence': keyword_prominence,
                'question_sim': question_sim,
                'answer_relevance': answer_relevance,
                'ngram_score': ngram_score,
                'fuzzy_score': fuzzy_score,
                'semantic_score': semantic_score,
                'matched_keywords': c['matched_keywords']
            })

        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)

        return reranked[:top_k]

    def test_retrieve_rerank(self, query: str, semantic_encoder=None):
        """
        Interactive test for retrieve-then-rerank with detailed output.
        """
        results = self.retrieve_then_rerank(query, top_k=10, semantic_encoder=semantic_encoder)

        print(f"\n  Top 10 Results:")
        print(f"  {'-'*80}")

        if tabulate:
            rows = []
            for r in results[:10]:
                rows.append([
                    f"{r['final_score']:.3f}",
                    f"{r.get('answer_keyword_score', 0):.2f}",
                    f"{r.get('keyword_prominence', 0):.2f}",
                    f"{r['fuzzy_score']:.2f}",
                    ', '.join(r['matched_keywords'][:2]),
                    r['question'][:35] + '...'
                ])
            print(tabulate(rows, headers=['Score', 'AnsKW', 'Prom', 'Fuzz', 'Matched', 'Question'], tablefmt='simple'))
        else:
            for i, r in enumerate(results[:10]):
                print(f"  {i+1}. [{r['final_score']:.3f}] {r['question'][:60]}")
                print(f"       Keywords: {r['matched_keywords']}")
                print(f"       Scores: ans_kw={r.get('answer_keyword_score', 0):.2f}, prom={r.get('keyword_prominence', 0):.2f}, fuzz={r['fuzzy_score']:.2f}")

        return results

    # =========================================================================
    # MENU
    # =========================================================================

    def menu(self, search_engine):
        """Interactive menu for experimental algorithms"""
        self._search_engine = search_engine  # Store for use in combined search
        while True:
            print(f"\n{'='*60}")
            print("EXPERIMENTAL ALGORITHMS")
            print(f"{'='*60}")
            print("\n[1] Normalized Search (fixes hyphen issues)")
            print("[2] Stemmed Search (NLTK Porter/Snowball)")
            print("[3] Fuzzy Search (handles typos)")
            print("[4] Character N-gram Search")
            print("[5] *** COMBINED (all 7 algos: RTR+Cos+BM25+Norm+Stem+Fuzz+Ngrm) ***")
            print("[6] Compare: Original vs Experimental")
            print("[7] Analyze specific question pair")
            print("")
            print("[R] Retrieve-then-rerank only")
            print("[T] Test keyword extraction")
            print("[B] Back to main menu")

            choice = input("\nChoice: ").strip().upper()

            if choice == '1':
                self._run_search("Normalized", self.normalized_search)
            elif choice == '2':
                self._run_search("Stemmed", self.stemmed_search)
            elif choice == '3':
                self._run_search("Fuzzy", self.fuzzy_search)
            elif choice == '4':
                self._run_search("Char N-gram", self.char_ngram_search)
            elif choice == '5':
                self._run_combined_search(search_engine=self._search_engine)
            elif choice == '6':
                self._compare_original_vs_experimental(search_engine)
            elif choice == '7':
                self._analyze_pair()
            elif choice == 'R':
                self._run_retrieve_rerank(search_engine)
            elif choice == 'T':
                self._test_keyword_extraction()
            elif choice == 'B':
                break

    def _run_search(self, name: str, search_func):
        """Run a specific search algorithm"""
        query = input(f"\nEnter query for {name} search: ").strip()
        if not query:
            return

        results = search_func(query)

        print(f"\n{name} Search Results:")
        if tabulate:
            rows = [[f"{score:.4f}", q[:60] + ('...' if len(q) > 60 else '')] for score, _, q in results[:10]]
            print(tabulate(rows, headers=['Score', 'Question'], tablefmt='simple'))
        else:
            for score, idx, q in results[:10]:
                print(f"  {score:.4f} - {q[:70]}")

    def _run_combined_search(self, search_engine=None):
        """Run combined search with ALL algorithms including cosine and BM25"""
        query = input("\nEnter query: ").strip()
        if not query:
            return

        results = self.combined_experimental_search(query, search_engine=search_engine)

        # Get keywords for highlighting
        keywords = self.extract_keywords(query, use_stemming=True)
        highlight_terms = set()
        for orig, stems in keywords:
            highlight_terms.add(orig.lower())
            for stem in stems:
                highlight_terms.add(stem.lower())
            # Also add variants
            for variant in self.generate_keyword_variants(orig):
                highlight_terms.add(variant.lower())

        print(f"\n{'='*110}")
        print(f"COMBINED RESULTS (7 algorithms) - Final = avg of ALL scores (zeros count against)")
        print(f"Keywords: {list(highlight_terms)[:10]}")
        print(f"{'='*110}")

        for i, r in enumerate(results[:10]):
            print(f"\n{'-'*110}")
            print(f"{i+1}. [Final: {r['final_score']:.2f}] [{r['num_algos']}/7 algos]")
            print(f"   RTR={r.get('rtr', 0):.2f}  Cos={r.get('cos', 0):.2f}  BM25={r.get('bm25', 0):.2f}  Norm={r.get('norm', 0):.2f}  Stem={r.get('stem', 0):.2f}  Fuzz={r.get('fuzz', 0):.2f}  Ngrm={r.get('ngrm', 0):.2f}")
            print(f"\n   Q: {r['question']}")

            # Highlight keywords in answer with >>> <<<
            answer = r.get('answer', '') or self.answers[r['idx']]
            answer_highlighted = answer
            for term in highlight_terms:
                if len(term) > 2:  # Skip very short terms
                    import re
                    # Case-insensitive replacement with markers
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    answer_highlighted = pattern.sub(f'>>>{term.upper()}<<<', answer_highlighted)

            print(f"\n   A: {answer_highlighted}")

        print(f"\n{'='*110}")
        print("   Keywords highlighted with >>>KEYWORD<<<")

    def _compare_original_vs_experimental(self, search_engine):
        """Compare original search vs experimental"""
        query = input("\nEnter query: ").strip()
        if not query:
            return

        print(f"\n{'='*60}")
        print(f"COMPARISON: '{query}'")
        print(f"{'='*60}")

        # Original semantic
        print("\n[ORIGINAL] Semantic Search:")
        orig = search_engine.similarity_search(query, top_k=5)
        if orig.get('top_k_results'):
            for i, (item, score) in enumerate(zip(orig['top_k_results'], orig.get('scores', []))):
                print(f"  {i+1}. [{score:.4f}] {item.get('question', '')[:60]}")

        # Normalized
        print("\n[EXPERIMENTAL] Normalized Search:")
        norm = self.normalized_search(query, 5)
        for i, (score, _, q) in enumerate(norm):
            print(f"  {i+1}. [{score:.4f}] {q[:60]}")

        # Combined
        print("\n[EXPERIMENTAL] Combined Search:")
        comb = self.combined_experimental_search(query, 5)
        for i, r in enumerate(comb):
            print(f"  {i+1}. [{r['final_score']:.4f}] {r['question'][:60]}")

    def _analyze_pair(self):
        """Analyze why two questions match or don't match"""
        q1 = input("\nEnter first question/phrase: ").strip()
        q2 = input("Enter second question/phrase: ").strip()
        if not q1 or not q2:
            return

        print(f"\n{'='*60}")
        print(f"ANALYZING PAIR")
        print(f"{'='*60}")
        print(f"\nQ1: '{q1}'")
        print(f"Q2: '{q2}'")

        # Normalized versions
        norm1, norm2 = self.normalize_text(q1), self.normalize_text(q2)
        print(f"\nNormalized:")
        print(f"  Q1: '{norm1}'")
        print(f"  Q2: '{norm2}'")
        print(f"  Match: {norm1 == norm2}")

        # Token overlap
        tokens1, tokens2 = set(norm1.split()), set(norm2.split())
        overlap = tokens1 & tokens2
        print(f"\nToken Overlap: {overlap}")
        print(f"  Jaccard: {self.jaccard_similarity(tokens1, tokens2):.4f}")

        # Stemmed versions
        if self.has_nltk:
            stem1, stem2 = self.stem_text(q1), self.stem_text(q2)
            print(f"\nStemmed:")
            print(f"  Q1: '{stem1}'")
            print(f"  Q2: '{stem2}'")

            # Stemmed token overlap
            stems1 = set(stem1.split())
            stems2 = set(stem2.split())
            stem_overlap = stems1 & stems2
            print(f"  Stem Overlap: {stem_overlap}")
            print(f"  Stem Jaccard: {self.jaccard_similarity(stems1, stems2):.4f}")

        # Character n-grams
        ng1, ng2 = self.char_ngrams(q1, 3), self.char_ngrams(q2, 3)
        print(f"\nChar 3-grams Jaccard: {self.jaccard_similarity(ng1, ng2):.4f}")

        # Fuzzy
        if self.has_fuzzy:
            print(f"\nFuzzy Scores:")
            print(f"  Ratio: {self.fuzz.ratio(q1, q2)}")
            print(f"  Partial Ratio: {self.fuzz.partial_ratio(q1, q2)}")
            print(f"  Token Set Ratio: {self.fuzz.token_set_ratio(q1, q2)}")
            print(f"  Token Sort Ratio: {self.fuzz.token_sort_ratio(q1, q2)}")

    def _run_retrieve_rerank(self, search_engine):
        """Run the retrieve-then-rerank algorithm"""
        query = input("\nEnter query for Retrieve-Then-Rerank: ").strip()
        if not query:
            return

        # Try to get semantic encoder from search_engine if available
        semantic_encoder = None
        if hasattr(search_engine, 'encoder'):
            semantic_encoder = search_engine.encoder

        results = self.test_retrieve_rerank(query, semantic_encoder=semantic_encoder)

        # Show comparison with original
        print(f"\n{'='*60}")
        print("COMPARISON WITH ORIGINAL SEMANTIC SEARCH:")
        print(f"{'='*60}")

        orig = search_engine.similarity_search(query, top_k=5)
        if orig.get('candidates'):
            print("\n[ORIGINAL] Semantic Search Top 5:")
            for i, item in enumerate(orig['candidates'][:5]):
                score = item.get('score', 0)
                question = item.get('question', '')[:60]
                print(f"  {i+1}. [{score:.4f}] {question}")

        print("\n[NEW] Retrieve-Then-Rerank Top 5:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. [{r['final_score']:.4f}] {r['question'][:60]}")
            print(f"       ^ kw={r['keyword_score']:.2f}, ans={r['answer_relevance']:.2f}, matched: {r['matched_keywords']}")

    def _test_keyword_extraction(self):
        """Test keyword extraction and stemming"""
        query = input("\nEnter query to analyze: ").strip()
        if not query:
            return

        print(f"\n{'='*60}")
        print(f"KEYWORD EXTRACTION: '{query}'")
        print(f"{'='*60}")

        # Show stopwords being used
        stopwords = self.get_stopwords()
        print(f"\nUsing NLTK stopwords: {self.has_nltk}")
        print(f"Stopwords count: {len(stopwords)}")

        # Extract keywords
        keywords = self.extract_keywords(query, use_stemming=True)
        print(f"\nExtracted keywords (original -> stems):")
        for orig, stems in keywords:
            print(f"  '{orig}' -> {stems}")

        # Show variants for each keyword
        print(f"\nKeyword variants:")
        for orig, stems in keywords:
            variants = self.generate_keyword_variants(orig)
            print(f"  '{orig}': {variants}")

        # Test against sample answers
        print(f"\n{'='*60}")
        print("Testing against first 5 answers that match:")
        print(f"{'='*60}")

        count = 0
        for idx, answer in enumerate(self.answers):
            for orig, stems in keywords:
                match_score = max((self.fuzzy_substring_match(orig, stem, answer, text_idx=idx) for stem in stems), default=0)
                if match_score > 0:
                    print(f"\n  [{idx}] Match for '{orig}' (stems: {stems})")
                    print(f"       Q: {self.questions[idx][:60]}...")
                    # Find and highlight the match in answer
                    answer_lower = answer.lower()
                    for variant in self.generate_keyword_variants(orig):
                        if variant.lower() in answer_lower:
                            start = answer_lower.find(variant.lower())
                            snippet = answer[max(0,start-20):start+len(variant)+30]
                            print(f"       A: ...{snippet}...")
                            break
                    count += 1
                    if count >= 5:
                        break
            if count >= 5:
                break

        print(f"\n  Total matches in dataset: ", end="")
        total = 0
        for idx, answer in enumerate(self.answers):
            for orig, stems in keywords:
                match_score = max((self.fuzzy_substring_match(orig, stem, answer, text_idx=idx) for stem in stems), default=0)
                if match_score > 0:
                    total += 1
                    break
        print(f"{total}")


if __name__ == "__main__":
    print("This module is meant to be used via menu.py")
    print("Run: python menu.py <json_file>")

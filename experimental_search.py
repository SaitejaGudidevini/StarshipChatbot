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

        # Pre-compute stems for all answers (fast, saves MILLIONS of stem calls)
        self._precompute_stems()

        # Gemini embeddings (lazy loaded)
        self.gemini_embeddings = None
        self.gemini_model = None

        # BGE-M3 embeddings for combined Q+A (lazy loaded)
        self.bge_qa_embeddings = None
        self.bge_encoder = None

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

        Caches results to disk based on hash of answers - only recomputes if data changes.
        """
        import hashlib
        import pickle
        import os

        self.answer_lemmas = []
        self.combined_lemmas = []

        if not self.has_spacy:
            print("  ⏭️  Skipping lemma precomputation (no spaCy)")
            return

        # Create a hash of the answers to detect changes
        answers_str = ''.join(self.answers[:100])  # Hash first 100 for speed
        answers_hash = hashlib.md5(f"{len(self.answers)}_{answers_str}".encode()).hexdigest()[:12]

        # Cache file path
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'lemmas_{answers_hash}.pkl')

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if cached.get('count') == len(self.answers):
                        self.answer_lemmas = cached['answer_lemmas']
                        self.combined_lemmas = cached['combined_lemmas']
                        print(f"  ✅ Loaded cached lemmas for {len(self.answers)} answers")
                        return
            except Exception as e:
                print(f"  ⚠️  Cache load failed: {e}")

        # Compute lemmas
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

        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'count': len(self.answers),
                    'answer_lemmas': self.answer_lemmas,
                    'combined_lemmas': self.combined_lemmas
                }, f)
            print(f"  💾 Cached lemmas to {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"  ⚠️  Failed to cache lemmas: {e}")

    def _precompute_stems(self):
        """
        Pre-compute stems for all answers to avoid calling porter.stem in the hot loop.
        This was causing 5+ MILLION stem calls per query (27k comparisons × 200 words).

        Caches results to disk based on hash of answers.
        """
        import hashlib
        import pickle
        import os

        self.answer_stems = []

        if not self.has_nltk:
            print("  ⏭️  Skipping stem precomputation (no NLTK)")
            return

        # Create a hash of the answers to detect changes
        answers_str = ''.join(self.answers[:100])
        answers_hash = hashlib.md5(f"{len(self.answers)}_{answers_str}".encode()).hexdigest()[:12]

        # Cache file path
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'stems_{answers_hash}.pkl')

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if cached.get('count') == len(self.answers):
                        self.answer_stems = cached['answer_stems']
                        print(f"  ✅ Loaded cached stems for {len(self.answers)} answers")
                        return
            except Exception as e:
                print(f"  ⚠️  Stem cache load failed: {e}")

        # Compute stems
        print(f"  🔄 Pre-computing stems for {len(self.answers)} answers...", end="", flush=True)
        for i, answer in enumerate(self.answers):
            text_normalized = self.normalize_text(answer)
            text_words = text_normalized.split()
            stems = set(self.porter.stem(w) for w in text_words)
            self.answer_stems.append(stems)

            if (i + 1) % 500 == 0:
                print(f" {i+1}", end="", flush=True)

        print(f" Done!")

        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'count': len(self.answers),
                    'answer_stems': self.answer_stems
                }, f)
            print(f"  💾 Cached stems to {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"  ⚠️  Failed to cache stems: {e}")

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

    def hybrid_rrf_search(self, query: str, top_k: int = 10, lexical_method: str = 'stemmed', semantic_method: str = 'gemini') -> List[Tuple[float, int, str]]:
        """
        Reciprocal Rank Fusion (RRF) - combines lexical and semantic search.
        Like concrete + rebar: lexical catches exact matches, semantic catches meaning.

        Args:
            lexical_method: 'stemmed', 'normalized', or 'fuzzy'
            semantic_method: 'gemini' or 'bge'

        Returns: List of (rrf_score, idx, question)
        """
        # Get lexical results - returns [(score, idx, question), ...]
        if lexical_method == 'stemmed':
            lexical_results = self.stemmed_search(query, top_k=20)
        elif lexical_method == 'normalized':
            lexical_results = self.normalized_search(query, top_k=20)
        elif lexical_method == 'fuzzy':
            lexical_results = self.fuzzy_search(query, top_k=20)
        else:
            lexical_results = self.stemmed_search(query, top_k=20)

        # Get semantic results - returns [{'idx': ..., 'score': ...}, ...]
        if semantic_method == 'gemini':
            semantic_results = self.gemini_semantic_search(query, top_k=20, quiet=True)
        else:
            semantic_results = self.bge_qa_search(query, top_k=20, quiet=True)

        if not semantic_results:
            # Fallback if semantic not available
            return lexical_results[:top_k]

        # Build rank maps: idx -> rank (0-based)
        # Lexical returns tuples: (score, idx, question)
        lex_rank = {r[1]: i for i, r in enumerate(lexical_results)}
        # Semantic returns dicts: {'idx': ..., 'score': ...}
        sem_rank = {r['idx']: i for i, r in enumerate(semantic_results)}

        # RRF constant (standard value from literature)
        k = 60

        # Compute RRF scores for all candidates
        all_idx = set(lex_rank.keys()) | set(sem_rank.keys())

        fused = []
        for idx in all_idx:
            rrf_score = 0
            # Contribution from lexical (if present in top 20)
            if idx in lex_rank:
                rrf_score += 1 / (k + lex_rank[idx])
            # Contribution from semantic (if present in top 20)
            if idx in sem_rank:
                rrf_score += 1 / (k + sem_rank[idx])

            fused.append((rrf_score, idx, self.questions[idx]))

        # Sort by RRF score descending
        fused.sort(reverse=True)
        return fused[:top_k]

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

        # Cosine similarity (semantic) - use BGE on Q+A combined (not just questions!)
        cosine = {}
        bm25 = {}
        if search_engine:
            print(f"    [6/7] Cosine similarity (BGE on Q+A)...")
            try:
                # Use our BGE Q+A search instead of question-only search
                bge_qa_results = self.bge_qa_search(query, search_engine=search_engine, top_k=100)
                if bge_qa_results:
                    for r in bge_qa_results:
                        cosine[r['idx']] = r['score']
                    print(f"      BGE Q+A: best={max(cosine.values()):.3f}, found={len(cosine)} results")
                else:
                    # Fallback to old question-only search
                    print(f"      (BGE Q+A failed, falling back to question-only)")
                    sem_result = search_engine.similarity_search(query, top_k=100)
                    if sem_result.get('candidates'):
                        scores_list = sem_result.get('scores', [])
                        for i, item in enumerate(sem_result['candidates']):
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
                    # Filter stopwords from query before BM25
                    stopwords = self.get_stopwords()
                    query_tokens = [w for w in query.lower().split() if w not in stopwords and len(w) > 2]

                    if query_tokens:
                        print(f"      BM25 tokens (stopwords removed): {query_tokens}")
                        bm25_scores = search_engine.bm25.get_scores(query_tokens)

                        # Smarter normalization: use percentile-based scaling
                        # Instead of max, use the 95th percentile to avoid outlier compression
                        scores_array = np.array(bm25_scores)
                        nonzero_scores = scores_array[scores_array > 0]

                        if len(nonzero_scores) > 0:
                            # Use 95th percentile as the "max" to preserve dynamic range
                            p95 = np.percentile(nonzero_scores, 95)
                            p50 = np.percentile(nonzero_scores, 50)  # median

                            print(f"      BM25 stats: min={nonzero_scores.min():.2f}, median={p50:.2f}, p95={p95:.2f}, max={nonzero_scores.max():.2f}")

                            for idx, score in enumerate(bm25_scores):
                                if score > 0:
                                    # Normalize: scores above p95 get capped at 1.0
                                    # This preserves more dynamic range
                                    norm_score = min(1.0, score / p95) if p95 > 0 else 0
                                    bm25[idx] = norm_score
                    else:
                        print(f"      BM25 skipped: no keywords after stopword removal")
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

            # Count how many algorithms found this result
            all_scores = list(scores.values())
            num_algos = len([v for v in all_scores if v > 0])

            # Require at least 2 algorithms to agree
            if num_algos < 2:
                continue

            # Compute family scores (group related algorithms)
            keyword_family = max(scores['rtr'], scores['bm25'])
            text_family = max(scores['norm'], scores['stem'], scores['fuzz'])
            semantic_family = scores['cos']

            # =========================================================
            # MULTIPLE SCORING METHODS - compare them all!
            # =========================================================
            scoring_methods = {}

            # Method 1: Simple Average (current)
            scoring_methods['avg'] = sum(all_scores) / len(all_scores)

            # Method 2: Average without n-gram (it's always tiny/useless)
            scores_no_ngram = [scores['rtr'], scores['cos'], scores['bm25'], scores['norm'], scores['stem'], scores['fuzz']]
            scoring_methods['avg6'] = sum(scores_no_ngram) / 6

            # Method 3: Family Max then Average (3 families)
            scoring_methods['fam_avg'] = (keyword_family + text_family + semantic_family) / 3

            # Method 4: Family Weighted (keyword heavy)
            scoring_methods['fam_wt'] = 0.5 * keyword_family + 0.3 * text_family + 0.2 * semantic_family

            # Method 5: Keyword Penalty (penalize if no keyword match)
            if keyword_family == 0:
                scoring_methods['kw_pen'] = 0.3 * text_family + 0.2 * semantic_family
            else:
                scoring_methods['kw_pen'] = 0.5 * keyword_family + 0.3 * text_family + 0.2 * semantic_family

            # Method 6: Geometric Mean of families (punishes zeros hard)
            import math
            if keyword_family > 0 and text_family > 0 and semantic_family > 0:
                scoring_methods['geo'] = (keyword_family * text_family * semantic_family) ** (1/3)
            else:
                scoring_methods['geo'] = 0

            # Method 7: RTR-dominant (RTR is the hero for keyword queries)
            scoring_methods['rtr_dom'] = 0.6 * scores['rtr'] + 0.2 * text_family + 0.2 * semantic_family

            # Method 8: Harmonic mean of families (sensitive to low values)
            if keyword_family > 0 and text_family > 0 and semantic_family > 0:
                scoring_methods['harm'] = 3 / (1/keyword_family + 1/text_family + 1/semantic_family)
            else:
                scoring_methods['harm'] = 0

            # Default final score uses family weighted
            final_score = scoring_methods['fam_wt']

            combined.append({
                'idx': idx,
                'question': self.questions[idx],
                'answer': self.answers[idx][:200] + '...' if len(self.answers[idx]) > 200 else self.answers[idx],
                'final_score': final_score,
                'num_algos': num_algos,
                'keyword_family': keyword_family,
                'text_family': text_family,
                'semantic_family': semantic_family,
                'scoring_methods': scoring_methods,
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

    def stem_match(self, keyword_stem: str, text: str, text_idx: int = None) -> bool:
        """
        Check if any word in text shares the same stem as keyword.
        This is the key insight: "founder", "co-founder", "founded" all stem to "found".

        Args:
            text_idx: If provided, use pre-computed stems (CRITICAL for performance!)
        """
        if not self.has_nltk:
            return keyword_stem.lower() in text.lower()

        # Use pre-computed stems if available (saves MILLIONS of stem calls)
        if text_idx is not None and hasattr(self, 'answer_stems') and self.answer_stems:
            return keyword_stem in self.answer_stems[text_idx]

        # Fallback: compute on the fly (slow)
        text_normalized = self.normalize_text(text)
        text_words = text_normalized.split()
        text_stems = set(self.porter.stem(w) for w in text_words)

        return keyword_stem in text_stems

    def fuzzy_substring_match(self, keyword: str, keyword_stem: str, text: str, threshold: int = 80, text_idx: int = None, use_combined: bool = False, keyword_lemmas: set = None, skip_fuzzy: bool = False) -> float:
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
            keyword_lemmas: Pre-computed lemmas for keyword (avoids spaCy call in hot loop)
            skip_fuzzy: If True, skip Tier 3 fuzzy matching (for fast retrieval phase)
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
            # Use pre-computed keyword lemmas if provided (CRITICAL for performance!)
            if keyword_lemmas is None:
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

        # TIER 3: Fuzzy substring match (SKIP in retrieval phase for performance)
        if self.has_fuzzy and not skip_fuzzy:
            best_fuzzy = 0
            text_for_fuzzy = text_lower[:300]  # Truncate for performance
            for variant in variants:
                score = self.fuzz.partial_ratio(variant.lower(), text_for_fuzzy)
                best_fuzzy = max(best_fuzzy, score)
            if best_fuzzy >= threshold:
                return 0.6 * (best_fuzzy / 100)  # Scale fuzzy score

        # TIER 4: Stem match - LOWEST PRIORITY (risky, can match wrong things)
        if self.stem_match(keyword_stem, text, text_idx=text_idx):
            return 0.3  # Low score for stem-only match

        return 0.0  # No match

    def answer_keyword_retrieval(self, query: str, min_keywords_match: int = 1, quiet: bool = False) -> List[Dict]:
        """
        STAGE 1: Retrieve all Q&As where the ANSWER contains query keywords.
        This is the "cast wide net" phase - find everything potentially relevant.

        Uses stemming so "cofounder" matches "Co-Founder", "founded", etc.
        Uses IDF weighting so rare keywords (like "founder") matter more than common ones (like "syra").

        Returns list of dicts with idx, question, answer, matched_keywords, match_count, idf_score
        """
        import time
        _t0 = time.time()

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

        _t1 = time.time()  # After keyword extraction + IDF

        # Sort keywords by IDF (rarest first) for display
        sorted_kws = sorted(keywords, key=lambda x: keyword_idfs[x[0]], reverse=True)
        if not quiet:
            print(f"\n  [Retrieval] Keywords (sorted by rarity):")
            for orig, stems in sorted_kws:
                idf = keyword_idfs[orig]
                # Show doc freq for each stem
                stem_info = ', '.join(f"{s}({self.stem_doc_freq.get(s, 0)})" for s in stems[:3])
                print(f"    '{orig}' -> stems: [{stem_info}] - IDF: {idf:.3f}")

        matches = []
        total_answers = len(self.answers)

        # PRE-COMPUTE keyword lemmas ONCE (not in the inner loop!)
        # This is CRITICAL for performance - was causing 13,000+ spaCy calls per query
        keyword_lemmas_cache = {}
        if self.has_spacy:
            for orig_kw, stems in keywords:
                kw_doc = self.nlp(orig_kw.lower())
                keyword_lemmas_cache[orig_kw] = set(token.lemma_ for token in kw_doc if len(token.lemma_) > 2)

        _t2 = time.time()  # After lemma pre-computation
        if not quiet:
            print(f"    [RTR-TIMING] kw+IDF: {(_t1-_t0)*1000:.0f}ms, lemma: {(_t2-_t1)*1000:.0f}ms", end="", flush=True)

        for idx, answer in enumerate(self.answers):
            # Progress indicator every 500 answers
            if not quiet and idx % 500 == 0 and idx > 0:
                print(f"    ... processing {idx}/{total_answers} answers", flush=True)

            matched_keywords = []
            matched_idfs = []
            matched_scores = []  # Track HOW WELL each keyword matched
            combined = self.combined_qa[idx]

            for orig_kw, stems in keywords:
                # Check if ANY of the stems match - get best score
                best_score = 0.0
                kw_lemmas = keyword_lemmas_cache.get(orig_kw)  # Pre-computed!
                for stem in stems:
                    # Pass text_idx for pre-computed lemmas, skip_fuzzy for fast retrieval
                    score = self.fuzzy_substring_match(orig_kw, stem, answer, threshold=75, text_idx=idx, use_combined=False, keyword_lemmas=kw_lemmas, skip_fuzzy=True)
                    if score > best_score:
                        best_score = score
                    if score == 0:  # Try combined if answer didn't match
                        score = self.fuzzy_substring_match(orig_kw, stem, combined, threshold=75, text_idx=idx, use_combined=True, keyword_lemmas=kw_lemmas, skip_fuzzy=True)
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

        _t3 = time.time()  # After main loop

        # Sort by match quality FIRST, then IDF (exact matches beat stem matches)
        matches.sort(key=lambda x: (x['match_quality'], x['idf_score']), reverse=True)

        if not quiet:
            print(f"  [Retrieval] Found {len(matches)} candidates matching keywords")
            print(f", loop: {(_t3-_t2)*1000:.0f}ms, TOTAL: {(_t3-_t0)*1000:.0f}ms")
        return matches

    def retrieve_then_rerank(self, query: str, top_k: int = 10, semantic_encoder=None, quiet: bool = False) -> List[Dict]:
        """
        THE BIG FIX: Retrieve-Then-Rerank Algorithm

        Stage 1: RETRIEVE - Cast wide net using keyword matching in answers (with stemming!)
        Stage 2: RERANK - Use multiple signals to score candidates

        This fixes the "cofounder" problem because:
        1. Stemming: "cofounder" stems to "cofound", matches "Co-Founder"
        2. We search in ANSWERS, not just questions
        3. Multiple ranking signals avoid semantic embedding failures
        """
        if not quiet:
            print(f"\n{'='*60}")
            print(f"RETRIEVE-THEN-RERANK: '{query}'")
            print(f"{'='*60}")

        # STAGE 1: RETRIEVE
        candidates = self.answer_keyword_retrieval(query, min_keywords_match=1, quiet=quiet)

        if not candidates:
            if not quiet:
                print("  [Warning] No keyword matches found - falling back to combined search")
            return self.combined_experimental_search(query, top_k)

        # STAGE 2: RERANK with multiple signals
        if not quiet:
            print(f"\n  [Rerank] Scoring {len(candidates)} candidates...")

        # Get the stemmed keywords for checking where matches occur
        query_keyword_stems = set()
        keywords = self.extract_keywords(query, use_stemming=True)
        for orig, stems in keywords:
            query_keyword_stems.update(stems)  # Add all stems

        # Pre-compute query stems ONCE (not per candidate!)
        query_stems = set()
        if self.has_nltk:
            for w in self.normalize_text(query).split():
                query_stems.add(self.porter.stem(w))
        else:
            query_stems = set(self.normalize_text(query).split())

        # Pre-compute query ngrams ONCE (not per candidate!)
        query_ngrams = self.char_ngrams(query, 3)

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
            question_norm = self.normalized_questions[idx]

            # Use PRE-COMPUTED stems (saves 100k+ stem calls in rerank phase!)
            if self.has_nltk and hasattr(self, 'answer_stems') and self.answer_stems:
                answer_stems = self.answer_stems[idx]
                # Questions are short, compute on the fly (or could pre-compute too)
                question_stems = set(self.porter.stem(w) for w in question_norm.split())
            else:
                answer_norm = self.normalize_text(answer)
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

            # Signal 3: Question similarity using stemmed tokens (query_stems pre-computed above)
            question_sim = self.jaccard_similarity(query_stems, question_stems)

            # Signal 4: Answer relevance - Jaccard of all query stems with answer stems
            answer_relevance = self.jaccard_similarity(query_stems, answer_stems)

            # Signal 5: Keyword prominence - what fraction of query keywords appear in answer?
            # (Simplified from density count to avoid re-stemming 400k+ words)
            keywords_in_answer = sum(1 for stem in query_keyword_stems if stem in answer_stems)
            keyword_prominence = keywords_in_answer / len(query_keyword_stems) if query_keyword_stems else 0

            # Signal 6: N-gram similarity for fuzzy matching (query_ngrams pre-computed above)
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
    # BGE-M3 ON COMBINED Q+A (THE RIGHT WAY)
    # =========================================================================

    def _init_bge_qa_embeddings(self, search_engine=None):
        """Compute BGE-M3 embeddings for combined Q+A (not just questions)."""
        import hashlib
        import pickle
        import os

        if self.bge_qa_embeddings is not None:
            return True

        # Get encoder from search engine
        if search_engine and hasattr(search_engine, 'encoder'):
            self.bge_encoder = search_engine.encoder
        else:
            print("  ❌ No BGE encoder available from search engine")
            return False

        # Cache file
        answers_str = ''.join(self.answers[:100])
        answers_hash = hashlib.md5(f"{len(self.answers)}_{answers_str}".encode()).hexdigest()[:12]
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'bge_qa_emb_{answers_hash}.pkl')

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if cached.get('count') == len(self.answers):
                        self.bge_qa_embeddings = cached['embeddings']
                        print(f"  ✅ Loaded cached BGE Q+A embeddings for {len(self.answers)} pairs")
                        return True
            except Exception as e:
                print(f"  ⚠️  Cache load failed: {e}")

        # Compute embeddings for combined Q+A
        print(f"  🔄 Computing BGE-M3 embeddings for {len(self.qa_pairs)} combined Q+A...")
        try:
            combined_texts = [f"{q} {a[:500]}" for q, a in zip(self.questions, self.answers)]
            self.bge_qa_embeddings = self.bge_encoder.encode(
                combined_texts,
                show_progress_bar=True,
                normalize_embeddings=True
            )

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'count': len(self.answers),
                    'embeddings': self.bge_qa_embeddings
                }, f)
            print(f"  💾 Cached BGE Q+A embeddings to {os.path.basename(cache_file)}")
            return True

        except Exception as e:
            print(f"  ❌ BGE Q+A embedding failed: {e}")
            return False

    def bge_qa_search(self, query: str, search_engine=None, top_k: int = 10, quiet: bool = False) -> List[Dict]:
        """Search using BGE-M3 on combined Q+A (not just questions)."""
        if not self._init_bge_qa_embeddings(search_engine):
            return []

        if not quiet:
            print(f"  🔍 Searching BGE-M3 on Q+A combined...")

        # Get query embedding
        query_emb = self.bge_encoder.encode([query], normalize_embeddings=True)[0]

        # Compute cosine similarities
        similarities = []
        for idx, doc_emb in enumerate(self.bge_qa_embeddings):
            sim = float(np.dot(query_emb, doc_emb))  # Already normalized, so dot = cosine
            similarities.append((sim, idx))

        # Sort by similarity
        similarities.sort(reverse=True)

        # Return top_k results
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                'idx': idx,
                'score': sim,
                'question': self.questions[idx],
                'answer': self.answers[idx][:300] + '...' if len(self.answers[idx]) > 300 else self.answers[idx]
            })

        return results

    # =========================================================================
    # GEMINI EMBEDDINGS (EXPERIMENTAL)
    # =========================================================================

    def _init_gemini(self):
        """Initialize Gemini embeddings - lazy loaded on first use."""
        if self.gemini_model is not None:
            return True

        try:
            import google.generativeai as genai
            import os

            # Try to load from .env file
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                print("  ❌ No GOOGLE_API_KEY or GEMINI_API_KEY found in environment")
                print("     Set it with: export GOOGLE_API_KEY='your-key'")
                return False

            genai.configure(api_key=api_key)
            self.gemini_model = genai
            print("  ✅ Gemini API initialized")
            return True
        except ImportError:
            print("  ❌ google-generativeai not installed")
            print("     Install with: pip install google-generativeai")
            return False
        except Exception as e:
            print(f"  ❌ Gemini init failed: {e}")
            return False

    def _get_gemini_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using Gemini."""
        result = self.gemini_model.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return np.array(result['embedding'])

    def _get_gemini_embeddings_batch(self, texts: list, task_type: str = "retrieval_document") -> np.ndarray:
        """Get embeddings for multiple texts using Gemini."""
        # Gemini has a batch limit, process in chunks
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            result = self.gemini_model.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type=task_type
            )
            all_embeddings.extend(result['embedding'])

            if (i + batch_size) % 500 == 0:
                print(f"    ... embedded {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)

        return np.array(all_embeddings)

    def _compute_gemini_embeddings(self, force_recompute: bool = False):
        """Compute and cache Gemini embeddings for all Q&A pairs."""
        import hashlib
        import pickle
        import os

        if self.gemini_embeddings is not None and not force_recompute:
            return True

        if not self._init_gemini():
            return False

        # Cache file
        answers_str = ''.join(self.answers[:100])
        answers_hash = hashlib.md5(f"{len(self.answers)}_{answers_str}".encode()).hexdigest()[:12]
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'gemini_emb_{answers_hash}.pkl')

        # Try to load from cache
        if not force_recompute and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if cached.get('count') == len(self.answers):
                        self.gemini_embeddings = cached['embeddings']
                        print(f"  ✅ Loaded cached Gemini embeddings for {len(self.answers)} Q&As")
                        return True
            except Exception as e:
                print(f"  ⚠️  Cache load failed: {e}")

        # Compute embeddings for combined Q+A
        print(f"  🔄 Computing Gemini embeddings for {len(self.qa_pairs)} Q&As...")
        print(f"     (This may take a few minutes and use API quota)")

        try:
            # Embed combined question + answer for better retrieval
            combined_texts = [f"{q} {a[:500]}" for q, a in zip(self.questions, self.answers)]
            self.gemini_embeddings = self._get_gemini_embeddings_batch(combined_texts)

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'count': len(self.answers),
                    'embeddings': self.gemini_embeddings
                }, f)
            print(f"  💾 Cached Gemini embeddings to {os.path.basename(cache_file)}")
            return True

        except Exception as e:
            print(f"  ❌ Gemini embedding failed: {e}")
            return False

    def gemini_semantic_search(self, query: str, top_k: int = 10, quiet: bool = False) -> List[Dict]:
        """Search using Gemini embeddings."""
        if not self._compute_gemini_embeddings():
            return []

        if not quiet:
            print(f"  🔍 Searching with Gemini embeddings...")

        # Get query embedding
        query_emb = self._get_gemini_embedding(query)

        # Compute cosine similarities
        similarities = []
        for idx, doc_emb in enumerate(self.gemini_embeddings):
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            similarities.append((float(sim), idx))

        # Sort by similarity
        similarities.sort(reverse=True)

        # Return top_k results
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({
                'idx': idx,
                'score': sim,
                'question': self.questions[idx],
                'answer': self.answers[idx][:300] + '...' if len(self.answers[idx]) > 300 else self.answers[idx]
            })

        return results

    def compare_embeddings(self, query: str, search_engine=None):
        """Compare BGE-M3 (Q only) vs BGE-M3 (Q+A) vs Gemini embeddings."""
        print(f"\n{'='*100}")
        print(f"EMBEDDING COMPARISON: '{query}'")
        print(f"{'='*100}")

        # BGE-M3 on Questions only (current behavior)
        bge_q_results = []
        if search_engine:
            print(f"\n📊 BGE-M3 on QUESTIONS only (current):")
            try:
                sem_result = search_engine.similarity_search(query, top_k=10)
                if sem_result.get('candidates'):
                    scores = sem_result.get('scores', [])
                    for i, item in enumerate(sem_result['candidates'][:10]):
                        q = item.question if hasattr(item, 'question') else item.get('question', '')
                        score = scores[i] if i < len(scores) else 0
                        print(f"   {i+1}. [{score:.3f}] {q[:70]}")
                        bge_q_results.append({'score': score, 'question': q})
            except Exception as e:
                print(f"   Error: {e}")

        # BGE-M3 on Combined Q+A (the right way!)
        print(f"\n📊 BGE-M3 on Q+A COMBINED (new):")
        bge_qa_results = self.bge_qa_search(query, search_engine=search_engine, top_k=10)
        if bge_qa_results:
            for i, r in enumerate(bge_qa_results[:10]):
                print(f"   {i+1}. [{r['score']:.3f}] {r['question'][:70]}")
        else:
            print("   (BGE Q+A search failed)")

        # Gemini on Combined Q+A
        print(f"\n📊 Gemini on Q+A COMBINED:")
        gemini_results = self.gemini_semantic_search(query, top_k=10)
        if gemini_results:
            for i, r in enumerate(gemini_results[:10]):
                print(f"   {i+1}. [{r['score']:.3f}] {r['question'][:70]}")
        else:
            print("   (Gemini search failed or not configured)")

        # Side by side comparison - all three
        print(f"\n{'='*120}")
        print(f"SIDE BY SIDE (Top 5):")
        print(f"{'='*120}")
        print(f"{'BGE Questions':<38} | {'BGE Q+A':<38} | {'Gemini Q+A':<38}")
        print(f"{'-'*38}-+-{'-'*38}-+-{'-'*38}")
        for i in range(5):
            bge_q = bge_q_results[i] if i < len(bge_q_results) else {'score': 0, 'question': '-'}
            bge_qa = bge_qa_results[i] if i < len(bge_qa_results) else {'score': 0, 'question': '-'}
            gem = gemini_results[i] if i < len(gemini_results) else {'score': 0, 'question': '-'}

            bge_q_str = f"[{bge_q['score']:.2f}] {bge_q['question'][:26]}"
            bge_qa_str = f"[{bge_qa['score']:.2f}] {bge_qa['question'][:26]}"
            gem_str = f"[{gem['score']:.2f}] {gem['question'][:26]}"
            print(f"{bge_q_str:<38} | {bge_qa_str:<38} | {gem_str:<38}")

    # =========================================================================
    # IQ TEST - Compare Embedding Models
    # =========================================================================

    def run_iq_test(self, csv_path: str, search_engine=None, sample_size: int = 100):
        """
        IQ Test: Compare embedding models on rephrased questions.
        Tests if search finds the correct original Q&A for rephrased questions.

        Args:
            csv_path: Path to CSV with columns: original_question, rephrased_question
            search_engine: Main search engine (for BGE encoder)
            sample_size: Max number of test cases to run (0 = all)
        """
        import csv
        import os

        print(f"\n{'='*80}")
        print(f"🧪 EMBEDDING IQ TEST")
        print(f"{'='*80}")

        # 1. Load CSV
        if not os.path.exists(csv_path):
            print(f"  ❌ CSV not found: {csv_path}")
            return

        test_cases = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'original': row.get('original_question', ''),
                    'rephrased': row.get('rephrased_question', '')
                })

        print(f"  📄 Loaded {len(test_cases)} test cases from {os.path.basename(csv_path)}")

        # 2. Build a lookup: question text -> index in self.questions
        question_to_idx = {q: i for i, q in enumerate(self.questions)}

        # Also normalized version for looser matching
        norm_question_to_idx = {self.normalize_text(q): i for i, q in enumerate(self.questions)}

        # Match CSV original questions to KB indices
        valid_cases = []
        for tc in test_cases:
            original = tc['original']

            # Find the index of this question in our KB
            if original in question_to_idx:
                tc['expected_idx'] = question_to_idx[original]
                valid_cases.append(tc)
            elif self.normalize_text(original) in norm_question_to_idx:
                tc['expected_idx'] = norm_question_to_idx[self.normalize_text(original)]
                valid_cases.append(tc)

        unmatched = len(test_cases) - len(valid_cases)
        print(f"  ✅ Matched {len(valid_cases)}/{len(test_cases)} test cases to KB")
        if unmatched > 0:
            print(f"  ⚠️  {unmatched} test cases NOT FOUND in KB (different JSON file?)")

        if sample_size and len(valid_cases) > sample_size:
            print(f"  📊 Sampling {sample_size} test cases")
            valid_cases = valid_cases[:sample_size]

        if not valid_cases:
            print(f"  ❌ No valid test cases found!")
            return

        # 3. Initialize embeddings
        print(f"\n  Initializing embeddings...")
        bge_ready = self._init_bge_qa_embeddings(search_engine)
        gemini_ready = self._compute_gemini_embeddings()

        if not bge_ready and not gemini_ready:
            print(f"  ❌ No embedding models available!")
            return

        # 4. Run tests
        print(f"\n  Running {len(valid_cases)} test cases...")
        print(f"  {'='*70}")

        bge_results = {'top1': 0, 'top3': 0, 'top5': 0, 'failures': []}
        gemini_results = {'top1': 0, 'top3': 0, 'top5': 0, 'failures': []}
        both_failed = []
        bge_only_passed = []
        gemini_only_passed = []

        for i, tc in enumerate(valid_cases):
            rephrased = tc['rephrased']
            expected_idx = tc['expected_idx']  # Just compare indices!

            # Progress
            if (i + 1) % 25 == 0:
                print(f"    ... {i+1}/{len(valid_cases)}")

            # BGE Q+A Search
            bge_pass_1, bge_pass_3, bge_pass_5 = False, False, False
            if bge_ready:
                bge_hits = self.bge_qa_search(rephrased, search_engine=search_engine, top_k=5)
                for rank, hit in enumerate(bge_hits):
                    if hit['idx'] == expected_idx:  # Simple index comparison!
                        if rank == 0:
                            bge_pass_1 = True
                        if rank < 3:
                            bge_pass_3 = True
                        if rank < 5:
                            bge_pass_5 = True
                        break

                if bge_pass_1:
                    bge_results['top1'] += 1
                if bge_pass_3:
                    bge_results['top3'] += 1
                if bge_pass_5:
                    bge_results['top5'] += 1
                if not bge_pass_1:
                    bge_results['failures'].append(tc)

            # Gemini Q+A Search
            gemini_pass_1, gemini_pass_3, gemini_pass_5 = False, False, False
            if gemini_ready:
                gemini_hits = self.gemini_semantic_search(rephrased, top_k=5)
                for rank, hit in enumerate(gemini_hits):
                    if hit['idx'] == expected_idx:  # Simple index comparison!
                        if rank == 0:
                            gemini_pass_1 = True
                        if rank < 3:
                            gemini_pass_3 = True
                        if rank < 5:
                            gemini_pass_5 = True
                        break

                if gemini_pass_1:
                    gemini_results['top1'] += 1
                if gemini_pass_3:
                    gemini_results['top3'] += 1
                if gemini_pass_5:
                    gemini_results['top5'] += 1
                if not gemini_pass_1:
                    gemini_results['failures'].append(tc)

            # Track differences
            if not bge_pass_1 and not gemini_pass_1:
                both_failed.append(tc)
            elif bge_pass_1 and not gemini_pass_1:
                bge_only_passed.append(tc)
            elif not bge_pass_1 and gemini_pass_1:
                gemini_only_passed.append(tc)

        # 5. Report Results
        matched = len(valid_cases)
        total_csv = len(test_cases)

        print(f"\n{'='*80}")
        print(f"🧪 IQ TEST RESULTS")
        print(f"{'='*80}")
        print(f"  CSV test cases: {total_csv}")
        print(f"  Matched to KB:  {matched} ({100*matched/total_csv:.1f}%)")
        if matched < total_csv:
            print(f"  ⚠️  Unmatched cases count as FAILURES in overall score")

        if bge_ready and matched > 0:
            print(f"\n  📊 BGE-M3 on Q+A:")
            print(f"     On matched ({matched}): {bge_results['top1']}/{matched} = {100*bge_results['top1']/matched:.1f}%")
            print(f"     OVERALL ({total_csv}):  {bge_results['top1']}/{total_csv} = {100*bge_results['top1']/total_csv:.1f}%")
            print(f"     Top-3 (matched): {bge_results['top3']}/{matched} ({100*bge_results['top3']/matched:.1f}%)")

        if gemini_ready and matched > 0:
            print(f"\n  📊 Gemini on Q+A:")
            print(f"     On matched ({matched}): {gemini_results['top1']}/{matched} = {100*gemini_results['top1']/matched:.1f}%")
            print(f"     OVERALL ({total_csv}):  {gemini_results['top1']}/{total_csv} = {100*gemini_results['top1']/total_csv:.1f}%")
            print(f"     Top-3 (matched): {gemini_results['top3']}/{matched} ({100*gemini_results['top3']/matched:.1f}%)")

        # Winner (based on overall accuracy)
        if bge_ready and gemini_ready and matched > 0:
            print(f"\n  {'='*70}")
            bge_pct = 100 * bge_results['top1'] / total_csv  # Use total CSV, not just matched
            gem_pct = 100 * gemini_results['top1'] / total_csv
            diff = gem_pct - bge_pct

            if diff > 0:
                print(f"  🏆 WINNER: Gemini (+{diff:.1f}% on top-1)")
            elif diff < 0:
                print(f"  🏆 WINNER: BGE-M3 (+{-diff:.1f}% on top-1)")
            else:
                print(f"  🤝 TIE: Both models equal on top-1")

            print(f"\n  📈 Difference Analysis:")
            print(f"     Both failed:       {len(both_failed)}")
            print(f"     BGE passed only:   {len(bge_only_passed)}")
            print(f"     Gemini passed only:{len(gemini_only_passed)}")

            # Show BOTH FAILED cases with full detail
            if both_failed:
                print(f"\n  {'='*70}")
                print(f"  ❌ BOTH FAILED ({len(both_failed)} cases) - Is the test data bad or search failing?")
                print(f"  {'='*70}")
                for i, tc in enumerate(both_failed[:10]):
                    idx = tc['expected_idx']
                    print(f"\n  [{i+1}]")
                    print(f"      ASKED:    {tc['rephrased']}")
                    print(f"      ORIGINAL: {tc['original']}")
                    print(f"      ANSWER:   {self.answers[idx][:150]}...")

            # Show examples where they differed - with full detail
            if gemini_only_passed:
                print(f"\n  {'='*70}")
                print(f"  🟢 GEMINI WON, BGE FAILED ({len(gemini_only_passed)} cases)")
                print(f"  {'='*70}")
                for i, tc in enumerate(gemini_only_passed[:5]):
                    idx = tc['expected_idx']
                    print(f"\n  [{i+1}]")
                    print(f"      ASKED:    {tc['rephrased']}")
                    print(f"      ORIGINAL: {tc['original']}")
                    print(f"      ANSWER:   {self.answers[idx][:150]}...")

            if bge_only_passed:
                print(f"\n  {'='*70}")
                print(f"  🔵 BGE WON, GEMINI FAILED ({len(bge_only_passed)} cases)")
                print(f"  {'='*70}")
                for i, tc in enumerate(bge_only_passed[:5]):
                    idx = tc['expected_idx']
                    print(f"\n  [{i+1}]")
                    print(f"      ASKED:    {tc['rephrased']}")
                    print(f"      ORIGINAL: {tc['original']}")
                    print(f"      ANSWER:   {self.answers[idx][:150]}...")

        print(f"\n{'='*80}")

    def run_iq_test_keyword(self, csv_path: str, search_engine=None, sample_size: int = 100):
        """
        IQ Test for keyword/lexical methods: RTR, Normalized, Stemmed, Fuzzy
        """
        import csv
        import os

        print(f"\n{'='*80}")
        print(f"🧪 KEYWORD/LEXICAL IQ TEST")
        print(f"{'='*80}")

        # Load and match test cases (same as embedding test)
        if not os.path.exists(csv_path):
            print(f"  ❌ CSV not found: {csv_path}")
            return

        test_cases = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'original': row.get('original_question', ''),
                    'rephrased': row.get('rephrased_question', '')
                })

        print(f"  📄 Loaded {len(test_cases)} test cases")

        # Build lookup
        question_to_idx = {q: i for i, q in enumerate(self.questions)}
        norm_question_to_idx = {self.normalize_text(q): i for i, q in enumerate(self.questions)}

        valid_cases = []
        for tc in test_cases:
            original = tc['original']
            if original in question_to_idx:
                tc['expected_idx'] = question_to_idx[original]
                valid_cases.append(tc)
            elif self.normalize_text(original) in norm_question_to_idx:
                tc['expected_idx'] = norm_question_to_idx[self.normalize_text(original)]
                valid_cases.append(tc)

        print(f"  ✅ Matched {len(valid_cases)}/{len(test_cases)} test cases to KB")

        if sample_size and len(valid_cases) > sample_size:
            valid_cases = valid_cases[:sample_size]

        if not valid_cases:
            print(f"  ❌ No valid test cases!")
            return

        # Test methods
        methods = {
            'RTR': {'top1': 0, 'top3': 0, 'failures': []},
            'Normalized': {'top1': 0, 'top3': 0, 'failures': []},
            'Stemmed': {'top1': 0, 'top3': 0, 'failures': []},
            'Fuzzy': {'top1': 0, 'top3': 0, 'failures': []},
        }

        import time

        # Track timings
        timings = {
            'RTR': 0.0,
            'Normalized': 0.0,
            'Stemmed': 0.0,
            'Fuzzy': 0.0,
        }

        print(f"\n  Running {len(valid_cases)} test cases with timing...")

        for i, tc in enumerate(valid_cases):
            rephrased = tc['rephrased']
            expected_idx = tc['expected_idx']

            if (i + 1) % 10 == 0:
                print(f"    ... {i+1}/{len(valid_cases)} | RTR:{timings['RTR']:.1f}s Norm:{timings['Normalized']:.1f}s Stem:{timings['Stemmed']:.1f}s Fuzz:{timings['Fuzzy']:.1f}s")

            # RTR
            t0 = time.time()
            rtr_results = self.retrieve_then_rerank(rephrased, top_k=5, quiet=True)
            timings['RTR'] += time.time() - t0
            for rank, r in enumerate(rtr_results):
                if r['idx'] == expected_idx:
                    if rank == 0:
                        methods['RTR']['top1'] += 1
                    if rank < 3:
                        methods['RTR']['top3'] += 1
                    break
            else:
                methods['RTR']['failures'].append(tc)

            # Normalized
            t0 = time.time()
            norm_results = self.normalized_search(rephrased, top_k=5)
            timings['Normalized'] += time.time() - t0
            for rank, (score, idx, q) in enumerate(norm_results):
                if idx == expected_idx:
                    if rank == 0:
                        methods['Normalized']['top1'] += 1
                    if rank < 3:
                        methods['Normalized']['top3'] += 1
                    break
            else:
                methods['Normalized']['failures'].append(tc)

            # Stemmed
            if self.has_nltk:
                t0 = time.time()
                stem_results = self.stemmed_search(rephrased, top_k=5)
                timings['Stemmed'] += time.time() - t0
                for rank, (score, idx, q) in enumerate(stem_results):
                    if idx == expected_idx:
                        if rank == 0:
                            methods['Stemmed']['top1'] += 1
                        if rank < 3:
                            methods['Stemmed']['top3'] += 1
                        break
                else:
                    methods['Stemmed']['failures'].append(tc)

            # Fuzzy
            if self.has_fuzzy:
                t0 = time.time()
                fuzz_results = self.fuzzy_search(rephrased, top_k=5)
                timings['Fuzzy'] += time.time() - t0
                for rank, (score, idx, q) in enumerate(fuzz_results):
                    if idx == expected_idx:
                        if rank == 0:
                            methods['Fuzzy']['top1'] += 1
                        if rank < 3:
                            methods['Fuzzy']['top3'] += 1
                        break
                else:
                    methods['Fuzzy']['failures'].append(tc)

        # Print timing summary
        print(f"\n  ⏱️  TIMING SUMMARY:")
        for name, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            avg_ms = (t / len(valid_cases)) * 1000
            print(f"     {name:<15} {t:.1f}s total, {avg_ms:.0f}ms per query")

        # Report
        total = len(valid_cases)
        print(f"\n{'='*80}")
        print(f"🧪 KEYWORD/LEXICAL IQ TEST RESULTS")
        print(f"{'='*80}")
        print(f"  Test cases: {total}")

        print(f"\n  {'Method':<15} {'Top-1':>15} {'Top-3':>15}")
        print(f"  {'-'*45}")
        for name, data in methods.items():
            if name == 'Stemmed' and not self.has_nltk:
                continue
            if name == 'Fuzzy' and not self.has_fuzzy:
                continue
            t1 = f"{data['top1']}/{total} ({100*data['top1']/total:.1f}%)"
            t3 = f"{data['top3']}/{total} ({100*data['top3']/total:.1f}%)"
            print(f"  {name:<15} {t1:>15} {t3:>15}")

        # Find best
        best_method = max(methods.keys(), key=lambda m: methods[m]['top1'])
        print(f"\n  🏆 BEST: {best_method} ({methods[best_method]['top1']}/{total})")

        # Show RTR failures
        rtr_failures = methods['RTR']['failures']
        if rtr_failures:
            print(f"\n  {'='*70}")
            print(f"  ❌ RTR FAILURES ({len(rtr_failures)} cases)")
            print(f"  {'='*70}")
            for i, tc in enumerate(rtr_failures[:5]):
                idx = tc['expected_idx']
                print(f"\n  [{i+1}]")
                print(f"      ASKED:    {tc['rephrased'][:80]}...")
                print(f"      ORIGINAL: {tc['original'][:80]}...")

        print(f"\n{'='*80}")

    def run_iq_test_all(self, csv_path: str, search_engine=None, sample_size: int = 100):
        """
        IQ Test comparing ALL methods: embeddings + keyword/lexical
        """
        import csv
        import os

        print(f"\n{'='*80}")
        print(f"🧪 FULL IQ TEST - ALL METHODS")
        print(f"{'='*80}")

        # Load and match test cases
        if not os.path.exists(csv_path):
            print(f"  ❌ CSV not found: {csv_path}")
            return

        test_cases = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append({
                    'original': row.get('original_question', ''),
                    'rephrased': row.get('rephrased_question', '')
                })

        print(f"  📄 Loaded {len(test_cases)} test cases")

        question_to_idx = {q: i for i, q in enumerate(self.questions)}
        norm_question_to_idx = {self.normalize_text(q): i for i, q in enumerate(self.questions)}

        valid_cases = []
        for tc in test_cases:
            original = tc['original']
            if original in question_to_idx:
                tc['expected_idx'] = question_to_idx[original]
                valid_cases.append(tc)
            elif self.normalize_text(original) in norm_question_to_idx:
                tc['expected_idx'] = norm_question_to_idx[self.normalize_text(original)]
                valid_cases.append(tc)

        print(f"  ✅ Matched {len(valid_cases)}/{len(test_cases)} test cases to KB")

        if sample_size and len(valid_cases) > sample_size:
            valid_cases = valid_cases[:sample_size]

        if not valid_cases:
            return

        # Initialize embeddings
        print(f"\n  Initializing...")
        bge_ready = self._init_bge_qa_embeddings(search_engine)
        gemini_ready = self._compute_gemini_embeddings()

        # All methods
        methods = {
            'BGE Q+A': {'top1': 0, 'top3': 0},
            'Gemini Q+A': {'top1': 0, 'top3': 0},
            'Hybrid RRF': {'top1': 0, 'top3': 0},  # Stemmed + Gemini combined
            'RTR': {'top1': 0, 'top3': 0},
            'Normalized': {'top1': 0, 'top3': 0},
            'Stemmed': {'top1': 0, 'top3': 0},
            'Fuzzy': {'top1': 0, 'top3': 0},
        }

        print(f"\n  Running {len(valid_cases)} test cases across all methods...")

        for i, tc in enumerate(valid_cases):
            rephrased = tc['rephrased']
            expected_idx = tc['expected_idx']

            if (i + 1) % 25 == 0:
                print(f"    ... {i+1}/{len(valid_cases)}")

            # BGE Q+A
            if bge_ready:
                results = self.bge_qa_search(rephrased, search_engine=search_engine, top_k=5, quiet=True)
                for rank, r in enumerate(results):
                    if r['idx'] == expected_idx:
                        if rank == 0: methods['BGE Q+A']['top1'] += 1
                        if rank < 3: methods['BGE Q+A']['top3'] += 1
                        break

            # Gemini Q+A
            if gemini_ready:
                results = self.gemini_semantic_search(rephrased, top_k=5, quiet=True)
                for rank, r in enumerate(results):
                    if r['idx'] == expected_idx:
                        if rank == 0: methods['Gemini Q+A']['top1'] += 1
                        if rank < 3: methods['Gemini Q+A']['top3'] += 1
                        break

            # RTR
            results = self.retrieve_then_rerank(rephrased, top_k=5, quiet=True)
            for rank, r in enumerate(results):
                if r['idx'] == expected_idx:
                    if rank == 0: methods['RTR']['top1'] += 1
                    if rank < 3: methods['RTR']['top3'] += 1
                    break

            # Normalized
            results = self.normalized_search(rephrased, top_k=5)
            for rank, (score, idx, q) in enumerate(results):
                if idx == expected_idx:
                    if rank == 0: methods['Normalized']['top1'] += 1
                    if rank < 3: methods['Normalized']['top3'] += 1
                    break

            # Stemmed
            if self.has_nltk:
                results = self.stemmed_search(rephrased, top_k=5)
                for rank, (score, idx, q) in enumerate(results):
                    if idx == expected_idx:
                        if rank == 0: methods['Stemmed']['top1'] += 1
                        if rank < 3: methods['Stemmed']['top3'] += 1
                        break

            # Fuzzy
            if self.has_fuzzy:
                results = self.fuzzy_search(rephrased, top_k=5)
                for rank, (score, idx, q) in enumerate(results):
                    if idx == expected_idx:
                        if rank == 0: methods['Fuzzy']['top1'] += 1
                        if rank < 3: methods['Fuzzy']['top3'] += 1
                        break

            # Hybrid RRF (Stemmed + Gemini)
            if gemini_ready and self.has_nltk:
                results = self.hybrid_rrf_search(rephrased, top_k=5, lexical_method='stemmed', semantic_method='gemini')
                for rank, (score, idx, q) in enumerate(results):
                    if idx == expected_idx:
                        if rank == 0: methods['Hybrid RRF']['top1'] += 1
                        if rank < 3: methods['Hybrid RRF']['top3'] += 1
                        break

        # Report
        total = len(valid_cases)
        print(f"\n{'='*80}")
        print(f"🧪 FULL IQ TEST RESULTS - ALL METHODS")
        print(f"{'='*80}")
        print(f"  Test cases: {total}")

        print(f"\n  {'Method':<15} {'Top-1':>15} {'Top-3':>15}")
        print(f"  {'-'*45}")

        sorted_methods = sorted(methods.items(), key=lambda x: x[1]['top1'], reverse=True)
        for name, data in sorted_methods:
            if name == 'Stemmed' and not self.has_nltk:
                continue
            if name == 'Fuzzy' and not self.has_fuzzy:
                continue
            if name == 'BGE Q+A' and not bge_ready:
                continue
            if name == 'Gemini Q+A' and not gemini_ready:
                continue
            if name == 'Hybrid RRF' and not (gemini_ready and self.has_nltk):
                continue
            t1 = f"{data['top1']}/{total} ({100*data['top1']/total:.1f}%)"
            t3 = f"{data['top3']}/{total} ({100*data['top3']/total:.1f}%)"
            print(f"  {name:<15} {t1:>15} {t3:>15}")

        # Winner
        best = sorted_methods[0]
        print(f"\n  🏆 WINNER: {best[0]} with {best[1]['top1']}/{total} ({100*best[1]['top1']/total:.1f}%)")

        print(f"\n{'='*80}")

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
            print("")
            print("[G] 🆕 Gemini embeddings search")
            print("[C] 🆕 Compare BGE-M3 vs Gemini")
            print("[I] 🧪 IQ Test (compare embedding accuracy)")
            print("[Q] 📝 Generate IQ Test CSV from current KB")
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
            elif choice == 'G':
                self._run_gemini_search()
            elif choice == 'C':
                self._run_embedding_comparison(search_engine)
            elif choice == 'I':
                self._run_iq_test_menu(search_engine)
            elif choice == 'Q':
                self._generate_iq_csv_menu()
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

        print(f"\n{'='*120}")
        print(f"COMBINED RESULTS - Comparing 8 Scoring Methods")
        print(f"Keywords: {list(highlight_terms)[:10]}")
        print(f"{'='*120}")

        # =========================================================
        # SCORING METHODS LEGEND
        # =========================================================
        print(f"\n📊 SCORING METHODS EXPLAINED:")
        print(f"   avg     = Simple average of all 7 algorithm scores")
        print(f"   avg6    = Average of 6 (drops n-gram which is always tiny)")
        print(f"   fam_avg = Family average: (keyword + text + semantic) / 3")
        print(f"   fam_wt  = Family weighted: 50% keyword + 30% text + 20% semantic")
        print(f"   kw_pen  = Keyword penalty: if no keyword match, cap score low")
        print(f"   geo     = Geometric mean of families (punishes zeros)")
        print(f"   rtr_dom = RTR dominant: 60% RTR + 20% text + 20% semantic")
        print(f"   harm    = Harmonic mean of families (sensitive to low values)")
        print(f"\n   Families: keyword=max(RTR,BM25)  text=max(Norm,Stem,Fuzz)  semantic=Cos")

        # =========================================================
        # COMPARISON TABLE - All methods for top 10
        # =========================================================
        print(f"\n{'='*120}")
        print(f"RANKING COMPARISON (top 10 by each method):")
        print(f"{'='*120}")

        if tabulate and results:
            # Build comparison table
            headers = ['#', 'avg', 'avg6', 'fam_avg', 'fam_wt', 'kw_pen', 'geo', 'rtr_dom', 'harm', 'Question (truncated)']
            rows = []
            for i, r in enumerate(results[:10]):
                sm = r.get('scoring_methods', {})
                rows.append([
                    i+1,
                    f"{sm.get('avg', 0):.2f}",
                    f"{sm.get('avg6', 0):.2f}",
                    f"{sm.get('fam_avg', 0):.2f}",
                    f"{sm.get('fam_wt', 0):.2f}",
                    f"{sm.get('kw_pen', 0):.2f}",
                    f"{sm.get('geo', 0):.2f}",
                    f"{sm.get('rtr_dom', 0):.2f}",
                    f"{sm.get('harm', 0):.2f}",
                    r['question'][:40] + '...'
                ])
            print(tabulate(rows, headers=headers, tablefmt='simple'))
        else:
            # Fallback without tabulate
            for i, r in enumerate(results[:10]):
                sm = r.get('scoring_methods', {})
                print(f"{i+1}. avg={sm.get('avg',0):.2f} fam_wt={sm.get('fam_wt',0):.2f} kw_pen={sm.get('kw_pen',0):.2f} | {r['question'][:50]}")

        # =========================================================
        # RE-RANK BY EACH METHOD and show what would be #1
        # =========================================================
        print(f"\n{'='*120}")
        print(f"WHAT WOULD BE #1 UNDER EACH METHOD?")
        print(f"{'='*120}")

        methods = ['avg', 'avg6', 'fam_avg', 'fam_wt', 'kw_pen', 'geo', 'rtr_dom', 'harm']
        for method in methods:
            # Sort by this method
            sorted_by_method = sorted(results, key=lambda x: x.get('scoring_methods', {}).get(method, 0), reverse=True)
            if sorted_by_method:
                top = sorted_by_method[0]
                score = top.get('scoring_methods', {}).get(method, 0)
                print(f"   {method:8} → [{score:.2f}] {top['question'][:70]}")

        # =========================================================
        # DETAILED VIEW OF TOP 5
        # =========================================================
        print(f"\n{'='*120}")
        print(f"DETAILED VIEW (top 5 by fam_wt):")
        print(f"{'='*120}")

        for i, r in enumerate(results[:5]):
            print(f"\n{'-'*120}")
            sm = r.get('scoring_methods', {})
            print(f"{i+1}. [{r['num_algos']}/7 algos]")

            # Raw scores
            print(f"   RAW:      RTR={r.get('rtr', 0):.2f}  Cos={r.get('cos', 0):.2f}  BM25={r.get('bm25', 0):.2f}  Norm={r.get('norm', 0):.2f}  Stem={r.get('stem', 0):.2f}  Fuzz={r.get('fuzz', 0):.2f}  Ngrm={r.get('ngrm', 0):.2f}")

            # Family scores
            print(f"   FAMILIES: keyword={r.get('keyword_family', 0):.2f}  text={r.get('text_family', 0):.2f}  semantic={r.get('semantic_family', 0):.2f}")

            # All scoring methods
            print(f"   METHODS:  avg={sm.get('avg',0):.2f}  avg6={sm.get('avg6',0):.2f}  fam_avg={sm.get('fam_avg',0):.2f}  fam_wt={sm.get('fam_wt',0):.2f}  kw_pen={sm.get('kw_pen',0):.2f}  geo={sm.get('geo',0):.2f}  rtr_dom={sm.get('rtr_dom',0):.2f}  harm={sm.get('harm',0):.2f}")

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

        print(f"\n{'='*120}")
        print("   Keywords highlighted with >>>KEYWORD<<<")
        print(f"{'='*120}")

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

    def _run_gemini_search(self):
        """Run Gemini embedding search"""
        query = input("\nEnter query for Gemini search: ").strip()
        if not query:
            return

        results = self.gemini_semantic_search(query, top_k=10)

        if results:
            print(f"\n{'='*80}")
            print(f"GEMINI EMBEDDING RESULTS: '{query}'")
            print(f"{'='*80}")
            for i, r in enumerate(results):
                print(f"\n{i+1}. [{r['score']:.3f}]")
                print(f"   Q: {r['question']}")
                print(f"   A: {r['answer'][:200]}...")

    def _run_embedding_comparison(self, search_engine):
        """Compare BGE-M3 vs Gemini embeddings"""
        query = input("\nEnter query to compare embeddings: ").strip()
        if not query:
            return

        self.compare_embeddings(query, search_engine)

    def _generate_iq_csv_menu(self):
        """Generate IQ test CSV from menu"""
        print(f"\n  This will generate rephrased questions from your current KB")
        print(f"  KB has {len(self.questions)} questions")

        sample_input = input("  How many questions to include? (Enter for 100, 0 for all): ").strip()
        try:
            sample_size = int(sample_input) if sample_input else 100
            if sample_size == 0:
                sample_size = len(self.questions)
        except ValueError:
            sample_size = 100

        use_gemini = input("  Use Gemini for natural rephrasing? (Y/n): ").strip().lower() != 'n'

        self.generate_iq_test_csv(sample_size=sample_size, use_gemini=use_gemini)

    def _run_iq_test_menu(self, search_engine):
        """Run the IQ test from menu"""
        import os

        # Default CSV path - use generated one if it exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generated_csv = os.path.join(script_dir, "iq_test_generated.csv")
        default_csv = os.path.join(script_dir, "SYRAhealth_rephrased_questions2.csv")

        if os.path.exists(generated_csv):
            default_path = generated_csv
            print(f"\n  Found generated CSV: iq_test_generated.csv")
        else:
            default_path = default_csv
            print(f"\n  Default CSV: SYRAhealth_rephrased_questions2.csv")
            print(f"  💡 Tip: Use option [Q] to generate a matching CSV from your KB")

        csv_input = input(f"  Enter CSV path (or press Enter for default): ").strip()

        if csv_input:
            csv_path = csv_input
        else:
            csv_path = default_path

        # Which test to run?
        print(f"\n  Which search method(s) to test?")
        print(f"    [1] BGE Q+A vs Gemini Q+A (embeddings only)")
        print(f"    [2] RTR + Keyword methods (lexical)")
        print(f"    [3] ALL methods (full comparison)")
        test_choice = input(f"  Choice (Enter for 1): ").strip() or "1"

        # Sample size
        sample_input = input("  Sample size (Enter for 100, 0 for all): ").strip()
        try:
            sample_size = int(sample_input) if sample_input else 100
        except ValueError:
            sample_size = 100

        if test_choice == "2":
            self.run_iq_test_keyword(csv_path, search_engine=search_engine, sample_size=sample_size)
        elif test_choice == "3":
            self.run_iq_test_all(csv_path, search_engine=search_engine, sample_size=sample_size)
        else:
            self.run_iq_test(csv_path, search_engine=search_engine, sample_size=sample_size)

    def generate_iq_test_csv(self, output_path: str = None, sample_size: int = 100, use_gemini: bool = True):
        """
        Generate an IQ test CSV from the current KB with rephrased questions.
        Uses Gemini to create natural variations, or falls back to programmatic rephrasing.
        """
        import csv
        import os
        import random

        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, "iq_test_generated.csv")

        print(f"\n{'='*80}")
        print(f"📝 GENERATING IQ TEST CSV")
        print(f"{'='*80}")
        print(f"  KB has {len(self.questions)} questions")

        # Sample questions if needed
        indices = list(range(len(self.questions)))
        if sample_size and sample_size < len(indices):
            random.shuffle(indices)
            indices = indices[:sample_size]
            print(f"  Sampling {sample_size} questions")

        test_cases = []

        # Try Gemini for natural rephrasing
        if use_gemini and self._init_gemini():
            print(f"  Using Gemini to generate natural rephrases...")
            print(f"  (This will use API quota)")

            # Batch questions for efficiency
            batch_size = 10
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_questions = [self.questions[i] for i in batch_indices]

                # Create prompt for Gemini
                prompt = """Rephrase each question below in a natural, conversational way.
Keep the same meaning but use different words/structure.
Return ONLY the rephrased questions, one per line, in the same order.

Questions:
"""
                for i, q in enumerate(batch_questions):
                    prompt += f"{i+1}. {q}\n"

                try:
                    # Try current models in order of preference
                    model_names = ['gemini-2.5-flash', 'gemini-2.0-flash-lite', 'gemini-pro']
                    response = None
                    for model_name in model_names:
                        try:
                            response = self.gemini_model.GenerativeModel(model_name).generate_content(prompt)
                            break
                        except Exception:
                            continue
                    if response is None:
                        raise Exception("No working model found")
                    rephrased_lines = response.text.strip().split('\n')

                    for i, idx in enumerate(batch_indices):
                        original = self.questions[idx]
                        if i < len(rephrased_lines):
                            # Clean up the rephrased text (remove numbering if present)
                            rephrased = rephrased_lines[i].strip()
                            rephrased = rephrased.lstrip('0123456789.-) ').strip()
                            if rephrased and len(rephrased) > 10:
                                test_cases.append({
                                    'original_question': original,
                                    'rephrased_question': rephrased
                                })
                            else:
                                # Fallback to programmatic
                                test_cases.append({
                                    'original_question': original,
                                    'rephrased_question': self._programmatic_rephrase(original)
                                })
                        else:
                            test_cases.append({
                                'original_question': original,
                                'rephrased_question': self._programmatic_rephrase(original)
                            })

                except Exception as e:
                    print(f"    Gemini error: {e}, using programmatic fallback")
                    for idx in batch_indices:
                        original = self.questions[idx]
                        test_cases.append({
                            'original_question': original,
                            'rephrased_question': self._programmatic_rephrase(original)
                        })

                # Progress
                print(f"    ... {min(batch_start + batch_size, len(indices))}/{len(indices)}")

        else:
            print(f"  Using programmatic rephrasing (Gemini not available)")
            for idx in indices:
                original = self.questions[idx]
                test_cases.append({
                    'original_question': original,
                    'rephrased_question': self._programmatic_rephrase(original)
                })

        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['original_question', 'rephrased_question'])
            writer.writeheader()
            writer.writerows(test_cases)

        print(f"\n  ✅ Generated {len(test_cases)} test cases")
        print(f"  📄 Saved to: {output_path}")

        # Show a few examples
        print(f"\n  Examples:")
        for tc in test_cases[:3]:
            print(f"    Original:  {tc['original_question'][:60]}...")
            print(f"    Rephrased: {tc['rephrased_question'][:60]}...")
            print()

        return output_path

    def _programmatic_rephrase(self, question: str) -> str:
        """Generate a simple programmatic rephrase of a question."""
        import random

        q = question.strip()

        # Common transformations
        transforms = [
            # What -> Which / Can you tell me what
            (r'^What ', ['Which ', 'Can you tell me what ', 'I want to know what ']),
            (r'^What is ', ['What\'s ', 'Tell me what is ', 'Explain what is ']),
            (r'^What are ', ['What\'re ', 'List ', 'Tell me about ']),

            # Who -> Which person / Can you tell me who
            (r'^Who ', ['Which person ', 'Can you tell me who ', 'I\'d like to know who ']),
            (r'^Who is ', ['Who\'s ', 'Tell me who is ', 'Can you explain who is ']),

            # How -> In what way / What's the way to
            (r'^How ', ['In what way ', 'What\'s the method to ', 'Can you explain how ']),
            (r'^How does ', ['In what way does ', 'Can you explain how does ']),
            (r'^How can ', ['What\'s the way to ', 'Is there a way to ']),

            # Why -> What's the reason / For what reason
            (r'^Why ', ['What\'s the reason ', 'For what reason ', 'Can you explain why ']),

            # Where -> In what location / What place
            (r'^Where ', ['In what location ', 'What place ', 'Can you tell me where ']),

            # When -> At what time / What time
            (r'^When ', ['At what time ', 'What time ', 'Can you tell me when ']),
        ]

        import re
        for pattern, replacements in transforms:
            if re.match(pattern, q, re.IGNORECASE):
                replacement = random.choice(replacements)
                q = re.sub(pattern, replacement, q, count=1, flags=re.IGNORECASE)
                break

        # If no transform matched, add a prefix
        if q == question.strip():
            prefixes = [
                "Can you tell me: ",
                "I'd like to know: ",
                "Please explain: ",
                "Help me understand: ",
            ]
            q = random.choice(prefixes) + q

        return q


if __name__ == "__main__":
    print("This module is meant to be used via menu.py")
    print("Run: python menu.py <json_file>")

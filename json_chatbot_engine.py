"""
JSON Chatbot Engine
===================

Complete chatbot processing engine for JSON Q&A data.
Combines all functionality: data loading, similarity search, rephrasing, and pipeline.

Architecture inspired by SecondBrain + process_chat_event.py:
1. Load and index JSON Q&A pairs
2. Similarity search with sentence transformers (questions)
3. Rephrase + retry with LLM
4. Answer-based similarity search (fallback to searching answer content)
5. Multi-stage fallback pipeline

Usage:
    from json_chatbot_engine import JSONChatbotEngine

    engine = JSONChatbotEngine('CSU_Progress.json')
    result = engine.process_question("What is the Healthy Indiana Plan?")
    print(result['answer'])
"""

import json
import os
import time
import pickle
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# V2 Architecture imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available - V2 architecture will be disabled")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS - JSON Q&A Structure
# ============================================================================

@dataclass
class QAPair:
    """
    Single Question-Answer pair with metadata

    Attributes:
        question: The question text
        answer: The answer text
        topic: Parent topic name
        topic_index: Index in topics list
        qa_index: Index within topic's Q&A pairs
        is_bucketed: Whether this is a bucketed Q&A
        bucket_id: Bucket identifier if bucketed
    """
    question: str
    answer: str
    topic: str
    topic_index: int
    qa_index: int
    is_bucketed: bool = False
    bucket_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'topic': self.topic,
            'topic_index': self.topic_index,
            'qa_index': self.qa_index,
            'is_bucketed': self.is_bucketed,
            'bucket_id': self.bucket_id
        }

    def __repr__(self) -> str:
        return f"QAPair(topic='{self.topic}', Q='{self.question[:50]}...')"


@dataclass
class Topic:
    """
    Topic with metadata and Q&A pairs

    Attributes:
        topic: Topic name/title
        semantic_path: Semantic URL path
        original_url: Original source URL
        browser_content: Raw browser content
        qa_pairs: List of QAPair objects
    """
    topic: str
    semantic_path: str
    original_url: str
    browser_content: str
    qa_pairs: List[QAPair] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Topic('{self.topic}', {len(self.qa_pairs)} Q&A pairs)"

    def get_all_questions(self) -> List[str]:
        """Get all questions in this topic"""
        return [qa.question for qa in self.qa_pairs]

    def search_qa(self, keyword: str) -> List[QAPair]:
        """Search Q&A pairs by keyword"""
        keyword_lower = keyword.lower()
        return [
            qa for qa in self.qa_pairs
            if keyword_lower in qa.question.lower()
            or keyword_lower in qa.answer.lower()
        ]


class QADataset:
    """
    Complete Q&A dataset with indexing and search capabilities

    Loads JSON file and creates searchable index of all Q&A pairs.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.topics: List[Topic] = []
        self.all_qa_pairs: List[QAPair] = []
        self.load_from_json(json_path)

    def load_from_json(self, json_path: str):
        """Load and index all Q&A pairs from JSON file"""
        logger.info(f"Loading JSON data from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Support both formats:
        # - New format: {"tree": {...}, "topics": [...]}
        # - Legacy format: [{topic1}, {topic2}, ...]
        if isinstance(data, dict) and 'topics' in data:
            topics_data = data['topics']
            logger.info(f"📦 New format detected (tree embedded)")
        else:
            topics_data = data
            logger.info(f"📦 Legacy format detected (array)")

        for topic_idx, topic_data in enumerate(topics_data):
            qa_pairs = []

            for qa_idx, qa_data in enumerate(topic_data.get('qa_pairs', [])):
                qa_pair = QAPair(
                    question=qa_data['question'],
                    answer=qa_data['answer'],
                    topic=topic_data['topic'],
                    topic_index=topic_idx,
                    qa_index=qa_idx,
                    is_bucketed=qa_data.get('is_bucketed', False),
                    bucket_id=qa_data.get('bucket_id')
                )
                qa_pairs.append(qa_pair)
                self.all_qa_pairs.append(qa_pair)

            topic = Topic(
                topic=topic_data['topic'],
                semantic_path=topic_data.get('semantic_path', ''),
                original_url=topic_data.get('original_url', ''),
                browser_content=topic_data.get('browser_content', ''),
                qa_pairs=qa_pairs
            )
            self.topics.append(topic)

        logger.info(f"✅ Loaded {len(self.topics)} topics with {len(self.all_qa_pairs)} Q&A pairs")

    def get_all_questions(self) -> List[str]:
        """Get all questions across all topics"""
        return [qa.question for qa in self.all_qa_pairs]

    def get_all_answers(self) -> List[str]:
        """Get all answers across all topics"""
        return [qa.answer for qa in self.all_qa_pairs]

    def search_by_keyword(self, keyword: str) -> List[QAPair]:
        """Simple keyword search across all Q&A pairs"""
        keyword_lower = keyword.lower()
        return [
            qa for qa in self.all_qa_pairs
            if keyword_lower in qa.question.lower()
            or keyword_lower in qa.answer.lower()
        ]

    def get_topic_names(self) -> List[str]:
        """Get all topic names"""
        return [topic.topic for topic in self.topics]


# ============================================================================
# SIMILARITY SEARCH ENGINE - Semantic Matching
# ============================================================================

class SimilaritySearchEngine:
    """
    Similarity search using sentence transformers

    Similar to SecondBrain.similaritySearch() but for JSON Q&A pairs.
    Uses cosine similarity to find most relevant Q&A pairs.
    """

    def __init__(self, dataset: QADataset, json_path: str = None, use_cache: bool = True):
        self.dataset = dataset

        # Thresholds (matching SecondBrain's values)
        self.SIMILARITY_THRESHOLD_IDEAL = 0.7
        self.SIMILARITY_THRESHOLD = 0.6

        # Answer-based search thresholds (lower because question-vs-answer semantic similarity is naturally weaker)
        self.ANSWER_THRESHOLD_IDEAL = 0.5
        self.ANSWER_THRESHOLD_MINIMAL = 0.4

        # Cross-encoder reranker thresholds (quora model outputs duplicate probability 0-1)
        self.RERANKER_THRESHOLD_IDEAL = 0.8   # High confidence duplicate question
        self.RERANKER_THRESHOLD_MINIMAL = 0.5  # At least 50% likely the same question

        logger.info("📊 Thresholds configured:")
        logger.info(f"   Semantic - Ideal: {self.SIMILARITY_THRESHOLD_IDEAL}, Minimal: {self.SIMILARITY_THRESHOLD}")
        logger.info(f"   Answer   - Ideal: {self.ANSWER_THRESHOLD_IDEAL}, Minimal: {self.ANSWER_THRESHOLD_MINIMAL}")
        logger.info(f"   Reranker - Ideal: {self.RERANKER_THRESHOLD_IDEAL}, Minimal: {self.RERANKER_THRESHOLD_MINIMAL}")

        # Initialize sentence transformer model
        logger.info("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        #self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer('BAAI/bge-m3')

        # Enable multi-processing to use all available CPU cores
        import os
        import torch
        cpu_count = os.cpu_count() or 4

        # Set number of threads for PyTorch (used by sentence-transformers)
        # CRITICAL FIX: Limit threads to avoid thrashing in containers (Railway has 48 host cores but limited vCPU)
        max_threads = int(os.getenv('MAX_CPU_THREADS', '4'))
        effective_threads = min(cpu_count, max_threads)
        torch.set_num_threads(effective_threads)

        logger.info(f"   Using {effective_threads} CPU cores for encoding (Host has {cpu_count})")

        # Encoded embeddings and hashes
        self.encoded_questions = None
        self.encoded_answers = None
        self.encoded_topics = None
        self.question_hashes = []
        self.answer_hashes = []

        # Cache handling - create unique cache file per JSON file
        if json_path:
            # Create cache filename based on JSON filename
            cache_file = json_path.replace('.json', '_qa_cache.pkl')
            logger.info(f"Using cache file: {cache_file}")
        else:
            cache_file = 'json_qa_cache.pkl'  # Fallback for backward compatibility

        if use_cache and os.path.exists(cache_file):
            logger.info(f"✅ Found existing cache: {cache_file}")
            self._load_cache(cache_file)
        else:
            logger.info("Encoding Q&A pairs and topics...")
            self._encode_all()
            if use_cache:
                self._save_cache(cache_file)

        # Initialize BM25 for keyword matching
        logger.info("Initializing BM25 for keyword search...")
        from rank_bm25 import BM25Okapi

        # Create corpus combining questions and answers
        self.bm25_corpus = []
        for qa in dataset.all_qa_pairs:
            doc = qa.question + " " + qa.answer
            self.bm25_corpus.append(doc)

        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]

        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("✅ BM25 keyword search initialized")

        # Initialize Cross-Encoder for reranking (quora model: question-question similarity)
        logger.info("Loading cross-encoder model for reranking (quora-distilroberta-base)...")
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder('cross-encoder/quora-distilroberta-base')
        logger.info("✅ Cross-encoder reranker initialized (question-question similarity)")

        logger.info("✅ Similarity search engine ready (hybrid + reranking)")

    def _compute_hash(self, text: str) -> str:
        """Compute MD5 hash of text for caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _encode_all(self):
        """Encode all questions, answers, and topics"""
        questions = self.dataset.get_all_questions()
        answers = self.dataset.get_all_answers()
        topics = self.dataset.get_topic_names()

        # Compute hashes
        self.question_hashes = [self._compute_hash(q) for q in questions]
        self.answer_hashes = [self._compute_hash(a) for a in answers]

        # Get CPU count for parallel processing
        import os
        # Use the same logic as init to avoid over-subscription
        cpu_count = os.cpu_count() or 4
        max_threads = int(os.getenv('MAX_CPU_THREADS', '4'))
        effective_threads = min(cpu_count, max_threads)

        logger.info(f"Encoding {len(questions)} questions...")
        self.encoded_questions = self.encoder.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32,
            normalize_embeddings=False
        )

        logger.info(f"Encoding {len(answers)} answers...")
        self.encoded_answers = self.encoder.encode(
            answers,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32,
            normalize_embeddings=False
        )

        logger.info(f"Encoding {len(topics)} topics...")
        self.encoded_topics = self.encoder.encode(
            topics,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=16,
            normalize_embeddings=False
        )

    def _save_cache(self, cache_file: str):
        """Save encoded data to cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'encoded_questions': self.encoded_questions,
                    'encoded_answers': self.encoded_answers,
                    'encoded_topics': self.encoded_topics,
                    'question_hashes': self.question_hashes,
                    'answer_hashes': self.answer_hashes,
                    'qa_count': len(self.dataset.all_qa_pairs),
                    'topic_count': len(self.dataset.topics)
                }, f)
            logger.info(f"✅ Cached encodings to {cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to cache encodings: {e}")

    def _load_cache(self, cache_file: str):
        """Load encoded data from cache with incremental update support"""
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)

            # Check if cache has hashes (new format)
            if 'question_hashes' not in cached:
                logger.warning("⚠️  Legacy cache format detected. Performing full re-encode...")
                self._encode_all()
                self._save_cache(cache_file)
                return

            # Get current data
            current_questions = self.dataset.get_all_questions()
            current_answers = self.dataset.get_all_answers()
            
            # Compute current hashes
            current_q_hashes = [self._compute_hash(q) for q in current_questions]
            current_a_hashes = [self._compute_hash(a) for a in current_answers]

            # ---------------------------------------------------------
            # INCREMENTAL UPDATE LOGIC
            # ---------------------------------------------------------
            
            # 1. Map cached hashes to embeddings
            cached_q_hashes = cached['question_hashes']
            cached_q_embeddings = cached['encoded_questions']
            
            # Create lookup: hash -> embedding
            # Handle duplicates by taking the first occurrence (same text = same embedding)
            q_hash_to_emb = {}
            for h, emb in zip(cached_q_hashes, cached_q_embeddings):
                q_hash_to_emb[h] = emb

            # 2. Reconstruct Question Embeddings
            new_q_embeddings = []
            q_indices_to_encode = []
            q_texts_to_encode = []

            reused_count = 0
            
            for i, h in enumerate(current_q_hashes):
                if h in q_hash_to_emb:
                    new_q_embeddings.append(q_hash_to_emb[h])
                    reused_count += 1
                else:
                    new_q_embeddings.append(None) # Placeholder
                    q_indices_to_encode.append(i)
                    q_texts_to_encode.append(current_questions[i])

            # 3. Encode new questions if any
            if q_texts_to_encode:
                logger.info(f"🔄 Incremental update: Encoding {len(q_texts_to_encode)} new/changed questions...")
                new_embeddings = self.encoder.encode(
                    q_texts_to_encode,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    batch_size=32,
                    normalize_embeddings=False
                )
                
                # Fill placeholders
                for idx, emb in zip(q_indices_to_encode, new_embeddings):
                    new_q_embeddings[idx] = emb
            
            self.encoded_questions = np.array(new_q_embeddings)
            self.question_hashes = current_q_hashes
            
            logger.info(f"   Questions: Reused {reused_count}, Encoded {len(q_texts_to_encode)}")

            # ---------------------------------------------------------
            # REPEAT FOR ANSWERS
            # ---------------------------------------------------------
            cached_a_hashes = cached['answer_hashes']
            cached_a_embeddings = cached['encoded_answers']
            
            a_hash_to_emb = {h: emb for h, emb in zip(cached_a_hashes, cached_a_embeddings)}
            
            new_a_embeddings = []
            a_indices_to_encode = []
            a_texts_to_encode = []
            
            for i, h in enumerate(current_a_hashes):
                if h in a_hash_to_emb:
                    new_a_embeddings.append(a_hash_to_emb[h])
                else:
                    new_a_embeddings.append(None)
                    a_indices_to_encode.append(i)
                    a_texts_to_encode.append(current_answers[i])
            
            if a_texts_to_encode:
                logger.info(f"🔄 Incremental update: Encoding {len(a_texts_to_encode)} new/changed answers...")
                new_embeddings = self.encoder.encode(
                    a_texts_to_encode,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    batch_size=32,
                    normalize_embeddings=False
                )
                for idx, emb in zip(a_indices_to_encode, new_embeddings):
                    new_a_embeddings[idx] = emb
            
            self.encoded_answers = np.array(new_a_embeddings)
            self.answer_hashes = current_a_hashes

            # ---------------------------------------------------------
            # TOPICS (Always re-encode as they are few)
            # ---------------------------------------------------------
            # We could optimize this too, but topics are usually < 100, so it's negligible (ms)
            topics = self.dataset.get_topic_names()
            self.encoded_topics = self.encoder.encode(
                topics,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )

            logger.info(f"✅ Loaded & Updated Cache: {len(self.encoded_questions)} Q&As ready")
            
            # Save updated cache if we made changes
            if q_texts_to_encode or a_texts_to_encode:
                self._save_cache(cache_file)

        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}, re-encoding...")
            self._encode_all()
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}, re-encoding...")
            self._encode_all()

    def similarity_search(self, user_question: str, search_answers: bool = False, top_k: int = 1) -> Dict:
        """
        Search for most similar Q&A pairs using semantic similarity

        Args:
            user_question: User's input question
            search_answers: If True, also search in answers
            top_k: Number of top candidates to return (default=1 for backward compatibility)

        Returns:
            Dict with:
            If top_k=1:
                - best_match: QAPair object
                - score: similarity score (0-1)
                - matched_by: 'question' or 'answer'
                - duration: search time in seconds
                - meets_ideal_threshold: bool
                - meets_minimal_threshold: bool
            If top_k>1:
                - candidates: List of top-K QAPair objects
                - scores: List of similarity scores
                - matched_by: 'question' or 'answer'
                - duration: search time in seconds
        """
        start_time = time.time()

        # Encode user question
        user_embedding = self.encoder.encode([user_question])[0]

        # Calculate similarities with questions
        question_similarities = np.dot(self.encoded_questions, user_embedding) / (
            np.linalg.norm(self.encoded_questions, axis=1) * np.linalg.norm(user_embedding)
        )

        # Default to question-based search
        similarities = question_similarities
        matched_by = 'question'

        # Optionally search answers too
        if search_answers:
            answer_similarities = np.dot(self.encoded_answers, user_embedding) / (
                np.linalg.norm(self.encoded_answers, axis=1) * np.linalg.norm(user_embedding)
            )
            # Use answer similarities if they're better
            if np.max(answer_similarities) > np.max(question_similarities):
                similarities = answer_similarities
                matched_by = 'answer'

        duration = time.time() - start_time

        if top_k == 1:
            # Single best match (backward compatible)
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            best_match = self.dataset.all_qa_pairs[best_idx]

            logger.info(f"Semantic search: score={best_score:.4f}, matched_by={matched_by}, duration={duration:.3f}s")

            return {
                'best_match': best_match,
                'score': float(best_score),
                'matched_by': matched_by,
                'duration': duration,
                'meets_ideal_threshold': best_score >= self.SIMILARITY_THRESHOLD_IDEAL,
                'meets_minimal_threshold': best_score >= self.SIMILARITY_THRESHOLD
            }
        else:
            # Top-K candidates for reranking
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_scores = similarities[top_k_indices]
            top_k_candidates = [self.dataset.all_qa_pairs[idx] for idx in top_k_indices]

            logger.info(f"Semantic search (top-{top_k}): best_score={top_k_scores[0]:.4f}, matched_by={matched_by}, duration={duration:.3f}s")

            return {
                'candidates': top_k_candidates,
                'scores': top_k_scores.tolist(),
                'matched_by': matched_by,
                'duration': duration
            }

    def search(self, user_question: str, search_answers: bool = False) -> Dict:
        """
        Search for most similar Q&A pair (wrapper for backward compatibility)
        """
        return self.similarity_search(user_question, search_answers, top_k=1)

    def hybrid_search(self, user_question: str, alpha=0.85, search_answers: bool = False, top_k: int = 1) -> Dict:
        """
        Hybrid search combining BM25 keyword matching and semantic similarity

        Args:
            user_question: User's input question
            alpha: Weight for semantic vs BM25 (default 0.85 = 85% semantic, 15% BM25)
            search_answers: If True, also search in answers
            top_k: Number of top candidates to return (default 1 for backward compatibility)

        Returns:
            Dict with:
            - best_match: QAPair object (or list if top_k > 1)
            - candidates: List of top-K QAPair objects (if top_k > 1)
            - scores: List of top-K scores (if top_k > 1)
            - score: combined hybrid score (0-1) of best match
            - semantic_score: semantic similarity score
            - bm25_score: BM25 keyword score
            - matched_by: 'hybrid'
            - duration: search time in seconds
            - meets_ideal_threshold: bool
            - meets_minimal_threshold: bool
        """
        start_time = time.time()

        # 1. SEMANTIC SEARCH
        user_embedding = self.encoder.encode([user_question])[0]

        if search_answers:
            # Search in both questions and answers
            question_similarities = np.dot(self.encoded_questions, user_embedding) / (
                np.linalg.norm(self.encoded_questions, axis=1) * np.linalg.norm(user_embedding)
            )
            answer_similarities = np.dot(self.encoded_answers, user_embedding) / (
                np.linalg.norm(self.encoded_answers, axis=1) * np.linalg.norm(user_embedding)
            )
            semantic_scores = np.maximum(question_similarities, answer_similarities)
        else:
            # Search only in questions
            semantic_scores = np.dot(self.encoded_questions, user_embedding) / (
                np.linalg.norm(self.encoded_questions, axis=1) * np.linalg.norm(user_embedding)
            )

        # 2. BM25 KEYWORD SEARCH
        tokenized_query = user_question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range (same scale as semantic)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # 3. HANDLE SIZE MISMATCH (safety check)
        if len(semantic_scores) != len(bm25_scores):
            logger.warning(f"⚠️  Size mismatch: semantic={len(semantic_scores)}, bm25={len(bm25_scores)}")
            # Resize to match the smaller size (safer than padding)
            min_size = min(len(semantic_scores), len(bm25_scores))
            semantic_scores = semantic_scores[:min_size]
            bm25_scores = bm25_scores[:min_size]
            logger.warning(f"   Resized both to {min_size} elements")

        # 4. COMBINE SCORES WITH ALPHA WEIGHTING
        hybrid_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

        # 5. GET TOP-K MATCHES
        if top_k == 1:
            # Single best match (backward compatible)
            best_idx = np.argmax(hybrid_scores)
            best_score = hybrid_scores[best_idx]
            best_match = self.dataset.all_qa_pairs[best_idx]

            duration = time.time() - start_time

            # Determine which method dominated the result
            semantic_contribution = alpha * semantic_scores[best_idx]
            bm25_contribution = (1 - alpha) * bm25_scores[best_idx]

            if semantic_contribution > bm25_contribution:
                dominant_method = "SEMANTIC"
            elif bm25_contribution > semantic_contribution:
                dominant_method = "BM25"
            else:
                dominant_method = "EQUAL"

            # Log results with dominant method
            logger.info(f"Hybrid search: semantic={semantic_scores[best_idx]:.4f}, "
                       f"bm25={bm25_scores[best_idx]:.4f}, "
                       f"hybrid={best_score:.4f}, 🎯 DOMINATED BY: {dominant_method}, duration={duration:.3f}s")

            return {
                'best_match': best_match,
                'score': float(best_score),
                'semantic_score': float(semantic_scores[best_idx]),
                'bm25_score': float(bm25_scores[best_idx]),
                'matched_by': 'hybrid',
                'duration': duration,
                'meets_ideal_threshold': best_score >= self.SIMILARITY_THRESHOLD_IDEAL,
                'meets_minimal_threshold': best_score >= self.SIMILARITY_THRESHOLD
            }
        else:
            # Top-K candidates for reranking
            top_k_indices = np.argsort(hybrid_scores)[::-1][:top_k]
            top_k_scores = hybrid_scores[top_k_indices]
            top_k_candidates = [self.dataset.all_qa_pairs[idx] for idx in top_k_indices]

            duration = time.time() - start_time

            logger.info(f"Hybrid search: Retrieved top-{top_k} candidates, "
                       f"scores range: {top_k_scores[0]:.4f} - {top_k_scores[-1]:.4f}, "
                       f"duration={duration:.3f}s")

            return {
                'candidates': top_k_candidates,
                'scores': top_k_scores.tolist(),
                'best_match': top_k_candidates[0],
                'score': float(top_k_scores[0]),
                'matched_by': 'hybrid_topk',
                'duration': duration,
                'meets_ideal_threshold': top_k_scores[0] >= self.SIMILARITY_THRESHOLD_IDEAL,
                'meets_minimal_threshold': top_k_scores[0] >= self.SIMILARITY_THRESHOLD
            }

    def search_topic(self, user_question: str) -> Optional[Tuple[int, float]]:
        """
        Search for most relevant topic

        Args:
            user_question: User's input question

        Returns:
            Tuple of (topic_index, score) or None if no match
        """
        user_embedding = self.encoder.encode([user_question])[0]

        similarities = np.dot(self.encoded_topics, user_embedding) / (
            np.linalg.norm(self.encoded_topics, axis=1) * np.linalg.norm(user_embedding)
        )

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Lower threshold for topic matching (0.5 vs 0.6 for Q&A)
        if best_score >= 0.3:
            logger.info(f"Topic match: '{self.dataset.topics[best_idx].topic}' (score={best_score:.4f})")
            return best_idx, float(best_score)

        return None

    def rerank_with_cross_encoder(self, user_question: str, candidates: List, hybrid_scores: List[float] = None) -> Dict:
        """
        Rerank top-K candidates using cross-encoder for higher precision

        Args:
            user_question: User's input question
            candidates: List of QAPair objects to rerank
            hybrid_scores: Optional list of hybrid scores for logging

        Returns:
            Dict with:
            - best_match: Best QAPair after reranking
            - score: Cross-encoder relevance score
            - reranked_candidates: List of candidates sorted by cross-encoder score
            - rerank_scores: List of cross-encoder scores
            - matched_by: 'reranked'
            - duration: reranking time
        """
        start_time = time.time()

        # Prepare query-candidate pairs for cross-encoder
        pairs = [(user_question, cand.question) for cand in candidates]

        # Get cross-encoder scores (higher = more relevant)
        rerank_scores = self.reranker.predict(pairs)

        # Normalize cross-encoder scores to 0-1 range using sigmoid
        # Cross-encoder returns logits, we need probabilities
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        rerank_scores_normalized = sigmoid(rerank_scores)

        # Sort by reranker score (descending)
        sorted_indices = np.argsort(rerank_scores_normalized)[::-1]
        reranked_candidates = [candidates[i] for i in sorted_indices]
        sorted_scores = [rerank_scores_normalized[i] for i in sorted_indices]

        best_match = reranked_candidates[0]
        best_score = sorted_scores[0]

        duration = time.time() - start_time

        # Log reranking results
        if hybrid_scores:
            logger.info(f"🔄 Reranked {len(candidates)} candidates:")
            logger.info(f"   Before: hybrid_score={hybrid_scores[0]:.4f}, question=\"{candidates[0].question[:50]}...\"")
            logger.info(f"   After:  rerank_score={best_score:.4f}, question=\"{best_match.question[:50]}...\"")
            if candidates[0] != best_match:
                logger.info(f"   ⚠️ RERANKER CHANGED THE RESULT!")
        else:
            logger.info(f"🔄 Reranking: best_score={best_score:.4f}, duration={duration:.3f}s")

        return {
            'best_match': best_match,
            'score': float(best_score),
            'reranked_candidates': reranked_candidates,
            'rerank_scores': sorted_scores,
            'matched_by': 'reranked',
            'duration': duration,
            'meets_ideal_threshold': best_score >= self.RERANKER_THRESHOLD_IDEAL,
            'meets_minimal_threshold': best_score >= self.RERANKER_THRESHOLD_MINIMAL
        }

    # Removed hardcoded pattern-based validation methods:
    # - is_person_query()
    # - validate_person_entity()
    #
    # These methods used hardcoded patterns that don't generalize to different JSON files.
    # Relying purely on semantic similarity and cross-encoder reranking scores instead.


# ============================================================================
# REPHRASING ENGINE - LLM-based Question Rephrasing
# ============================================================================

class QuestionRephraser:
    """
    Rephrases user questions using Groq LLM

    Similar to SecondBrain.rephrase() but using Groq instead of Syra API.
    Helps improve matching by simplifying and standardizing user questions.
    """

    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            logger.warning("⚠️ GROQ_API_KEY not found - rephrasing will be disabled")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            self.model = "llama-3.3-70b-versatile"
            logger.info("✅ Question rephraser initialized")

    def rephrase(self, user_question: str, example_questions: List[str] = None) -> Tuple[Optional[str], Dict]:
        """
        Rephrase user question for better matching

        Args:
            user_question: Original user question
            example_questions: Optional list of example questions from dataset

        Returns:
            Tuple of (rephrased_text, info_dict)
        """
        if not self.client:
            logger.warning("Rephrasing skipped - no API key")
            return None, {'error': 'No API key'}

        start_time = time.time()

        # Build prompt with examples
        if example_questions and len(example_questions) > 0:
            examples = "\n".join(f"- {q}" for q in example_questions[:10])
            prompt = f"""You are helping rephrase user questions to match a Q&A knowledge base.

Example questions from the knowledge base:
{examples}

User's question: "{user_question}"

Rephrase the user's question to be:
1. Clear and concise
2. Similar in style to the examples above
3. Focused on the core intent
4. Without personal pronouns (I, me, my, we, us, our)
5. PRESERVE any names, people, locations, dates, or specific identifiers mentioned by the user
6. If asking about a specific person by name, keep the name exactly as provided

Examples of good rephrasing:
- "who is becky?" → "Who is Becky?"
- "tell me about john smith" → "What information is available about John Smith?"
- "where is the indianapolis office" → "Where is the Indianapolis office located?"
- "i need help with enrollment" → "How do I enroll?" (no name to preserve)

Return ONLY the rephrased question, nothing else."""
        else:
            prompt = f"""Rephrase this question to be clear, concise, and focused on the core intent.

IMPORTANT: PRESERVE any names, people, locations, dates, or specific identifiers.

"{user_question}"

Return ONLY the rephrased question, nothing else."""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,  # Lower temperature for more conservative, literal rephrasing
                max_tokens=100
            )

            rephrased = response.choices[0].message.content.strip()
            duration = time.time() - start_time

            logger.info(f"✅ Rephrased in {duration:.3f}s: '{user_question}' -> '{rephrased}'")

            return rephrased, {
                'original': user_question,
                'rephrased': rephrased,
                'duration': duration
            }

        except Exception as e:
            logger.error(f"❌ Rephrasing failed: {e}")
            duration = time.time() - start_time
            return None, {
                'original': user_question,
                'rephrased': None,
                'duration': duration,
                'error': str(e)
            }


# ============================================================================
# MAIN CHATBOT ENGINE - Multi-Stage Pipeline
# ============================================================================

class JSONChatbotEngine:
    """
    Main chatbot engine with multi-stage processing pipeline

    Architecture inspired by process_chat_event.py:

    STAGE 1: Primary Similarity Search (ideal threshold 0.7)
        ↓ (if score < 0.7)
    STAGE 2: Rephrase + Retry (minimal threshold 0.6)
        ↓ (if score < 0.6)
    STAGE 2.5: Answer-Based Search (search_answers=True, threshold 0.6)
        ↓ (if score < 0.6)
    STAGE 3: Topic-Level Search (threshold 0.5)
        ↓ (if no match)
    STAGE 4: Fallback Response

    Usage:
        engine = JSONChatbotEngine('CSU_Progress.json')
        result = engine.process_question("What is One Plan?")
        print(result['answer'])
    """

    def __init__(self, json_path: str, enable_rephrasing: bool = True):
        """
        Initialize chatbot engine

        Args:
            json_path: Path to JSON Q&A file
            enable_rephrasing: Whether to enable LLM rephrasing (requires GROQ_API_KEY)
        """
        logger.info("="*60)
        logger.info("JSON CHATBOT ENGINE - INITIALIZATION")
        logger.info("="*60)

        # Store json_path for reference
        self.json_path = json_path

        # Load dataset
        self.dataset = QADataset(json_path)

        # Initialize similarity search (pass json_path for unique pickle cache)
        self.search_engine = SimilaritySearchEngine(self.dataset, json_path=json_path)

        # Initialize rephraser (optional)
        self.rephraser = None
        if enable_rephrasing:
            self.rephraser = QuestionRephraser()

        logger.info("="*60)
        logger.info("✅ JSON CHATBOT ENGINE READY")
        logger.info(f"   Topics: {len(self.dataset.topics)}")
        logger.info(f"   Q&A Pairs: {len(self.dataset.all_qa_pairs)}")
        logger.info(f"   Rephrasing: {'Enabled' if self.rephraser else 'Disabled'}")
        logger.info("="*60)

        # Initialize V2 architecture components (lazy initialization)
        self.v2_enabled = False
        self.enricher_v2 = None
        self.query_parser_v2 = None
        self.parallel_retriever_v2 = None
        self.verifier_v2 = None

    def enable_v2_architecture(self):
        """
        Enable V2 parallel-fused hybrid architecture.

        This initializes the new components for metadata-enriched, parallel retrieval
        with RRF fusion and LLM verification. Falls back to V1 if dependencies missing.
        """
        if self.v2_enabled:
            logger.info("V2 architecture already enabled")
            return

        logger.info("="*70)
        logger.info("ENABLING V2 ARCHITECTURE (Parallel-Fused Hybrid)")
        logger.info("="*70)

        if not self.rephraser or not self.rephraser.client:
            logger.error("❌ V2 requires GROQ_API_KEY - V2 disabled")
            return

        try:
            # Initialize V2 components
            self.enricher_v2 = MetadataEnricher(self.rephraser.client)
            self.query_parser_v2 = QueryParserV2(self.rephraser.client)
            self.parallel_retriever_v2 = ParallelRetrieverV2(self.search_engine)
            self.verifier_v2 = LLMVerifierV2(self.rephraser.client)

            # Enrich dataset with metadata
            self.parallel_retriever_v2.enrich_dataset(self.enricher_v2, self.json_path)

            self.v2_enabled = True
            logger.info("="*70)
            logger.info("✅ V2 ARCHITECTURE ENABLED")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"❌ Failed to enable V2 architecture: {e}")
            self.v2_enabled = False

    def process_question_v2(self, user_question: str) -> Dict[str, Any]:
        """
        Process question using V2 parallel-fused architecture.

        This method implements the team lead's recommended architecture:
        1. Query parsing (intent + entity extraction)
        2. Parallel retrieval (4 retrievers)
        3. RRF fusion
        4. LLM verification (NLI)

        Args:
            user_question: User's input question

        Returns:
            Dict with answer, confidence, and metadata
        """
        if not self.v2_enabled:
            logger.warning("V2 architecture not enabled - call enable_v2_architecture() first")
            return self.process_question(user_question)  # Fallback to V1

        start_time = time.time()

        logger.info("="*70)
        logger.info(f"PROCESSING QUERY (V2): '{user_question}'")
        logger.info("="*70)

        # Step 1: Query Analysis
        logger.info("\n📊 STEP 1: Query Analysis (LLM)")
        query_analysis = self.query_parser_v2.parse(user_question)
        logger.info(f"   Intent: {query_analysis.intent}")
        logger.info(f"   Entities: {query_analysis.entities}")
        logger.info(f"   Semantic Query: {query_analysis.semantic_query}")

        # Step 2: Parallel Retrieval + RRF Fusion
        logger.info("\n🔍 STEP 2: Parallel Retrieval + RRF Fusion")
        # Increased from top_k=10 to top_k=30 for better coverage (Improvement #3)
        # With 6,279 Q&A pairs, top-30 gives 3x better recall
        candidates, retrieval_details = self.parallel_retriever_v2.retrieve(query_analysis, top_k=30)

        if not candidates:
            logger.warning("No candidates found!")
            return self._fallback_response_v2()

        logger.info(f"\n🏆 Top 3 RRF candidates:")
        for i, cand in enumerate(candidates[:3], 1):
            logger.info(f"   {i}. [RRF Score: {cand.score:.4f}] {cand.qa_pair.question[:60]}...")

        # Step 3: LLM Verification
        logger.info("\n✓ STEP 3: LLM Verification (NLI)")
        verified = self.verifier_v2.verify(user_question, candidates, top_k=1)

        if not verified:
            logger.warning("⚠️ No candidates passed verification - using top RRF candidate")
            verified = [candidates[0]]  # Fallback to top RRF if verification too strict

        # Return verified answer
        best_candidate = verified[0]
        duration = time.time() - start_time

        # CONFIDENCE THRESHOLD CHECK - Return fallback if score too low
        # Note: RRF scores are typically 0.01-0.1 range (not 0-1 like similarity scores)
        # RRF formula: sum(1/(k+rank)) where k=60, so max ~0.26 for perfect ranks
        MIN_RRF_THRESHOLD = 0.02  # 2% RRF score threshold (lowered: LLM verification is the real gatekeeper)
        if best_candidate.score < MIN_RRF_THRESHOLD:
            logger.warning(f"⚠️ RRF score {best_candidate.score:.4f} below threshold {MIN_RRF_THRESHOLD} - returning fallback")
            fallback = self._fallback_response_v2()
            fallback['pipeline_info'] = {
                'architecture': 'v2_parallel_fused',
                'query_analysis': {
                    'intent': query_analysis.intent,
                    'entities': query_analysis.entities,
                    'semantic_query': query_analysis.semantic_query
                },
                'retrieval_details': retrieval_details,
                'candidates_evaluated': len(candidates),
                'candidates_verified': len(verified),
                'rejected_reason': f'RRF score {best_candidate.score:.4f} < threshold {MIN_RRF_THRESHOLD}',
                'duration': duration
            }
            return fallback

        logger.info("\n"+"="*70)
        logger.info(f"✅ V2 ANSWER FOUND in {duration:.2f}s")
        logger.info(f"   Question: {best_candidate.qa_pair.question}")
        logger.info(f"   Answer: {best_candidate.qa_pair.answer[:100]}...")
        logger.info("="*70)

        return {
            'answer': best_candidate.qa_pair.answer,
            'source_qa': best_candidate.qa_pair,
            'matched_by': 'v2_parallel_fused_verified',
            'confidence': float(best_candidate.score),
            'source_topic': best_candidate.qa_pair.topic,
            'source_qa_index': best_candidate.qa_pair.qa_index,
            'duration': duration,
            'pipeline_info': {
                'architecture': 'v2_parallel_fused',
                'query_analysis': {
                    'intent': query_analysis.intent,
                    'entities': query_analysis.entities,
                    'semantic_query': query_analysis.semantic_query
                },
                'retrieval_details': retrieval_details,
                'candidates_evaluated': len(candidates),
                'candidates_verified': len(verified),
                'duration': duration
            }
        }

    def _fallback_response_v2(self) -> Dict[str, Any]:
        """Fallback when no answer found (V2)"""
        return {
            'answer': "I don't have specific information to answer that question. Please try rephrasing or ask about a different topic.",
            'source_qa': None,
            'matched_by': 'v2_fallback',
            'confidence': 0.0,
            'source_topic': None,
            'source_qa_index': None,
            'duration': 0.0
        }

    # =========================================================================
    # V3 ARCHITECTURE: Gemini + Q&A Embeddings
    # =========================================================================
    #
    # How it works:
    # 1. Combine each Question + Answer into one text
    # 2. Gemini embeds it into a 768-dim vector (captures MEANING)
    # 3. User query → Gemini embedding
    # 4. Cosine similarity → find most similar Q+A
    #
    # Result: 86% top-1, 99% top-3 accuracy
    # =========================================================================

    def enable_v3_architecture(self):
        """Enable V3: Gemini + Q&A embeddings"""
        import google.generativeai as genai
        import os
        import pickle

        logger.info("="*70)
        logger.info("ENABLING V3: Gemini + Q&A Embeddings")
        logger.info("="*70)

        # Configure Gemini
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            logger.error("❌ No GOOGLE_API_KEY or GEMINI_API_KEY found")
            self.v3_enabled = False
            return

        genai.configure(api_key=api_key)
        self.genai = genai

        # Get all Q+A pairs with topic info
        all_qa = self.search_engine.dataset.all_qa_pairs
        self.v3_questions = [qa.question for qa in all_qa]
        self.v3_answers = [qa.answer for qa in all_qa]
        self.v3_topics = [qa.topic for qa in all_qa]

        # Get URLs from topics dataset
        topic_urls = {t.topic: t.original_url for t in self.search_engine.dataset.topics}
        self.v3_urls = [topic_urls.get(qa.topic, '') for qa in all_qa]

        # Cache file - use /app/data on Railway, local config/cache otherwise
        data_hash = hashlib.md5(f"{len(all_qa)}_{self.v3_questions[0][:50]}".encode()).hexdigest()[:12]
        if os.path.exists('/app/data'):
            cache_dir = '/app/data/cache'
        else:
            cache_dir = os.path.join(os.path.dirname(self.json_path), 'config', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'gemini_v3_{data_hash}.pkl')

        # Try load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.v3_embeddings = pickle.load(f)
                logger.info(f"✅ Loaded cached Gemini embeddings ({len(self.v3_embeddings)} Q+As)")
                self.v3_enabled = True
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Compute embeddings
        logger.info(f"🔄 Computing Gemini embeddings for {len(all_qa)} Q+As...")
        self.v3_embeddings = []

        for i, qa in enumerate(all_qa):
            combined = f"{qa.question} {qa.answer}"[:2000]
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=combined,
                    task_type="retrieval_document"
                )
                self.v3_embeddings.append(result['embedding'])
            except Exception as e:
                logger.error(f"Embedding failed for Q+A {i}: {e}")
                self.v3_embeddings.append([0] * 768)

            if (i + 1) % 100 == 0:
                logger.info(f"   ... {i+1}/{len(all_qa)}")

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.v3_embeddings, f)
        logger.info(f"💾 Cached embeddings to {cache_file}")

        self.v3_enabled = True
        logger.info("✅ V3 Architecture enabled!")

    def process_question_v3(self, user_question: str) -> Dict[str, Any]:
        """
        Process question using V3: Gemini + Q&A embeddings.

        Simple pipeline:
        1. Embed user query with Gemini
        2. Cosine similarity against all Q+A embeddings
        3. Return best match
        """
        if not hasattr(self, 'v3_enabled') or not self.v3_enabled:
            logger.warning("V3 not enabled - call enable_v3_architecture() first")
            return self.process_question(user_question)

        start_time = time.time()

        logger.info(f"\n🔍 V3 Search: '{user_question}'")

        # Embed query
        result = self.genai.embed_content(
            model="models/text-embedding-004",
            content=user_question,
            task_type="retrieval_query"
        )
        query_emb = np.array(result['embedding'])

        # Cosine similarity
        doc_embs = np.array(self.v3_embeddings)
        similarities = np.dot(doc_embs, query_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
        )

        # Top result
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        duration = time.time() - start_time

        logger.info(f"   ✅ Match: score={best_score:.4f}, idx={best_idx}")
        logger.info(f"   Q: {self.v3_questions[best_idx][:60]}...")

        return {
            'answer': self.v3_answers[best_idx],
            'source_qa': {
                'question': self.v3_questions[best_idx],
                'answer': self.v3_answers[best_idx],
                'topic': self.v3_topics[best_idx]
            },
            'source_topic': self.v3_topics[best_idx],
            'source_url': self.v3_urls[best_idx],
            'matched_by': 'v3_gemini_qa',
            'confidence': best_score,
            'source_qa_index': best_idx,
            'duration': duration
        }

    def process_question(self, user_question: str, session_id: str = "default") -> Dict:
        """
        Process user question through multi-stage pipeline (Full-Scan mode).
        
        All stages run unconditionally. No early returns. The best candidate
        across all stages wins at the end.

        Args:
            user_question: User's input question
            session_id: Session identifier for logging

        Returns:
            Dict with:
            - answer: str, the response text
            - source_qa: QAPair or None
            - matched_by: str (similarity_ideal/similarity_minimal/rephrase/answer_similarity/topic/fallback)
            - confidence: float (0-1)
            - pipeline_info: Dict with detailed stage information
        """
        logger.info("="*60)
        logger.info(f"PROCESSING QUESTION (Full-Scan): '{user_question}'")
        logger.info("="*60)

        pipeline_info = {
            'session_id': session_id,
            'original_question': user_question,
            'timestamp': datetime.now().isoformat(),
            'stages': []
        }

        # Collect all candidates from every stage — pick the best at the end
        all_candidates = []  # List of (score, qa_pair, matched_by, source)

        # ====================================================================
        # STAGE 1: Two-Stage Retrieval (Semantic Search + Cross-Encoder Reranking)
        # ====================================================================
        logger.info("\n🔍 STAGE 1: Two-Stage Retrieval (Semantic + Reranking)")

        # Step 1a: Get top-10 candidates with pure semantic search
        logger.info("   Step 1a: Semantic search for top-10 candidates")
        semantic_result = self.search_engine.similarity_search(user_question, top_k=10)
        candidates = semantic_result['candidates']
        semantic_scores = semantic_result['scores']

        logger.info(f"   📊 Semantic top match: score={semantic_scores[0]:.4f}, question=\"{candidates[0].question[:60]}...\"")

        # Step 1b: Rerank with cross-encoder
        logger.info("   Step 1b: Cross-encoder reranking")
        search_result = self.search_engine.rerank_with_cross_encoder(
            user_question,
            candidates,
            hybrid_scores=semantic_scores
        )

        stage_1_info = {
            'stage_number': 1,
            'stage_name': 'Stage 1a: Semantic (Top-10) → Stage 1b: Reranking',
            'stage': 'primary_similarity',
            'score': float(search_result['score']),
            'matched_by': search_result['matched_by'],
            'duration': float(search_result['duration']),
            'best_match_question': search_result['best_match'].question,
            'best_match_topic': search_result['best_match'].topic,
            'threshold_ideal': float(self.search_engine.RERANKER_THRESHOLD_IDEAL),
            'threshold_minimal': float(self.search_engine.RERANKER_THRESHOLD_MINIMAL),
            'meets_ideal': bool(search_result['meets_ideal_threshold']),
            'meets_minimal': bool(search_result['meets_minimal_threshold']),
            'top_k_candidates': int(len(candidates)),
            'reranker_used': True,
            'semantic_score': float(semantic_scores[0])
        }
        pipeline_info['stages'].append(stage_1_info)

        # Collect Stage 1 candidate (no early return)
        all_candidates.append({
            'score': search_result['score'],
            'qa_pair': search_result['best_match'],
            'matched_by': 'similarity_ideal' if search_result['meets_ideal_threshold'] else 'similarity_minimal',
            'source': 'stage_1_reranker'
        })
        logger.info(f"   📥 Collected: score={search_result['score']:.4f}, question=\"{search_result['best_match'].question[:60]}...\"")

        # ====================================================================
        # STAGE 2: Rephrase + Retry
        # ====================================================================
        if self.rephraser:
            logger.info("\n🔄 STAGE 2: Rephrase + Retry")

            example_questions = self.dataset.get_all_questions()[:20]
            rephrased_text, rephrase_info = self.rephraser.rephrase(
                user_question,
                example_questions=example_questions
            )

            stage_2_info = {
                'stage_number': 2,
                'stage_name': 'Rephrase + Retry',
                'stage': 'rephrase',
                'rephrased_text': rephrased_text,
                'duration': float(rephrase_info.get('duration', 0)),
                'score': 0.0
            }
            pipeline_info['stages'].append(stage_2_info)

            if rephrased_text and rephrased_text != user_question:
                logger.info(f"   Rephrased: '{rephrased_text}'")

                rephrase_hybrid = self.search_engine.hybrid_search(rephrased_text, alpha=0.85, top_k=10)
                rephrase_search = self.search_engine.rerank_with_cross_encoder(
                    rephrased_text,
                    rephrase_hybrid['candidates'],
                    hybrid_scores=rephrase_hybrid['scores']
                )

                stage_2_search_info = {
                    'stage': 'rephrase_similarity',
                    'score': float(rephrase_search['score']),
                    'matched_by': rephrase_search['matched_by'],
                    'duration': float(rephrase_search['duration']),
                    'best_match_question': rephrase_search['best_match'].question if 'best_match' in rephrase_search else None,
                    'meets_ideal': bool(rephrase_search.get('meets_ideal_threshold', False)),
                    'meets_minimal': bool(rephrase_search.get('meets_minimal_threshold', False))
                }
                pipeline_info['stages'].append(stage_2_search_info)

                # Collect Stage 2 candidate (no early return)
                all_candidates.append({
                    'score': rephrase_search['score'],
                    'qa_pair': rephrase_search['best_match'],
                    'matched_by': 'rephrase_similarity',
                    'source': 'stage_2_rephrase'
                })
                logger.info(f"   📥 Collected: score={rephrase_search['score']:.4f}, question=\"{rephrase_search['best_match'].question[:60]}...\"")
            else:
                logger.info("   ⚠️ Rephrasing failed or produced same text")

        # ====================================================================
        # STAGE 2.5: Search in Answers (Extended Search)
        # ====================================================================
        logger.info("\n🔎 STAGE 2.5: Answer-Based Search (No Reranking)")
        logger.info("   Searching in answer content instead of questions...")

        answer_search_result = self.search_engine.hybrid_search(user_question, alpha=0.5, search_answers=True, top_k=1)

        meets_ideal = answer_search_result['score'] >= self.search_engine.ANSWER_THRESHOLD_IDEAL
        meets_minimal = answer_search_result['score'] >= self.search_engine.ANSWER_THRESHOLD_MINIMAL

        stage_2_5_info = {
            'stage_number': 2.5,
            'stage_name': 'Answer-Based Search (Hybrid Only)',
            'stage': 'answer_search',
            'score': float(answer_search_result['score']),
            'matched_by': answer_search_result['matched_by'],
            'duration': float(answer_search_result['duration']),
            'best_match_question': answer_search_result['best_match'].question,
            'best_match_topic': answer_search_result['best_match'].topic,
            'meets_ideal': bool(meets_ideal),
            'meets_minimal': bool(meets_minimal),
            'reranker_used': False
        }
        pipeline_info['stages'].append(stage_2_5_info)

        # Collect Stage 2.5 candidate separately (used as fallback only)
        answer_candidate = {
            'score': answer_search_result['score'],
            'qa_pair': answer_search_result['best_match'],
            'matched_by': 'answer_similarity',
            'source': 'stage_2_5_answer'
        }
        logger.info(f"   📥 Collected: score={answer_search_result['score']:.4f}, question=\"{answer_search_result['best_match'].question[:60]}...\"")

        logger.info("\n📚 STAGE 3: Topic-Level Search - SKIPPED (disabled)")

        # ====================================================================
        # FINAL: Unified Selection — All stages compete equally
        # ====================================================================
        logger.info("\n🏆 FINAL: Unified Selection (all stages equal)...")

        # Normalize scores to [0, 1] range so question-based and answer-based compete fairly
        # Question-based (reranker): scores are 0-1 (quora model probability)
        # Answer-based (hybrid): scores are 0-1 (semantic/BM25 blend)
        # Both are already in comparable ranges — just pick the best above threshold

        # Collect all candidates into one pool
        unified_candidates = []
        for c in all_candidates:
            threshold = self.search_engine.RERANKER_THRESHOLD_MINIMAL if c['source'] in ('stage_1_reranker', 'stage_2_rephrase') else self.search_engine.ANSWER_THRESHOLD_MINIMAL
            if c['score'] >= threshold:
                unified_candidates.append(c)

        # Add answer-based candidate if it meets threshold
        if answer_candidate['score'] >= self.search_engine.ANSWER_THRESHOLD_MINIMAL:
            unified_candidates.append(answer_candidate)

        # Sort all by score — highest wins regardless of stage
        unified_candidates.sort(key=lambda c: c['score'], reverse=True)

        # Log all candidates for diagnostics
        all_for_log = list(all_candidates) + [answer_candidate]
        all_for_log.sort(key=lambda c: c['score'], reverse=True)
        for i, cand in enumerate(all_for_log):
            marker = "👑" if i == 0 else "  "
            logger.info(f"   {marker} #{i+1}: score={cand['score']:.4f} | {cand['source']} | \"{cand['qa_pair'].question[:60]}...\"")

        best = unified_candidates[0] if unified_candidates else None

        if best is not None:
            logger.info(f"\n✅ BEST MATCH: {best['matched_by']} (score: {best['score']:.4f})")
            logger.info(f"   Question: {best['qa_pair'].question}")
            logger.info(f"   Topic: {best['qa_pair'].topic}")
            logger.info(f"   Source: {best['source']}")
            logger.info("="*60)

            return {
                'answer': best['qa_pair'].answer,
                'source_qa': best['qa_pair'],
                'matched_by': best['matched_by'],
                'confidence': best['score'],
                'pipeline_info': pipeline_info
            }

        # ====================================================================
        # STAGE 4: Fallback Response (no candidate met threshold)
        # ====================================================================
        # Determine best score across all stages for diagnostics
        all_for_fallback = question_candidates + [answer_candidate]
        all_for_fallback.sort(key=lambda c: c['score'], reverse=True)
        best_overall_score = all_for_fallback[0]['score'] if all_for_fallback else 0.0

        logger.info("\n❌ STAGE 4: Fallback Response")
        logger.info(f"   Best score across all stages was {best_overall_score:.4f} but below threshold — returning fallback")

        stage_4_info = {
            'stage_number': 4,
            'stage_name': 'Fallback Response',
            'stage': 'fallback',
            'score': 0.0
        }
        pipeline_info['stages'].append(stage_4_info)

        logger.info("="*60)

        fallback_answer = (
            "I don't have specific information about that question. "
            "Here are some topics I can help with:\n\n"
        )

        for i, topic in enumerate(self.dataset.topics[:10], 1):
            fallback_answer += f"{i}. {topic.topic}\n"

        if len(self.dataset.topics) > 10:
            fallback_answer += f"\n...and {len(self.dataset.topics) - 10} more topics."

        fallback_answer += "\n\nPlease ask about one of these topics, or try rephrasing your question."

        return {
            'answer': fallback_answer,
            'source_qa': None,
            'matched_by': 'fallback',
            'confidence': 0.0,
            'pipeline_info': pipeline_info,
            'suggested_topics': [t.topic for t in self.dataset.topics[:10]]
        }

    def get_all_topics(self) -> List[Dict]:
        """Get all topics with metadata"""
        return [
            {
                'name': topic.topic,
                'url': topic.original_url,
                'semantic_path': topic.semantic_path,
                'qa_count': len(topic.qa_pairs)
            }
            for topic in self.dataset.topics
        ]

    def get_topic_qa_pairs(self, topic_name: str) -> Optional[List[Dict]]:
        """Get all Q&A pairs for a specific topic"""
        for topic in self.dataset.topics:
            if topic.topic.lower() == topic_name.lower():
                return [qa.to_dict() for qa in topic.qa_pairs]
        return None


# ============================================================================
# V2 ARCHITECTURE: PARALLEL-FUSED HYBRID WITH METADATA ENRICHMENT
# ============================================================================
# Based on architectural analysis identifying sequential fallback as flawed.
# Implements: Metadata extraction, Query parsing, Parallel retrieval, RRF fusion, LLM verification
# ============================================================================

@dataclass
class QueryAnalysisV2:
    """Structured output from LLM query parser (V2)"""
    intent: str  # e.g., "person_role_lookup", "role_person_lookup", "general_faq"
    entities: Dict[str, str] = field(default_factory=dict)
    semantic_query: str = ""


@dataclass
class RetrievalCandidate:
    """Single candidate from a retriever"""
    qa_pair: QAPair
    score: float
    retriever_name: str


class MetadataEnricher:
    """
    Enriches Q&A pairs with extracted metadata using NER + LLM.
    Part of V2 ingestion-time enrichment.
    """

    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.enabled = True
                logger.info("✅ Metadata enricher initialized (NER + LLM)")
            except OSError:
                logger.warning("spaCy model not found - run: python -m spacy download en_core_web_sm")
                self.enabled = False
        else:
            self.enabled = False

    def extract_entities(self, qa_pair: QAPair) -> Dict[str, List[str]]:
        """Extract entities from Q&A pair using spaCy NER only (no LLM calls)"""
        if not self.enabled:
            return {'persons': [], 'roles': [], 'orgs': []}

        combined_text = f"{qa_pair.question} {qa_pair.answer}"
        entities = {'persons': [], 'roles': [], 'orgs': []}

        # NER for PERSON and ORG using spaCy (fast, no LLM calls)
        doc = self.nlp(combined_text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities['persons'].append(ent.text)
            elif ent.label_ == "ORG":
                entities['orgs'].append(ent.text)

        # Note: Role extraction via LLM disabled to avoid token drain
        # spaCy NER will handle most entity types efficiently
        entities['roles'] = []

        return entities

    def _extract_roles_llm(self, text: str) -> List[str]:
        """Extract role/title entities using LLM"""
        prompt = f"""Extract job titles/roles from this text. Return ONLY a JSON list.

Text: "{text}"

JSON list (or [] if none):"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )

            result = response.choices[0].message.content.strip()
            import re
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                roles = json.loads(json_match.group())
                return [r for r in roles if isinstance(r, str)]
        except:
            pass

        return []


class QueryParserV2:
    """Parses user queries into structured QueryAnalysisV2 objects (V2)"""

    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client

    def parse(self, user_query: str) -> QueryAnalysisV2:
        """Parse query into intent and entities"""
        prompt = f"""Analyze this query. Return ONLY valid JSON:
{{
  "intent": "person_role_lookup" | "role_person_lookup" | "general_faq",
  "entities": {{"person": "", "role": ""}},
  "semantic_query": "rephrased query"
}}

Query: "{user_query}"

JSON:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )

            result = response.choices[0].message.content.strip()
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return QueryAnalysisV2(
                    intent=parsed.get('intent', 'general_faq'),
                    entities=parsed.get('entities', {}),
                    semantic_query=parsed.get('semantic_query', user_query)
                )
        except:
            pass

        return QueryAnalysisV2(intent='general_faq', entities={}, semantic_query=user_query)


class ParallelRetrieverV2:
    """
    Implements parallel-fused hybrid retrieval with RRF.
    Replaces sequential fallback pipeline.
    """

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.dataset = search_engine.dataset
        self.encoder = search_engine.encoder
        self.bm25 = search_engine.bm25

        # Store enriched metadata (populated by enrich_dataset)
        self.qa_metadata: Dict[str, Dict[str, List[str]]] = {}

        # Retriever weights for RRF fusion (Improvement #4)
        # Empirically tuned based on retriever performance characteristics
        self.retriever_weights = {
            'Filtered-Semantic': 1.5,  # Best for entity-specific queries
            'BM25': 1.0,                # Baseline keyword matching
            'Q-Semantic': 1.2,          # Good for paraphrased questions
            'A-Semantic': 0.8           # Weakest, often retrieves off-topic
        }

    def _compute_hash(self, text: str) -> str:
        """Compute MD5 hash of text for caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def enrich_dataset(self, enricher: MetadataEnricher, json_path: str = None):
        """Enrich all Q&A pairs with metadata (Incremental Update)"""
        if not enricher.enabled:
            logger.warning("Metadata enricher not available - V2 will use limited filtering")
            return

        # Determine cache file
        if json_path:
            cache_file = json_path.replace('.json', '_metadata_cache.pkl')
        else:
            cache_file = 'metadata_cache.pkl'

        # Load cache if exists
        cached_metadata = {}
        cached_hashes = {}
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    cached_metadata = cache_data.get('metadata', {})
                    cached_hashes = cache_data.get('hashes', {})
                logger.info(f"✅ Found existing metadata cache: {cache_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load metadata cache: {e}")

        logger.info("Enriching Q&A dataset with metadata...")
        
        current_hashes = {}
        new_metadata = {}
        
        items_to_process = []
        reused_count = 0
        
        # Identify what needs processing
        for qa_pair in self.dataset.all_qa_pairs:
            # Unique key for the QA pair
            key = f"{qa_pair.topic}_{qa_pair.topic_index}_{qa_pair.qa_index}"
            
            # Compute hash of content (question + answer + topic)
            content_str = f"{qa_pair.question}|{qa_pair.answer}|{qa_pair.topic}"
            content_hash = self._compute_hash(content_str)
            
            current_hashes[key] = content_hash
            
            # Check if we can reuse cached metadata
            if key in cached_hashes and cached_hashes[key] == content_hash and key in cached_metadata:
                new_metadata[key] = cached_metadata[key]
                reused_count += 1
            else:
                items_to_process.append((key, qa_pair))

        # Process new/changed items
        if items_to_process:
            logger.info(f"🔄 Incremental enrichment: Processing {len(items_to_process)} new/changed items...")
            
            # Process in batches to show progress
            total = len(items_to_process)
            for i, (key, qa_pair) in enumerate(items_to_process):
                if i % 100 == 0:
                    logger.info(f"   Processing {i}/{total}...")
                
                entities = enricher.extract_entities(qa_pair)
                new_metadata[key] = entities
        
        self.qa_metadata = new_metadata
        
        logger.info(f"✅ Enriched {len(self.qa_metadata)} Q&A pairs (Reused: {reused_count}, New: {len(items_to_process)})")
        
        # Save cache if changes made
        if items_to_process or len(self.qa_metadata) != len(cached_metadata):
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'metadata': self.qa_metadata,
                        'hashes': current_hashes
                    }, f)
                logger.info(f"✅ Cached metadata to {cache_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save metadata cache: {e}")

    def retrieve(self, query_analysis: QueryAnalysisV2, top_k: int = 10):
        """Execute parallel retrieval + RRF fusion"""
        logger.info("🔍 Running parallel retrieval (4 strategies)...")

        # Run 4 retrievers
        candidates_1 = self._retriever_filtered_semantic(query_analysis, 20)
        candidates_2 = self._retriever_bm25(query_analysis.semantic_query, 20)
        candidates_3 = self._retriever_question_semantic(query_analysis.semantic_query, 20)
        candidates_4 = self._retriever_answer_semantic(query_analysis.semantic_query, 20)

        # Store retriever details for UI with user-friendly names
        retriever_details = {
            'retriever_1': {
                **self._format_retriever_results(candidates_1[:3], "Smart Filter + Meaning Search"),
                'description': 'Filters by entities (people, roles, etc.) then searches by meaning',
                'technical_name': 'Filtered-Semantic'
            },
            'retriever_2': {
                **self._format_retriever_results(candidates_2[:3], "Exact Word Matching"),
                'description': 'Finds questions with exact keyword matches',
                'technical_name': 'BM25-Keyword'
            },
            'retriever_3': {
                **self._format_retriever_results(candidates_3[:3], "Question Similarity"),
                'description': 'Finds questions with similar meanings',
                'technical_name': 'Q-Semantic'
            },
            'retriever_4': {
                **self._format_retriever_results(candidates_4[:3], "Answer Content Search"),
                'description': 'Searches answer text for relevant information',
                'technical_name': 'A-Semantic'
            }
        }

        # RRF fusion
        fused, rrf_details = self._rrf_fusion([candidates_1, candidates_2, candidates_3, candidates_4], top_k)

        logger.info(f"✅ RRF fusion complete: {len(fused)} candidates")

        return fused, {
            'retrievers': retriever_details,
            'rrf_fusion': rrf_details
        }

    def _format_retriever_results(self, candidates: List[RetrievalCandidate], name: str) -> Dict:
        """Format retriever results for UI display"""
        return {
            'name': name,
            'results': [
                {
                    'rank': i + 1,
                    'score': float(cand.score),
                    'score_display': self._score_to_stars(cand.score),
                    'question': cand.qa_pair.question,
                    'topic': cand.qa_pair.topic
                }
                for i, cand in enumerate(candidates)
            ]
        }

    def _score_to_stars(self, score: float) -> str:
        """Convert score to star rating for better UX"""
        if score >= 0.8:
            return "⭐⭐⭐⭐⭐"
        elif score >= 0.6:
            return "⭐⭐⭐⭐"
        elif score >= 0.4:
            return "⭐⭐⭐"
        elif score >= 0.2:
            return "⭐⭐"
        else:
            return "⭐"

    def _rrf_score_to_label(self, score: float) -> str:
        """Convert RRF score to user-friendly label"""
        if score >= 0.06:
            return "Very High"
        elif score >= 0.05:
            return "High"
        elif score >= 0.04:
            return "Medium"
        elif score >= 0.03:
            return "Low"
        else:
            return "Very Low"

    def _vote_count_to_consensus(self, count: int) -> str:
        """Convert vote count to consensus label"""
        if count == 4:
            return "✓✓✓✓ Strong Consensus"
        elif count == 3:
            return "✓✓✓ Good Consensus"
        elif count == 2:
            return "✓✓ Moderate Agreement"
        else:
            return "✓ Weak Agreement"

    def _retriever_filtered_semantic(self, query: QueryAnalysisV2, top_k: int) -> List[RetrievalCandidate]:
        """Retriever 1: Metadata-filtered semantic"""
        filtered_pairs = self.dataset.all_qa_pairs

        # Apply metadata filters
        if query.intent == "person_role_lookup" and query.entities.get('person'):
            person = query.entities['person'].lower()
            filtered_pairs = [
                qa for qa in filtered_pairs
                if self._has_person(qa, person)
            ]
            logger.info(f"   R1: Filtered to {len(filtered_pairs)} with person='{person}'")

        elif query.intent == "role_person_lookup" and query.entities.get('role'):
            role = query.entities['role'].lower()
            filtered_pairs = [
                qa for qa in filtered_pairs
                if self._has_role(qa, role)
            ]
            logger.info(f"   R1: Filtered to {len(filtered_pairs)} with role='{role}'")

        if not filtered_pairs:
            return []

        # Semantic search
        query_emb = self.encoder.encode([query.semantic_query])[0]
        scores = []
        for qa in filtered_pairs:
            idx = self.dataset.all_qa_pairs.index(qa)
            sim = np.dot(self.search_engine.encoded_questions[idx], query_emb) / (
                np.linalg.norm(self.search_engine.encoded_questions[idx]) * np.linalg.norm(query_emb)
            )
            scores.append(sim)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [RetrievalCandidate(filtered_pairs[i], scores[i], "Filtered-Semantic") for i in top_indices]

        # Log top 3 results
        logger.info(f"   📋 Retriever 1 (Filtered-Semantic) - Top 3:")
        for rank, cand in enumerate(results[:3], 1):
            logger.info(f"      #{rank} [Score: {cand.score:.4f}] {cand.qa_pair.question[:80]}...")

        return results

    def _has_person(self, qa: QAPair, person_name: str) -> bool:
        """Check if QA pair has person entity"""
        key = f"{qa.topic}_{qa.topic_index}_{qa.qa_index}"
        if key in self.qa_metadata:
            return any(person_name in p.lower() for p in self.qa_metadata[key]['persons'])
        # Fallback: check in text
        combined = f"{qa.question} {qa.answer}".lower()
        return person_name in combined

    def _has_role(self, qa: QAPair, role_name: str) -> bool:
        """Check if QA pair has role entity"""
        key = f"{qa.topic}_{qa.topic_index}_{qa.qa_index}"
        if key in self.qa_metadata:
            return any(role_name in r.lower() for r in self.qa_metadata[key]['roles'])
        # Fallback: check in text
        combined = f"{qa.question} {qa.answer}".lower()
        return role_name in combined

    def _retriever_bm25(self, query: str, top_k: int) -> List[RetrievalCandidate]:
        """Retriever 2: BM25 keyword"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [RetrievalCandidate(self.dataset.all_qa_pairs[i], scores[i], "BM25") for i in top_indices]

        # Log top 3 results
        logger.info(f"   📋 Retriever 2 (BM25-Keyword) - Top 3:")
        for rank, cand in enumerate(results[:3], 1):
            logger.info(f"      #{rank} [Score: {cand.score:.4f}] {cand.qa_pair.question[:80]}...")

        return results

    def _retriever_question_semantic(self, query: str, top_k: int) -> List[RetrievalCandidate]:
        """Retriever 3: Semantic on questions"""
        query_emb = self.encoder.encode([query])[0]
        scores = np.dot(self.search_engine.encoded_questions, query_emb) / (
            np.linalg.norm(self.search_engine.encoded_questions, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [RetrievalCandidate(self.dataset.all_qa_pairs[i], scores[i], "Q-Semantic") for i in top_indices]

        # Log top 3 results
        logger.info(f"   📋 Retriever 3 (Q-Semantic) - Top 3:")
        for rank, cand in enumerate(results[:3], 1):
            logger.info(f"      #{rank} [Score: {cand.score:.4f}] {cand.qa_pair.question[:80]}...")

        return results

    def _retriever_answer_semantic(self, query: str, top_k: int) -> List[RetrievalCandidate]:
        """Retriever 4: Semantic on answers"""
        query_emb = self.encoder.encode([query])[0]
        scores = np.dot(self.search_engine.encoded_answers, query_emb) / (
            np.linalg.norm(self.search_engine.encoded_answers, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [RetrievalCandidate(self.dataset.all_qa_pairs[i], scores[i], "A-Semantic") for i in top_indices]

        # Log top 3 results
        logger.info(f"   📋 Retriever 4 (A-Semantic) - Top 3:")
        for rank, cand in enumerate(results[:3], 1):
            logger.info(f"      #{rank} [Score: {cand.score:.4f}] {cand.qa_pair.question[:80]}...")

        return results

    def _rrf_fusion(self, retriever_results: List[List[RetrievalCandidate]], top_k: int, k: int = 60):
        """Reciprocal Rank Fusion with weighted retrievers (Improvement #4)"""
        rrf_scores = {}

        logger.info(f"\n   🔄 RRF Fusion Process (k={k}, weighted):")

        for retriever_idx, candidates in enumerate(retriever_results, 1):
            for rank, cand in enumerate(candidates, 1):
                qa_key = f"{cand.qa_pair.topic}_{cand.qa_pair.topic_index}_{cand.qa_pair.qa_index}"

                if qa_key not in rrf_scores:
                    rrf_scores[qa_key] = {
                        'score': 0.0,
                        'qa_pair': cand.qa_pair,
                        'sources': [],
                        'calculations': []
                    }

                # Apply retriever-specific weight (Improvement #4)
                weight = self.retriever_weights.get(cand.retriever_name, 1.0)
                base_score = 1.0 / (k + rank)
                score_contribution = weight * base_score

                rrf_scores[qa_key]['score'] += score_contribution
                rrf_scores[qa_key]['sources'].append(f"{cand.retriever_name}#{rank}")
                rrf_scores[qa_key]['calculations'].append(
                    f"{cand.retriever_name}@{rank}={score_contribution:.4f}(w={weight})"
                )

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        # Log top 3 RRF calculations
        logger.info(f"\n   🏆 RRF Top 3 Calculations:")
        for idx, (qa_key, data) in enumerate(sorted_items[:3], 1):
            calc_str = " + ".join(data['calculations'])
            logger.info(f"      #{idx} [Total: {data['score']:.4f}] = {calc_str}")
            logger.info(f"          Question: {data['qa_pair'].question[:80]}...")
            logger.info(f"          Voted by: {', '.join(data['sources'])}")

        # Prepare RRF details for UI
        rrf_details = [
            {
                'rank': idx,
                'total_score': float(data['score']),
                'score_display': self._rrf_score_to_label(data['score']),
                'question': data['qa_pair'].question,
                'topic': data['qa_pair'].topic,
                'calculation': " + ".join(data['calculations']),
                'calculation_simple': f"{len(data['sources'])} out of 4 methods agreed",
                'voted_by': data['sources'],
                'vote_count': len(data['sources']),
                'consensus': self._vote_count_to_consensus(len(data['sources']))
            }
            for idx, (_, data) in enumerate(sorted_items[:3], 1)
        ]

        candidates = [
            RetrievalCandidate(
                data['qa_pair'],
                data['score'],
                f"RRF({', '.join(data['sources'][:2])})"
            )
            for _, data in sorted_items[:top_k]
        ]

        return candidates, rrf_details


class LLMVerifierV2:
    """LLM-as-a-Verifier using NLI for factual validation"""

    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client

    def verify(self, query: str, candidates: List[RetrievalCandidate], top_k: int = 3) -> List[RetrievalCandidate]:
        """Verify candidates using NLI"""
        logger.info(f"🔍 Verifying top {len(candidates[:5])} candidates...")

        verified = []
        for cand in candidates[:5]:
            label = self._verify_single(query, cand)

            if label == "Entailment":
                verified.append(cand)
                logger.info(f"   ✅ VERIFIED: {cand.qa_pair.question[:60]}...")
                if len(verified) >= top_k:
                    break
            else:
                logger.info(f"   ❌ REJECTED ({label})")

        return verified

    def _verify_single(self, query: str, cand: RetrievalCandidate) -> str:
        """Verify single candidate with NLI (Improvement #1: Strengthened prompt)"""
        prompt = f"""You are a strict factual verifier for a Q&A system. Determine if the Document DIRECTLY and COMPLETELY answers the Query.

Query: "{query}"
Document: "{cand.qa_pair.answer}"

CLASSIFICATION RULES:
1. "Entailment" - Document contains the EXACT answer to the Query
   Example: Q: "Who handles admissions?" D: "Contact Enrollment at (800)..." → Entailment

2. "Neutral" - Document is related but doesn't directly answer the Query
   Example: Q: "Who handles admissions?" D: "Info about transfer admissions at..." → Neutral

3. "Contradiction" - Document contradicts or conflicts with the Query

STRICT CRITERIA:
- If Document mentions a related topic but doesn't answer the specific question → Neutral
- If Document answers a different question about the same topic → Neutral
- Only label "Entailment" if you're 100% confident the Document answers the Query

Return ONLY one word: Entailment, Neutral, or Contradiction

Label:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20
            )

            result = response.choices[0].message.content.strip()
            if "Entailment" in result:
                return "Entailment"
            elif "Contradiction" in result:
                return "Contradiction"
            else:
                return "Neutral"
        except:
            return "Neutral"


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the engine
    print("\n" + "="*60)
    print("TESTING JSON CHATBOT ENGINE")
    print("="*60)

    # Initialize engine
    engine = JSONChatbotEngine('CSU_Progress.json')

    # Test questions
    test_questions = [
        "What is the Healthy Indiana Plan?",
        "How do I find a doctor?",
        "What is One Plan?",
        "Tell me about cost sharing",
        "What is quantum physics?"  # Should fallback
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print("="*60)

        result = engine.process_question(question)

        print(f"\nMatched by: {result['matched_by']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nAnswer:\n{result['answer']}")
        print("="*60)

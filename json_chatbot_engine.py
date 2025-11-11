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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

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

        for topic_idx, topic_data in enumerate(data):
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

        # Thresholds (adjusted for better accuracy)
        self.SIMILARITY_THRESHOLD_IDEAL = 0.7
        self.SIMILARITY_THRESHOLD = 0.50  # Raised from 0.45 to force weak matches to answer-based search

        # Initialize sentence transformer model
        logger.info("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Encoded embeddings
        self.encoded_questions = None
        self.encoded_answers = None
        self.encoded_topics = None

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

        logger.info("✅ Similarity search engine ready")

    def _encode_all(self):
        """Encode all questions, answers, and topics"""
        questions = self.dataset.get_all_questions()
        answers = self.dataset.get_all_answers()
        topics = self.dataset.get_topic_names()

        logger.info(f"Encoding {len(questions)} questions...")
        self.encoded_questions = self.encoder.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Encoding {len(answers)} answers...")
        self.encoded_answers = self.encoder.encode(
            answers,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Encoding {len(topics)} topics...")
        self.encoded_topics = self.encoder.encode(
            topics,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def _save_cache(self, cache_file: str):
        """Save encoded data to cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'encoded_questions': self.encoded_questions,
                    'encoded_answers': self.encoded_answers,
                    'encoded_topics': self.encoded_topics
                }, f)
            logger.info(f"✅ Cached encodings to {cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to cache encodings: {e}")

    def _load_cache(self, cache_file: str):
        """Load encoded data from cache"""
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            self.encoded_questions = cached['encoded_questions']
            self.encoded_answers = cached['encoded_answers']
            self.encoded_topics = cached['encoded_topics']
            logger.info("✅ Loaded cached encodings")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}, re-encoding...")
            self._encode_all()

    def search(self, user_question: str, search_answers: bool = False) -> Dict:
        """
        Search for most similar Q&A pair

        Args:
            user_question: User's input question
            search_answers: If True, also search in answers

        Returns:
            Dict with:
            - best_match: QAPair object
            - score: similarity score (0-1)
            - matched_by: 'question' or 'answer'
            - duration: search time in seconds
            - meets_ideal_threshold: bool
            - meets_minimal_threshold: bool
        """
        start_time = time.time()

        # Encode user question
        user_embedding = self.encoder.encode([user_question])[0]

        # Calculate similarities with questions
        question_similarities = np.dot(self.encoded_questions, user_embedding) / (
            np.linalg.norm(self.encoded_questions, axis=1) * np.linalg.norm(user_embedding)
        )

        # Get best question match
        best_question_idx = np.argmax(question_similarities)
        best_question_score = question_similarities[best_question_idx]

        best_idx = best_question_idx
        best_score = best_question_score
        matched_by = 'question'

        # Optionally search answers too
        if search_answers:
            answer_similarities = np.dot(self.encoded_answers, user_embedding) / (
                np.linalg.norm(self.encoded_answers, axis=1) * np.linalg.norm(user_embedding)
            )
            best_answer_idx = np.argmax(answer_similarities)
            best_answer_score = answer_similarities[best_answer_idx]

            if best_answer_score > best_question_score:
                best_idx = best_answer_idx
                best_score = best_answer_score
                matched_by = 'answer'

        duration = time.time() - start_time

        # Get QAPair object
        best_match = self.dataset.all_qa_pairs[best_idx]

        # Log results
        logger.info(f"Similarity search: score={best_score:.4f}, matched_by={matched_by}, duration={duration:.3f}s")

        return {
            'best_match': best_match,
            'score': float(best_score),
            'matched_by': matched_by,
            'duration': duration,
            'meets_ideal_threshold': best_score >= self.SIMILARITY_THRESHOLD_IDEAL,
            'meets_minimal_threshold': best_score >= self.SIMILARITY_THRESHOLD
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
        if best_score >= 0.5:
            logger.info(f"Topic match: '{self.dataset.topics[best_idx].topic}' (score={best_score:.4f})")
            return best_idx, float(best_score)

        return None


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

    def process_question(self, user_question: str, session_id: str = "default") -> Dict:
        """
        Process user question through multi-stage pipeline

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
        logger.info(f"PROCESSING QUESTION: '{user_question}'")
        logger.info("="*60)

        pipeline_info = {
            'session_id': session_id,
            'original_question': user_question,
            'timestamp': datetime.now().isoformat(),
            'stages': []
        }

        # ====================================================================
        # STAGE 1: Primary Similarity Search (IDEAL threshold)
        # ====================================================================
        logger.info("\n🔍 STAGE 1: Primary Similarity Search")

        search_result = self.search_engine.search(user_question)

        stage_1_info = {
            'stage': 'primary_similarity',
            'score': search_result['score'],
            'matched_by': search_result['matched_by'],
            'duration': search_result['duration']
        }
        pipeline_info['stages'].append(stage_1_info)

        if search_result['meets_ideal_threshold']:
            logger.info(f"✅ IDEAL MATCH FOUND (score: {search_result['score']:.4f})")
            logger.info(f"   Question: {search_result['best_match'].question}")
            logger.info(f"   Topic: {search_result['best_match'].topic}")
            logger.info("="*60)

            return {
                'answer': search_result['best_match'].answer,
                'source_qa': search_result['best_match'],
                'matched_by': 'similarity_ideal',
                'confidence': search_result['score'],
                'pipeline_info': pipeline_info
            }

        logger.info(f"⚠️ Score {search_result['score']:.4f} below IDEAL threshold {self.search_engine.SIMILARITY_THRESHOLD_IDEAL}")
        logger.info(f"   Best match found: \"{search_result['best_match'].question}\"")
        logger.info(f"   From topic: {search_result['best_match'].topic}")

        # ====================================================================
        # STAGE 2: Rephrase + Retry (MINIMAL threshold)
        # ====================================================================
        if self.rephraser:
            logger.info("\n🔄 STAGE 2: Rephrase + Retry")

            # Get example questions for better rephrasing
            example_questions = self.dataset.get_all_questions()[:20]
            rephrased_text, rephrase_info = self.rephraser.rephrase(
                user_question,
                example_questions=example_questions
            )

            stage_2_info = {
                'stage': 'rephrase',
                'rephrased_text': rephrased_text,
                'duration': rephrase_info.get('duration', 0)
            }
            pipeline_info['stages'].append(stage_2_info)

            if rephrased_text and rephrased_text != user_question:
                logger.info(f"   Rephrased: '{rephrased_text}'")

                # Search with rephrased text
                rephrase_search = self.search_engine.search(rephrased_text)

                stage_2_search_info = {
                    'stage': 'rephrase_similarity',
                    'score': rephrase_search['score'],
                    'matched_by': rephrase_search['matched_by'],
                    'duration': rephrase_search['duration']
                }
                pipeline_info['stages'].append(stage_2_search_info)

                # Use better score
                if rephrase_search['score'] > search_result['score']:
                    search_result = rephrase_search

                    if search_result['meets_minimal_threshold']:
                        logger.info(f"✅ REPHRASE MATCH FOUND (score: {search_result['score']:.4f})")
                        logger.info(f"   Question: {search_result['best_match'].question}")
                        logger.info(f"   Topic: {search_result['best_match'].topic}")
                        logger.info("="*60)

                        return {
                            'answer': search_result['best_match'].answer,
                            'source_qa': search_result['best_match'],
                            'matched_by': 'rephrase_similarity',
                            'confidence': search_result['score'],
                            'pipeline_info': pipeline_info
                        }
                    else:
                        logger.info(f"   ⚠️ Rephrase improved score to {rephrase_search['score']:.4f} but still below threshold")
                        logger.info(f"   Best match: \"{rephrase_search['best_match'].question}\"")
                        logger.info(f"   From topic: {rephrase_search['best_match'].topic}")
            else:
                logger.info("   ⚠️ Rephrasing failed or produced same text")

        # Check if original meets minimal threshold
        if search_result['meets_minimal_threshold']:
            logger.info(f"✅ MINIMAL MATCH FOUND (score: {search_result['score']:.4f})")
            logger.info(f"   Question: {search_result['best_match'].question}")
            logger.info(f"   Topic: {search_result['best_match'].topic}")
            logger.info("="*60)

            return {
                'answer': search_result['best_match'].answer,
                'source_qa': search_result['best_match'],
                'matched_by': 'similarity_minimal',
                'confidence': search_result['score'],
                'pipeline_info': pipeline_info
            }

        logger.info(f"⚠️ Score {search_result['score']:.4f} below MINIMAL threshold {self.search_engine.SIMILARITY_THRESHOLD}")
        logger.info(f"   Best match so far: \"{search_result['best_match'].question}\"")
        logger.info(f"   From topic: {search_result['best_match'].topic}")

        # ====================================================================
        # STAGE 2.5: Search in Answers (Extended Search)
        # ====================================================================
        logger.info("\n🔎 STAGE 2.5: Answer-Based Search")
        logger.info("   Searching in answer content instead of questions...")

        answer_search_result = self.search_engine.search(user_question, search_answers=True)

        stage_2_5_info = {
            'stage': 'answer_search',
            'score': answer_search_result['score'],
            'matched_by': answer_search_result['matched_by'],
            'duration': answer_search_result['duration']
        }
        pipeline_info['stages'].append(stage_2_5_info)

        if answer_search_result['meets_minimal_threshold']:
            logger.info(f"✅ ANSWER MATCH FOUND (score: {answer_search_result['score']:.4f})")
            logger.info(f"   Matched by: {answer_search_result['matched_by']}")
            logger.info(f"   Question: {answer_search_result['best_match'].question}")
            logger.info(f"   Topic: {answer_search_result['best_match'].topic}")
            logger.info("="*60)

            return {
                'answer': answer_search_result['best_match'].answer,
                'source_qa': answer_search_result['best_match'],
                'matched_by': 'answer_similarity',
                'confidence': answer_search_result['score'],
                'pipeline_info': pipeline_info
            }

        logger.info(f"⚠️ Answer search score {answer_search_result['score']:.4f} also below threshold")
        logger.info(f"   Best {answer_search_result['matched_by']} match: \"{answer_search_result['best_match'].question}\"")
        logger.info(f"   From topic: {answer_search_result['best_match'].topic}")
        if answer_search_result['matched_by'] == 'answer':
            logger.info(f"   Answer preview: {answer_search_result['best_match'].answer[:100]}...")

        # ====================================================================
        # STAGE 3: Topic-Level Search
        # ====================================================================
        logger.info("\n📚 STAGE 3: Topic-Level Search")

        topic_result = self.search_engine.search_topic(user_question)

        if topic_result:
            topic_idx, topic_score = topic_result
            topic = self.dataset.topics[topic_idx]

            stage_3_info = {
                'stage': 'topic_search',
                'topic_name': topic.topic,
                'score': topic_score
            }
            pipeline_info['stages'].append(stage_3_info)

            logger.info(f"✅ TOPIC MATCH FOUND: '{topic.topic}' (score: {topic_score:.4f})")
            logger.info("="*60)

            # Build topic overview response
            answer = f"I found information about '{topic.topic}'.\n\n"
            answer += f"Here are some questions I can answer about this topic:\n\n"

            for i, qa in enumerate(topic.qa_pairs[:5], 1):
                answer += f"{i}. {qa.question}\n"

            if len(topic.qa_pairs) > 5:
                answer += f"\n...and {len(topic.qa_pairs) - 5} more questions."

            return {
                'answer': answer,
                'source_qa': None,
                'matched_by': 'topic',
                'confidence': topic_score,
                'pipeline_info': pipeline_info,
                'suggested_questions': [qa.question for qa in topic.qa_pairs[:5]]
            }

        logger.info("⚠️ No topic match found (highest topic similarity below 0.5 threshold)")

        # ====================================================================
        # STAGE 4: Fallback Response
        # ====================================================================
        logger.info("\n❌ STAGE 4: Fallback Response")

        stage_4_info = {'stage': 'fallback'}
        pipeline_info['stages'].append(stage_4_info)

        logger.info("No match found in any stage, returning fallback")
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

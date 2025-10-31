"""
LangGraph Chatbot Backend
=========================

Multi-agent LangGraph system for:
1. Simple chatbot responses
2. Q&A modification (Supervisor + Simplifier agents)

Architecture:
- USER INPUT → LangGraph Agent → LLM (Groq) → RESPONSE
- USER REQUEST → Supervisor → Simplifier Agent → Modified JSON

Features:
- State management with TypedDict
- Multiple agent nodes
- Groq AI integration (Llama 4)
- Async execution
- Q&A simplification
"""

import os
import logging
import aiosqlite
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Optional
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS - Class-based structure for JSON data navigation
# ============================================================================

@dataclass
class QAPair:
    """
    Represents a single Question-Answer pair

    Attributes:
        question: The question text
        answer: The answer text
        metadata: Optional metadata (is_unified, is_bucketed, bucket_id, etc.)
    """
    question: str
    answer: str
    metadata: Dict = None

    def __post_init__(self):
        """Initialize metadata as empty dict if None"""
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self) -> str:
        return f"QAPair(Q: {self.question[:50]}...)"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "question": self.question,
            "answer": self.answer
        }
        # Add all metadata fields to the result
        if self.metadata:
            result.update(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'QAPair':
        """Create QAPair from dictionary"""
        # Extract question and answer
        question = data['question']
        answer = data['answer']

        # All other fields go into metadata
        metadata = {k: v for k, v in data.items() if k not in ['question', 'answer']}

        return cls(
            question=question,
            answer=answer,
            metadata=metadata if metadata else {}
        )


@dataclass
class Topic:
    """
    Represents a topic with metadata and Q&A pairs

    Attributes:
        topic: Topic name/title
        semantic_path: Semantic URL path
        original_url: Original source URL
        browser_content: Raw browser content
        extraction_method: Method used for extraction
        processing_time: Time taken to process (seconds)
        status: Processing status
        error: Error message if any
        qa_pairs: List of QAPair objects
        qa_generation_status: Status of Q&A generation
        qa_generation_time: Time taken for Q&A generation
        qa_model: Model used for Q&A generation
        qa_count: Number of Q&A pairs
    """
    topic: str
    semantic_path: str
    original_url: str
    browser_content: str
    extraction_method: str
    processing_time: float
    status: str
    error: str
    qa_pairs: List[QAPair] = field(default_factory=list)
    qa_generation_status: str = ""
    qa_generation_time: float = 0.0
    qa_model: str = ""
    qa_count: int = 0

    def __repr__(self) -> str:
        return f"Topic('{self.topic}', {len(self.qa_pairs)} Q&A pairs)"

    def get_qa_pair(self, index: int) -> QAPair:
        """Get Q&A pair by index"""
        if 0 <= index < len(self.qa_pairs):
            return self.qa_pairs[index]
        raise IndexError(f"Q&A index {index} out of range (0-{len(self.qa_pairs)-1})")

    def add_qa_pair(self, qa: QAPair) -> None:
        """Add a new Q&A pair"""
        self.qa_pairs.append(qa)
        self.qa_count = len(self.qa_pairs)

    def remove_qa_pair(self, index: int) -> QAPair:
        """Remove and return Q&A pair at index"""
        if 0 <= index < len(self.qa_pairs):
            qa = self.qa_pairs.pop(index)
            self.qa_count = len(self.qa_pairs)
            return qa
        raise IndexError(f"Q&A index {index} out of range")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "topic": self.topic,
            "semantic_path": self.semantic_path,
            "original_url": self.original_url,
            "browser_content": self.browser_content,
            "extraction_method": self.extraction_method,
            "processing_time": self.processing_time,
            "status": self.status,
            "error": self.error,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "qa_generation_status": self.qa_generation_status,
            "qa_generation_time": self.qa_generation_time,
            "qa_model": self.qa_model,
            "qa_count": self.qa_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Topic':
        """Create Topic from dictionary"""
        qa_pairs = [QAPair.from_dict(qa) for qa in data.get('qa_pairs', [])]

        return cls(
            topic=data.get('topic', ''),
            semantic_path=data.get('semantic_path', ''),
            original_url=data.get('original_url', ''),
            browser_content=data.get('browser_content', ''),
            extraction_method=data.get('extraction_method', ''),
            processing_time=data.get('processing_time', 0.0),
            status=data.get('status', ''),
            error=data.get('error', ''),
            qa_pairs=qa_pairs,
            qa_generation_status=data.get('qa_generation_status', ''),
            qa_generation_time=data.get('qa_generation_time', 0.0),
            qa_model=data.get('qa_model', ''),
            qa_count=data.get('qa_count', len(qa_pairs))
        )


class QADataset:
    """
    Main container for Q&A dataset - manages all topics

    Provides easy navigation, search, and modification of Q&A data.

    Usage:
        dataset = QADataset.from_json('data.json')
        topic = dataset.get_topic(0)
        qa = topic.get_qa_pair(0)
        dataset.save_to_json('data.json')
    """

    def __init__(self, topics: List[Topic] = None):
        """Initialize dataset with list of Topic objects"""
        self.topics = topics or []

    def __repr__(self) -> str:
        total_qa = sum(len(t.qa_pairs) for t in self.topics)
        return f"QADataset({len(self.topics)} topics, {total_qa} total Q&A pairs)"

    def __len__(self) -> int:
        return len(self.topics)

    def __getitem__(self, index: int) -> Topic:
        """Access topic by index: dataset[0]"""
        return self.topics[index]

    def get_topic(self, index: int) -> Topic:
        """Get topic by index (0-based)"""
        if 0 <= index < len(self.topics):
            return self.topics[index]
        raise IndexError(f"Topic index {index} out of range (0-{len(self.topics)-1})")

    def get_qa_pair(self, topic_index: int, qa_index: int) -> QAPair:
        """Get specific Q&A pair"""
        topic = self.get_topic(topic_index)
        return topic.get_qa_pair(qa_index)

    def find_topic_by_name(self, name: str) -> Optional[Topic]:
        """Find topic by exact name match"""
        for topic in self.topics:
            if topic.topic == name:
                return topic
        return None

    def search_questions(self, keyword: str) -> List[Dict]:
        """Search all questions containing keyword"""
        results = []
        keyword_lower = keyword.lower()

        for topic_idx, topic in enumerate(self.topics):
            for qa_idx, qa in enumerate(topic.qa_pairs):
                if keyword_lower in qa.question.lower():
                    results.append({
                        'topic_index': topic_idx,
                        'qa_index': qa_idx,
                        'topic_name': topic.topic,
                        'qa_pair': qa
                    })
        return results

    def total_topics(self) -> int:
        """Total number of topics"""
        return len(self.topics)

    def total_qa_pairs(self) -> int:
        """Total number of Q&A pairs across all topics"""
        return sum(len(t.qa_pairs) for t in self.topics)

    def to_dict(self) -> List[Dict]:
        """Convert entire dataset to list of dicts (JSON format)"""
        return [topic.to_dict() for topic in self.topics]

    def save_to_json(self, file_path: str, indent: int = 2) -> None:
        """Save dataset to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
        logger.info(f"✅ Saved {len(self.topics)} topics to {file_path}")

    @classmethod
    def from_dict(cls, data: List[Dict]) -> 'QADataset':
        """Create QADataset from list of dicts"""
        topics = [Topic.from_dict(topic_data) for topic_data in data]
        return cls(topics)

    @classmethod
    def from_json(cls, file_path: str) -> 'QADataset':
        """Load dataset from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# SIMPLIFY AGENT - Simplifies Q&A pairs using LLM
# ============================================================================

class SimplifyAgent:
    """
    Agent that keeps original Q&A pairs and adds 10 new questions with ONE unified answer

    Usage:
        qa_pairs = [{"question": "...", "answer": "..."}]
        enhanced = await SimplifyAgent.simplify(qa_pairs)
        # Result: Original Q&A pairs + 10 new questions with same comprehensive answer
    """

    @staticmethod
    async def simplify(qa_pairs: List[Dict]) -> List[Dict]:
        """
        Keep original Q&A pairs and append 10 new questions with ONE unified answer

        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys

        Returns:
            Original Q&A pairs + 10 new Q&A pairs (all with same unified answer)
        """
        try:
            num_original = len(qa_pairs)
            logger.info(f"🔄 SimplifyAgent: Keeping {num_original} original Q&A pairs and generating 10 new questions with unified answer")

            # Step 1: Format Q&A pairs for LLM analysis
            formatted_qa = SimplifyAgent._format_qa_for_llm(qa_pairs)

            # Step 2: Create prompt to generate 10 questions + 1 unified answer
            prompt = SimplifyAgent._create_unified_prompt(formatted_qa, num_original)

            # Step 3: Send to LLM
            llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.3,
                api_key=os.getenv("GROQ_API_KEY")
            )

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            llm_output = response.content
            logger.info(f"✓ LLM response received ({len(llm_output)} chars)")
            logger.info(f"📝 LLM Response Preview:\n{llm_output[:500]}...")

            # Step 4: Parse LLM response to get 10 questions and unified answer
            new_questions, unified_answer = SimplifyAgent._parse_unified_response(llm_output)

            # Step 5: Create result - original pairs + 10 new pairs with same answer
            result_pairs = qa_pairs.copy()  # Keep all originals intact

            for question in new_questions:
                result_pairs.append({
                    "question": question,
                    "answer": unified_answer,  # Same answer for all 10 questions
                    "is_unified": True  # Mark as unified Q&A from SimplifyAgent
                })

            logger.info(f"✅ SimplifyAgent: Successfully created {len(result_pairs)} Q&A pairs")
            logger.info(f"   - Original pairs kept: {num_original}")
            logger.info(f"   - New questions added: {len(new_questions)}")
            logger.info(f"   - Unified answer length: {len(unified_answer)} chars (~{len(unified_answer.split())} words)")

            return result_pairs

        except Exception as e:
            logger.error(f"❌ SimplifyAgent error: {e}")
            # Return original pairs if generation fails
            return qa_pairs

    @staticmethod
    async def dynamic_adjust(qa_pairs: List[Dict]) -> List[Dict]:
        """
        Apply dynamic adjustment - intelligently group similar Q&A pairs into optimized buckets

        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys

        Returns:
            List of bucketed Q&A pairs with bucket_id and is_bucketed flags
        """
        try:
            num_original = len(qa_pairs)
            logger.info(f"🎯 Dynamic Adjustment: Analyzing {num_original} Q&A pairs for intelligent grouping")

            # Step 1: Format Q&A pairs for LLM analysis
            formatted_qa = SimplifyAgent._format_qa_for_llm(qa_pairs)

            # Step 2: Create prompt for bucketing
            prompt = SimplifyAgent._create_unified_varied_prompt(formatted_qa, num_original)

            # Step 3: Send to LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                api_key=os.getenv("GROQ_API_KEY")
            )

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            llm_output = response.content
            logger.info(f"✓ LLM response received ({len(llm_output)} chars)")
            logger.info(f"📝 LLM Response Preview:\n{llm_output[:500]}...")

            # Step 4: Parse LLM response to get bucketed Q&A pairs
            bucketed_pairs = SimplifyAgent._parse_bucketed_response(llm_output)

            if not bucketed_pairs:
                logger.warning("⚠️ No bucketed pairs returned, keeping original Q&A pairs")
                return qa_pairs

            # Count unique buckets
            unique_buckets = len(set(pair["bucket_id"] for pair in bucketed_pairs))

            logger.info(f"✅ Dynamic Adjustment: Successfully created {len(bucketed_pairs)} Q&A pairs")
            logger.info(f"   - Original pairs analyzed: {num_original}")
            logger.info(f"   - Buckets created: {unique_buckets}")
            logger.info(f"   - Avg questions per bucket: {len(bucketed_pairs) / unique_buckets:.1f}")

            return bucketed_pairs

        except Exception as e:
            logger.error(f"❌ Dynamic Adjustment error: {e}")
            # Return original pairs if generation fails
            return qa_pairs

    @staticmethod
    def _format_qa_for_llm(qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs into readable text for LLM"""
        formatted = ""
        for i, qa in enumerate(qa_pairs, 1):
            formatted += f"\n{i}. Q: {qa['question']}\n"
            formatted += f"   A: {qa['answer']}\n"
        return formatted

    @staticmethod
    def _create_unified_varied_prompt(formatted_qa: str, num_original: int) -> str:
        """Create the prompt for LLM to generate n number of answers and each answer has k number of questions attached"""
        return f"""You are an expert at analyzing Q&A pairs and creating smart groupings where similar questions share optimized answers.

Your task:
1. Read ALL {num_original} Q&A pairs below carefully
2. Understand the style, tone, and format of the existing answers
3. Identify patterns and themes - which questions can naturally share the same answer?
4. Create BUCKETS - group similar questions together
5. Generate ONE comprehensive answer per bucket (max 120 words each)
6. Each original question must be assigned to exactly ONE bucket

Original Q&A pairs for reference:
{formatted_qa}

Requirements:
- Generate the optimal number of buckets (typically 3-7 buckets, but use your judgment)
- Each bucket should have 2 or more related questions from the original list
- Group questions that can be answered comprehensively by a single answer
- Each answer should:
  * Be written in the SAME style/tone as the original answers
  * Be maximum 120 words
  * Be clear, concise, and comprehensive
  * Address ALL questions in that bucket naturally
- Use the EXACT original questions (do not rephrase them)
- Ensure every original question appears in exactly one bucket

Respond in this EXACT format:

BUCKET_1:
Q: [exact original question text]
Q: [exact original question text]
Q: [exact original question text]
ANSWER: [comprehensive answer for these questions, max 120 words]

BUCKET_2:
Q: [exact original question text]
Q: [exact original question text]
ANSWER: [comprehensive answer for these questions, max 120 words]

BUCKET_3:
Q: [exact original question text]
Q: [exact original question text]
Q: [exact original question text]
Q: [exact original question text]
ANSWER: [comprehensive answer for these questions, max 120 words]

[Continue with more buckets as needed...]

IMPORTANT: Use the exact format above. Each bucket must have at least 2 questions and one ANSWER line."""

    @staticmethod
    def _create_unified_prompt(formatted_qa: str, num_original: int) -> str:
        """Create the prompt for LLM to generate 10 questions + 1 unified answer"""
        return f"""You are an expert at analyzing Q&A pairs and creating comprehensive questions with unified answers.

Your task:
1. Read ALL {num_original} Q&A pairs below carefully
2. Understand the style, tone, and format of the existing answers
3. Generate 10 NEW questions based on the content (variations, related questions, etc.)
4. Create ONE comprehensive answer (max 120 words) that addresses ALL 10 new questions

Original Q&A pairs for reference:
{formatted_qa}

Requirements:
- Generate exactly 10 new questions that are relevant to the content above
- The 10 questions should be variations or related questions covering the main themes
- Create ONE unified answer that addresses ALL 10 questions
- The unified answer should:
  * Be written in the SAME style/tone as the original answers
  * Be maximum 120 words
  * Be clear, concise, and comprehensive
  * Address all 10 questions naturally

Respond in this EXACT format:
Q1: [question 1]
Q2: [question 2]
Q3: [question 3]
Q4: [question 4]
Q5: [question 5]
Q6: [question 6]
Q7: [question 7]
Q8: [question 8]
Q9: [question 9]
Q10: [question 10]
UNIFIED_ANSWER: [your comprehensive answer here, max 120 words]"""

    @staticmethod
    def _parse_unified_response(llm_output: str) -> tuple:
        """
        Parse LLM response to extract 10 questions and unified answer

        Returns:
            tuple: (list of 10 questions, unified answer string)
        """
        try:
            questions = []
            unified_answer = ""

            lines = llm_output.strip().split('\n')
            collecting_answer = False
            answer_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for question patterns (Q1:, Q2:, etc.)
                if line.startswith('Q') and ':' in line:
                    # Extract question text after "Q#:"
                    question_text = line.split(':', 1)[1].strip()
                    questions.append(question_text)

                # Check for unified answer marker
                elif 'UNIFIED_ANSWER:' in line:
                    collecting_answer = True
                    # Get text after marker if present on same line
                    answer_text = line.split('UNIFIED_ANSWER:', 1)[1].strip()
                    if answer_text:
                        answer_lines.append(answer_text)

                # Collect answer lines
                elif collecting_answer:
                    answer_lines.append(line)

            # Join answer lines
            unified_answer = " ".join(answer_lines).strip()

            # Validate we got 10 questions
            if len(questions) < 10:
                logger.warning(f"⚠️ Only got {len(questions)} questions, expected 10. Padding with generic questions.")
                # Pad with generic questions if needed
                while len(questions) < 10:
                    questions.append(f"Additional question {len(questions) + 1}")

            # Take only first 10 questions
            questions = questions[:10]

            # Validate unified answer
            if not unified_answer:
                logger.warning(f"⚠️ No unified answer found, using fallback")
                unified_answer = "Information not available. Please refer to the original Q&A pairs for detailed answers."

            word_count = len(unified_answer.split())
            logger.info(f"✓ Parsed {len(questions)} questions and unified answer ({word_count} words)")

            return questions, unified_answer

        except Exception as e:
            logger.error(f"❌ Error parsing unified response: {e}")
            # Return 10 generic questions and a fallback answer
            generic_questions = [f"Question {i+1}" for i in range(10)]
            fallback_answer = "Information not available."
            return generic_questions, fallback_answer

    @staticmethod
    def _parse_bucketed_response(llm_output: str) -> List[Dict]:
        """
        Parse LLM response in bucketed format to extract grouped Q&A pairs

        Expected format:
        BUCKET_1:
        Q: question 1
        Q: question 2
        ANSWER: answer text

        BUCKET_2:
        Q: question 3
        ANSWER: answer text

        Returns:
            List[Dict]: List of Q&A pairs with is_bucketed flag and bucket_id
        """
        try:
            result_pairs = []
            current_bucket_id = None
            current_questions = []
            current_answer = ""

            lines = llm_output.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for bucket marker (BUCKET_1:, BUCKET_2:, etc.)
                if line.startswith('BUCKET_') and ':' in line:
                    # Save previous bucket if exists
                    if current_bucket_id and current_questions and current_answer:
                        for question in current_questions:
                            result_pairs.append({
                                "question": question,
                                "answer": current_answer,
                                "is_bucketed": True,
                                "bucket_id": current_bucket_id
                            })

                    # Start new bucket
                    current_bucket_id = line.split(':')[0].strip()
                    current_questions = []
                    current_answer = ""

                # Check for question (Q:)
                elif line.startswith('Q:'):
                    question_text = line.split(':', 1)[1].strip()
                    current_questions.append(question_text)

                # Check for answer (ANSWER:)
                elif line.startswith('ANSWER:'):
                    current_answer = line.split(':', 1)[1].strip()

            # Save the last bucket
            if current_bucket_id and current_questions and current_answer:
                for question in current_questions:
                    result_pairs.append({
                        "question": question,
                        "answer": current_answer,
                        "is_bucketed": True,
                        "bucket_id": current_bucket_id
                    })

            logger.info(f"✓ Parsed {len(result_pairs)} Q&A pairs from bucketed response")

            # Count unique buckets
            unique_buckets = len(set(pair["bucket_id"] for pair in result_pairs))
            logger.info(f"✓ Found {unique_buckets} buckets")

            return result_pairs

        except Exception as e:
            logger.error(f"❌ Error parsing bucketed response: {e}")
            return []


# ============================================================================
# STATE SCHEMA
# ============================================================================

class ChatbotState(TypedDict):
    """State that flows through the LangGraph"""
    messages: List[Dict[str, str]]  # Chat history: [{"role": "user/assistant", "content": "..."}]
    user_query: str                  # Current user query
    llm_response: str                # LLM's response
    system_prompt: str               # System instructions for LLM
    error: str                       # Error message if any


# ============================================================================
# LLM AGENT NODE
# ============================================================================

async def llm_agent_node(state: ChatbotState) -> ChatbotState:
    """
    LangGraph Node: Process user query with LLM

    Takes user input, sends to Groq LLM, returns response
    """
    user_query = state.get("user_query", "")
    system_prompt = state.get("system_prompt", "You are a helpful AI assistant.")
    messages = state.get("messages", [])

    if not user_query:
        logger.warning("No user query provided")
        return {
            **state,
            "llm_response": "I didn't receive a query. Please ask me something!",
            "error": "No user query"
        }

    logger.info(f"Processing query: {user_query[:50]}...")

    try:
        # Initialize Groq LLM
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key,
            temperature=0.7,
            max_tokens=1000
        )

        # Build message chain for LLM
        llm_messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                llm_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                llm_messages.append(AIMessage(content=msg["content"]))

        # Add current user query
        llm_messages.append(HumanMessage(content=user_query))

        # Get LLM response
        response = await llm.ainvoke(llm_messages)
        llm_response = response.content

        logger.info(f"LLM response generated: {len(llm_response)} characters")

        # Update conversation history
        updated_messages = messages + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": llm_response}
        ]

        return {
            **state,
            "llm_response": llm_response,
            "messages": updated_messages,
            "error": ""
        }

    except Exception as e:
        logger.error(f"LLM agent failed: {e}")
        return {
            **state,
            "llm_response": f"Sorry, I encountered an error: {str(e)}",
            "error": str(e)
        }


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_chatbot_graph():
    """
    Build simple LangGraph workflow (no checkpointing):

    START → llm_agent → END

    Returns:
        Compiled LangGraph
    """
    # Create graph builder
    builder = StateGraph(ChatbotState)

    # Add LLM agent node
    builder.add_node("llm_agent", llm_agent_node)

    # Define edges
    builder.add_edge(START, "llm_agent")
    builder.add_edge("llm_agent", END)

    # Compile graph without checkpointer
    graph = builder.compile()

    logger.info("LangGraph chatbot compiled successfully with AsyncSqliteSaver")

    return graph


# ============================================================================
# CHATBOT CLASS
# ============================================================================

class LangGraphChatbot:
    """
    Simple chatbot using LangGraph + Groq LLM with persistence

    Usage:
        chatbot = LangGraphChatbot(graph, system_prompt)
        response = await chatbot.chat("Hello, how are you?", thread_id="user_123")
    """

    def __init__(self, graph, system_prompt: str = None):
        """
        Initialize chatbot with pre-built graph

        Args:
            graph: Compiled LangGraph with checkpointer
            system_prompt: Custom system instructions for LLM
        """
        self.graph = graph
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        logger.info("LangGraphChatbot initialized")

    async def chat(self, user_query: str, thread_id: str = "default") -> str:
        """
        Send a query to the chatbot and get response

        Args:
            user_query: User's question/message
            thread_id: Thread ID for conversation persistence (default: "default")

        Returns:
            LLM's response
        """
        # Create initial state
        initial_state = {
            "messages": [],  # Will be loaded from checkpoint
            "user_query": user_query,
            "llm_response": "",
            "system_prompt": self.system_prompt,
            "error": ""
        }

        # Config with thread_id for persistence
        config = {"configurable": {"thread_id": thread_id}}

        # Run LangGraph with checkpointer
        result = await self.graph.ainvoke(initial_state, config)

        # Return response (history managed by checkpointer)
        return result.get("llm_response", "No response generated")

    async def reset_conversation(self, thread_id: str = "default"):
        """
        Clear conversation history for a thread

        Args:
            thread_id: Thread ID to reset
        """
        # Get checkpointer and clear the thread
        config = {"configurable": {"thread_id": thread_id}}
        # Note: LangGraph will handle this through checkpoint management
        logger.info(f"Conversation history for thread '{thread_id}' can be reset by deleting checkpoint")

    async def get_history(self, thread_id: str = "default") -> List[Dict[str, str]]:
        """
        Get conversation history for a thread

        Args:
            thread_id: Thread ID to retrieve history for

        Returns:
            List of message dictionaries
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.aget_state(config)
        return state.values.get("messages", []) if state else []


# ============================================================================
# TESTING
# ============================================================================

async def test_chatbot():
    """Test the chatbot"""
    print("\n" + "="*60)
    print("LANGGRAPH CHATBOT - SIMPLE BACKEND")
    print("="*60)

    # Initialize chatbot
    chatbot = LangGraphChatbot(
        system_prompt="You are a helpful AI assistant for the Starship Chatbot project."
    )

    # Test queries
    test_queries = [
        "Hello! What can you help me with?",
        "What is LangGraph?",
        "How does it work with FastAPI?"
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        response = await chatbot.chat(query)
        print(f"Assistant: {response}")

    print("\n" + "="*60)
    print("CONVERSATION HISTORY:")
    print("="*60)
    for msg in chatbot.get_history():
        print(f"{msg['role'].upper()}: {msg['content'][:100]}...")


# ============================================================================
# Q&A MODIFICATION SYSTEM (Supervisor + Simplifier)
# ============================================================================

# Main thread ID for Q&A modifications
QA_THREAD_ID = "qa_modifications_main"
SELECTIVE_THREAD_ID = "selective_modifications_main"

# ⚠️  WARNING: These are DIFFERENT checkpoints!
# - Selective modifications save to: SELECTIVE_THREAD_ID
# - HTML viewer reads from: QA_THREAD_ID
# This causes changes to NOT appear in HTML viewer after reload!


class QAModifierState(TypedDict):
    """State for Q&A modification workflow with versioning"""
    user_request: str
    current_data: List[Dict]      # Current Q&A data (from state)
    target_topic: str
    modification_type: str
    modified_data: List[Dict]     # Modified Q&A data
    agent_response: str
    error: str
    version: int                  # Version number
    timestamp: str                # Modification timestamp


async def supervisor_agent_node(state: QAModifierState) -> QAModifierState:
    """Supervisor Agent: Routes to Simplifier"""
    user_request = state.get("user_request", "")
    if not user_request:
        return {**state, "modification_type": "unknown", "error": "No request"}

    logger.info(f"Supervisor: {user_request[:100]}...")

    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=os.getenv('GROQ_API_KEY'),
            temperature=0.3,
            max_tokens=300
        )

        messages = [
            SystemMessage(content="""Analyze Q&A modification requests.
Output format:
TYPE: simplify
TOPIC: [topic name or "all"]"""),
            HumanMessage(content=f"Request: {user_request}")
        ]

        response = await llm.ainvoke(messages)
        mod_type = "simplify" if "simplify" in response.content.lower() else "unknown"
        topic = "all"

        for line in response.content.split('\n'):
            if "TOPIC:" in line:
                topic = line.split("TOPIC:")[1].strip()

        logger.info(f"Supervisor: type={mod_type}, topic={topic}")
        return {**state, "modification_type": mod_type, "target_topic": topic, "error": ""}

    except Exception as e:
        logger.error(f"Supervisor failed: {e}")
        return {**state, "modification_type": "unknown", "error": str(e)}


async def simplifier_agent_node(state: QAModifierState) -> QAModifierState:
    """Simplifier Agent: Simplifies Q&A pairs"""
    current_data = state.get("current_data", [])
    target_topic = state.get("target_topic", "all")

    if not current_data:
        return {**state, "modified_data": [], "agent_response": "No data", "error": "No data"}

    logger.info(f"Simplifying: {target_topic}")

    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=os.getenv('GROQ_API_KEY'),
            temperature=0.3,
            max_tokens=2000
        )

        modified_data = []
        topics_modified = 0

        for item in current_data:
            topic_name = item.get("topic", "")
            should_modify = (target_topic.lower() == "all" or target_topic.lower() in topic_name.lower())

            if should_modify and item.get("qa_pairs"):
                qa_text = "\n\n".join([f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}"
                                      for i, qa in enumerate(item["qa_pairs"])])

                messages = [
                    SystemMessage(content="Simplify Q&A to 8th grade level. Keep URLs/phones. Format: Q1: ... A1: ..."),
                    HumanMessage(content=qa_text)
                ]

                response = await llm.ainvoke(messages)
                simplified_pairs = []
                current_q = current_a = None

                for line in response.content.split('\n'):
                    line = line.strip()
                    if line.startswith('Q') and ':' in line:
                        if current_q and current_a:
                            simplified_pairs.append({"question": current_q, "answer": current_a})
                        current_q = line.split(':', 1)[1].strip()
                    elif line.startswith('A') and ':' in line:
                        current_a = line.split(':', 1)[1].strip()

                if current_q and current_a:
                    simplified_pairs.append({"question": current_q, "answer": current_a})

                modified_item = item.copy()
                modified_item["qa_pairs"] = simplified_pairs
                modified_item["qa_count"] = len(simplified_pairs)
                modified_data.append(modified_item)
                topics_modified += 1
            else:
                modified_data.append(item)

        return {**state, "modified_data": modified_data, "agent_response": f"Simplified {topics_modified} topics", "error": "", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Simplifier failed: {e}")
        return {**state, "modified_data": current_data, "agent_response": f"Failed: {e}", "error": str(e), "timestamp": datetime.now().isoformat()}


def route_to_agent(state: QAModifierState) -> Literal["simplifier", "end"]:
    """Route to Simplifier or END"""
    return "simplifier" if state.get("modification_type") == "simplify" else "end"


# ============================================================================
# SELECTIVE MODIFICATION SYSTEM (3-Agent Pattern)
# ============================================================================

class SelectiveModificationState(TypedDict):
    """State for 3-agent selective modification workflow"""
    # INPUT (from API)
    user_request: str
    topic_index: int
    selected_qa_indices: List[int]
    all_data: List[Dict]

    # AGENT 1 OUTPUT (Supervisor Extraction)
    extracted_topic: Dict
    extracted_qa_pairs: List[Dict]
    extraction_summary: str

    # AGENT 2 OUTPUT (Main Modification)
    modified_qa_pairs: List[Dict]
    modification_summary: str

    # AGENT 3 OUTPUT (Wrapper Reassembly)
    final_data: List[Dict]
    agent_response: str

    # TRACKING
    version: int
    timestamp: str
    error: str


# ============================================================================
# MERGE Q&A SYSTEM (3-Agent Pattern: Extract → Merge → Reassemble)
# ============================================================================

class MergeQAState(TypedDict):
    """
    State for 3-agent merge workflow

    Workflow: Agent1 (Extract N Q&A) → Merge Agent (Merge with LLM) → Agent3 (Append merged)
    """
    # INPUT (from API)
    user_request: str  # Optional merge instruction
    topic_index: int
    selected_qa_indices: List[int]  # Must have 2 or more indices
    all_data: List[Dict]

    # AGENT 1 OUTPUT (Supervisor Extraction - REUSED)
    extracted_topic: Dict
    extracted_qa_pairs: List[Dict]  # 2 or more Q&A pairs
    extraction_summary: str

    # MERGE AGENT OUTPUT (New Agent with Groq LLM)
    merged_qa_pair: Dict  # Single merged Q&A pair
    merge_summary: str

    # AGENT 3 OUTPUT (Wrapper Reassembly - REUSED)
    # NOTE: Originals kept intact, merged pair appended at end
    final_data: List[Dict]
    agent_response: str

    # TRACKING
    version: int
    timestamp: str
    error: str


# ============================================================================
# SELECTIVE MODIFICATION - AGENT FUNCTIONS
# ============================================================================

async def agent1_supervisor_extraction(state: SelectiveModificationState) -> SelectiveModificationState:
    """
    AGENT 1: Supervisor Extraction (No LLM - Simple Python)

    Extracts selected Q&A pairs using indices from all_data
    """
    all_data = state.get("all_data", [])
    topic_index = state.get("topic_index", 0)
    selected_qa_indices = state.get("selected_qa_indices", [])

    logger.info(f"🔍 Agent 1 (Supervisor): Extracting data from topic #{topic_index}")

    try:
        # Extract topic using index
        extracted_topic = all_data[topic_index].copy()
        topic_name = extracted_topic.get("topic", "Unknown Topic")

        # Extract selected Q&A pairs using indices
        all_qa_pairs = extracted_topic.get("qa_pairs", [])
        extracted_qa_pairs = [all_qa_pairs[i].copy() for i in selected_qa_indices if i < len(all_qa_pairs)]

        extraction_summary = f"Extracted {len(extracted_qa_pairs)} Q&A pairs from '{topic_name}'"

        logger.info(f"✓ Agent 1: {extraction_summary}")

        return {
            **state,
            "extracted_topic": extracted_topic,
            "extracted_qa_pairs": extracted_qa_pairs,
            "extraction_summary": extraction_summary,
            "error": ""
        }

    except Exception as e:
        logger.error(f"❌ Agent 1 failed: {e}")
        return {
            **state,
            "extracted_topic": {},
            "extracted_qa_pairs": [],
            "extraction_summary": f"Extraction failed: {e}",
            "error": str(e)
        }


async def agent2_main_modification(state: SelectiveModificationState) -> SelectiveModificationState:
    """
    AGENT 2: Main Modification (Uses LLM)

    Modifies the extracted Q&A pairs based on user request
    """
    extracted_qa_pairs = state.get("extracted_qa_pairs", [])
    selected_qa_indices = state.get("selected_qa_indices", [])
    user_request = state.get("user_request", "")

    if not extracted_qa_pairs:
        return {
            **state,
            "modified_qa_pairs": [],
            "modification_summary": "No Q&A pairs to modify",
            "error": "No extracted Q&A pairs"
        }

    logger.info(f"🤖 Agent 2 (Main): Modifying {len(extracted_qa_pairs)} Q&A pairs")

    try:
        # Initialize LLM
        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=os.getenv('GROQ_API_KEY'),
            temperature=0.3,
            max_tokens=2000
        )

        # Format Q&A pairs for LLM (preserve original numbering)
        qa_text = "\n\n".join([
            f"Q{selected_qa_indices[i]+1}: {qa['question']}\nA{selected_qa_indices[i]+1}: {qa['answer']}"
            for i, qa in enumerate(extracted_qa_pairs)
        ])

        # Create prompt
        messages = [
            SystemMessage(content=f"""You are modifying Q&A pairs based on the user's request.

User Request: {user_request}

IMPORTANT: Apply the modification to ALL the Q&A pairs provided below.
Keep the original question numbers (Q1, Q2, Q3, etc.) in your response.
Return the modified Q&A pairs in the exact same format: Q#: ... A#: ...

If the user request doesn't make sense for some pairs, still try to apply it in a reasonable way."""),
            HumanMessage(content=qa_text)
        ]

        # Get LLM response
        response = await llm.ainvoke(messages)

        # Parse modified Q&A pairs
        modified_pairs = []
        current_q = current_a = None

        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_q and current_a:
                    modified_pairs.append({"question": current_q, "answer": current_a})
                current_q = line.split(':', 1)[1].strip()
            elif line.startswith('A') and ':' in line:
                current_a = line.split(':', 1)[1].strip()

        if current_q and current_a:
            modified_pairs.append({"question": current_q, "answer": current_a})

        modification_summary = f"Modified {len(modified_pairs)} Q&A pairs using LLM"

        logger.info(f"✓ Agent 2: {modification_summary}")

        # 🔍 DETAILED OUTPUT: Show what Agent 2 is passing to Agent 3
        logger.info("="*60)
        logger.info("📤 AGENT 2 OUTPUT (Passing to Agent 3):")
        logger.info(f"  Number of modified Q&A pairs: {len(modified_pairs)}")
        for idx, qa_pair in enumerate(modified_pairs):
            logger.info(f"  --- Modified Pair #{idx + 1} ---")
            logger.info(f"  Question: {qa_pair.get('question', 'N/A')[:100]}...")
            logger.info(f"  Answer: {qa_pair.get('answer', 'N/A')[:100]}...")
        logger.info("="*60)

        return {
            **state,
            "modified_qa_pairs": modified_pairs,
            "modification_summary": modification_summary,
            "error": ""
        }

    except Exception as e:
        logger.error(f"❌ Agent 2 failed: {e}")
        return {
            **state,
            "modified_qa_pairs": extracted_qa_pairs,  # Return original if failed
            "modification_summary": f"Modification failed: {e}",
            "error": str(e)
        }


async def agent3_wrapper_reassembly(state: SelectiveModificationState) -> SelectiveModificationState:
    """
    AGENT 3: Wrapper Reassembly (No LLM - Simple Python)

    Puts modified Q&A pairs back into their original positions in all_data
    """
    all_data = state.get("all_data", [])
    topic_index = state.get("topic_index", 0)
    selected_qa_indices = state.get("selected_qa_indices", [])
    modified_qa_pairs = state.get("modified_qa_pairs", [])

    logger.info(f"📦 Agent 3 (Wrapper): Reassembling data")

    # 🔍 DETAILED INPUT: Show what Agent 3 received from Agent 2
    logger.info("="*60)
    logger.info("📥 AGENT 3 INPUT (Received from Agent 2):")
    logger.info(f"  Number of modified Q&A pairs received: {len(modified_qa_pairs)}")
    logger.info(f"  Selected indices to replace: {selected_qa_indices}")
    logger.info(f"  Topic index: {topic_index}")
    for idx, qa_pair in enumerate(modified_qa_pairs):
        logger.info(f"  --- Received Pair #{idx + 1} ---")
        logger.info(f"  Question: {qa_pair.get('question', 'N/A')[:100]}...")
        logger.info(f"  Answer: {qa_pair.get('answer', 'N/A')[:100]}...")
    logger.info("="*60)

    try:
        # Create deep copy of all data
        final_data = [topic.copy() for topic in all_data]

        # Get the topic to modify
        topic_to_modify = final_data[topic_index]
        qa_pairs = topic_to_modify.get("qa_pairs", []).copy()

        # Replace selected Q&A pairs with modified versions
        for i, qa_index in enumerate(selected_qa_indices):
            if i < len(modified_qa_pairs) and qa_index < len(qa_pairs):
                qa_pairs[qa_index] = modified_qa_pairs[i]

        # Update topic with modified Q&A pairs
        topic_to_modify["qa_pairs"] = qa_pairs
        topic_to_modify["qa_count"] = len(qa_pairs)
        final_data[topic_index] = topic_to_modify

        topic_name = all_data[topic_index].get("topic", "Unknown Topic")
        agent_response = f"Modified {len(modified_qa_pairs)} Q&A pairs in '{topic_name}'"

        logger.info(f"✓ Agent 3: {agent_response}")

        # 🔍 DETAILED OUTPUT: Show final reassembled data
        logger.info("="*60)
        logger.info("📤 AGENT 3 OUTPUT (Final Result):")
        logger.info(f"  Topic modified: '{topic_name}'")
        logger.info(f"  Total topics in final_data: {len(final_data)}")
        logger.info(f"  Modified Q&A pairs count: {len(modified_qa_pairs)}")
        logger.info(f"  Replaced at indices: {selected_qa_indices}")
        logger.info(f"  Total Q&A pairs in topic after modification: {len(qa_pairs)}")
        # Show the modified Q&A pairs in their final positions
        for i, qa_index in enumerate(selected_qa_indices):
            if qa_index < len(qa_pairs):
                logger.info(f"  --- Final Q&A at index {qa_index} ---")
                logger.info(f"  Question: {qa_pairs[qa_index].get('question', 'N/A')[:100]}...")
                logger.info(f"  Answer: {qa_pairs[qa_index].get('answer', 'N/A')[:100]}...")
        logger.info("="*60)

        return {
            **state,
            "final_data": final_data,
            "agent_response": agent_response,
            "timestamp": datetime.now().isoformat(),
            "error": ""
        }

    except Exception as e:
        logger.error(f"❌ Agent 3 failed: {e}")
        return {
            **state,
            "final_data": all_data,  # Return original if failed
            "agent_response": f"Reassembly failed: {e}",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# ============================================================================
# MERGE Q&A - AGENT FUNCTIONS
# ============================================================================

async def agent_merge_qa_pairs(state: MergeQAState) -> MergeQAState:
    """
    MERGE AGENT: Intelligently merges 2+ Q&A pairs into 1 using Groq LLM

    Takes 2 or more Q&A pairs and uses LLM to create 1 comprehensive merged Q&A pair
    """
    extracted_qa_pairs = state.get("extracted_qa_pairs", [])
    user_request = state.get("user_request", "Merge these Q&A pairs intelligently")

    if len(extracted_qa_pairs) < 2:
        return {
            **state,
            "merged_qa_pair": {},
            "merge_summary": f"Error: Need at least 2 Q&A pairs, got {len(extracted_qa_pairs)}",
            "error": "Invalid number of Q&A pairs for merge"
        }

    num_pairs = len(extracted_qa_pairs)
    logger.info(f"🔀 Merge Agent: Merging {num_pairs} Q&A pairs using Groq LLM")

    try:
        # Initialize Groq LLM (same as Agent 2)
        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=os.getenv('GROQ_API_KEY'),
            temperature=0.3,
            max_tokens=2000
        )

        # Build Q&A pairs list dynamically
        qa_pairs_text = ""
        for i, qa in enumerate(extracted_qa_pairs, 1):
            qa_pairs_text += f"""
Q&A PAIR {i}:
Question: {qa['question']}
Answer: {qa['answer']}
"""

        # Create prompt for merging
        merge_prompt = f"""You are an expert at merging Q&A pairs intelligently.

USER REQUEST: {user_request}

You have {num_pairs} Q&A pairs that need to be merged into 1 comprehensive Q&A pair.
{qa_pairs_text}
YOUR TASK:
1. Create 1 merged question that captures the essence of ALL {num_pairs} questions
2. Create 1 merged answer that comprehensively answers ALL {num_pairs} questions
3. The merged Q&A should be clear, concise, and complete
4. Preserve all important information from all pairs
5. Eliminate redundancy while maintaining clarity

FORMAT YOUR RESPONSE EXACTLY AS:
MERGED_QUESTION: [your merged question here]
MERGED_ANSWER: [your merged answer here]

Be intelligent and natural in the merge - don't just concatenate!"""

        messages = [
            SystemMessage(content="You are an expert at intelligently merging Q&A pairs."),
            HumanMessage(content=merge_prompt)
        ]

        # Call LLM
        logger.info(f"  Calling Groq LLM to merge Q&A pairs...")
        response = await llm.ainvoke(messages)

        # Parse response
        response_text = response.content.strip()
        merged_question = ""
        merged_answer = ""

        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('MERGED_QUESTION:'):
                merged_question = line.replace('MERGED_QUESTION:', '').strip()
            elif line.startswith('MERGED_ANSWER:'):
                merged_answer = line.replace('MERGED_ANSWER:', '').strip()

        # Validate parsing
        if not merged_question or not merged_answer:
            # Fallback: try to split by first occurrence
            if 'MERGED_ANSWER:' in response_text:
                parts = response_text.split('MERGED_ANSWER:', 1)
                merged_question = parts[0].replace('MERGED_QUESTION:', '').strip()
                merged_answer = parts[1].strip()

        merged_qa_pair = {
            "question": merged_question if merged_question else "Merged Question",
            "answer": merged_answer if merged_answer else response_text
        }

        merge_summary = f"Successfully merged {num_pairs} Q&A pairs into 1 using LLM"

        logger.info(f"✓ Merge Agent: {merge_summary}")
        logger.info("="*60)
        logger.info("📤 MERGE AGENT OUTPUT:")
        logger.info(f"  Merged Question: {merged_qa_pair['question'][:100]}...")
        logger.info(f"  Merged Answer: {merged_qa_pair['answer'][:100]}...")
        logger.info("="*60)

        return {
            **state,
            "merged_qa_pair": merged_qa_pair,
            "merge_summary": merge_summary,
            "error": ""
        }

    except Exception as e:
        logger.error(f"❌ Merge Agent failed: {e}")
        # Fallback: simple concatenation
        all_questions = " ".join([qa['question'] for qa in extracted_qa_pairs])
        all_answers = " ".join([qa['answer'] for qa in extracted_qa_pairs])
        merged_qa_pair = {
            "question": all_questions,
            "answer": all_answers
        }
        return {
            **state,
            "merged_qa_pair": merged_qa_pair,
            "merge_summary": f"Merge failed, used simple concatenation: {e}",
            "error": str(e)
        }


async def agent1_extraction_for_merge(state: MergeQAState) -> MergeQAState:
    """
    Wrapper for Agent1 to work with MergeQAState

    Extracts 2 or more Q&A pairs for merging
    """
    all_data = state.get("all_data", [])
    topic_index = state.get("topic_index", 0)
    selected_qa_indices = state.get("selected_qa_indices", [])

    num_selected = len(selected_qa_indices)
    logger.info(f"🔍 Agent 1 (Extraction for Merge): Extracting {num_selected} Q&A pairs from topic #{topic_index}")

    # Validate at least 2 indices
    if num_selected < 2:
        return {
            **state,
            "extracted_qa_pairs": [],
            "extraction_summary": f"Error: Need at least 2 indices, got {num_selected}",
            "error": "Must select at least 2 Q&A pairs to merge"
        }

    try:
        # Extract topic
        extracted_topic = all_data[topic_index].copy()
        topic_name = extracted_topic.get("topic", "Unknown Topic")

        # Extract selected Q&A pairs
        all_qa_pairs = extracted_topic.get("qa_pairs", [])
        extracted_qa_pairs = [all_qa_pairs[i].copy() for i in selected_qa_indices if i < len(all_qa_pairs)]

        if len(extracted_qa_pairs) < 2:
            return {
                **state,
                "extracted_qa_pairs": [],
                "extraction_summary": f"Error: Could only extract {len(extracted_qa_pairs)} Q&A pairs",
                "error": "Failed to extract at least 2 Q&A pairs"
            }

        extraction_summary = f"Extracted {len(extracted_qa_pairs)} Q&A pairs from '{topic_name}' for merging"

        logger.info(f"✓ Agent 1: {extraction_summary}")

        return {
            **state,
            "extracted_topic": extracted_topic,
            "extracted_qa_pairs": extracted_qa_pairs,
            "extraction_summary": extraction_summary,
            "error": ""
        }

    except Exception as e:
        logger.error(f"❌ Agent 1 (Merge) failed: {e}")
        return {
            **state,
            "extracted_qa_pairs": [],
            "extraction_summary": f"Extraction failed: {e}",
            "error": str(e)
        }


async def agent3_reassembly_for_merge(state: MergeQAState) -> MergeQAState:
    """
    Wrapper for Agent3 to work with MergeQAState

    Keeps original Q&A pairs intact and appends merged Q&A pair at the end
    """
    all_data = state.get("all_data", [])
    topic_index = state.get("topic_index", 0)
    selected_qa_indices = state.get("selected_qa_indices", [])
    merged_qa_pair = state.get("merged_qa_pair", {})
    num_selected = len(selected_qa_indices)

    logger.info(f"📦 Agent 3 (Reassembly for Merge): Appending merged Q&A to dataset")

    try:
        # Create deep copy of all data
        final_data = [topic.copy() for topic in all_data]

        # Get the topic to modify
        topic_to_modify = final_data[topic_index]
        qa_pairs = topic_to_modify.get("qa_pairs", []).copy()

        # KEEP originals intact - just append the merged Q&A at the end
        qa_pairs.append(merged_qa_pair)

        # Update topic
        topic_to_modify["qa_pairs"] = qa_pairs
        topic_to_modify["qa_count"] = len(qa_pairs)
        final_data[topic_index] = topic_to_modify

        topic_name = all_data[topic_index].get("topic", "Unknown Topic")
        agent_response = f"Merged {num_selected} Q&A pairs into 1 and appended to '{topic_name}' (originals kept)"

        logger.info(f"✓ Agent 3: {agent_response}")
        logger.info(f"  Original Q&A pairs kept intact: {selected_qa_indices}")
        logger.info(f"  Appended merged Q&A at index: {len(qa_pairs) - 1}")
        logger.info(f"  Total Q&A pairs after merge: {len(qa_pairs)}")

        return {
            **state,
            "final_data": final_data,
            "agent_response": agent_response,
            "timestamp": datetime.now().isoformat(),
            "error": ""
        }

    except Exception as e:
        logger.error(f"❌ Agent 3 (Merge) failed: {e}")
        return {
            **state,
            "final_data": all_data,
            "agent_response": f"Merge reassembly failed: {e}",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def build_selective_modifier_graph():
    """
    Build 3-Agent Selective Modifier graph (no checkpointing)

    Workflow: START → Agent1 (Extraction) → Agent2 (Modification) → Agent3 (Reassembly) → END

    Returns:
        Compiled LangGraph
    """
    builder = StateGraph(SelectiveModificationState)

    # Add 3 agent nodes
    builder.add_node("agent1_extraction", agent1_supervisor_extraction)
    builder.add_node("agent2_modification", agent2_main_modification)
    builder.add_node("agent3_reassembly", agent3_wrapper_reassembly)

    # Define sequential flow
    builder.add_edge(START, "agent1_extraction")
    builder.add_edge("agent1_extraction", "agent2_modification")
    builder.add_edge("agent2_modification", "agent3_reassembly")
    builder.add_edge("agent3_reassembly", END)

    logger.info("✓ Selective Modifier graph (3-agent) compiled")
    return builder.compile()


def build_merge_qa_graph():
    """
    Build 3-Agent Merge Q&A graph (no checkpointing)

    Workflow: START → Agent1 (Extract 2+ Q&A) → Merge Agent (LLM Merge) → Agent3 (Append merged) → END

    Returns:
        Compiled LangGraph
    """
    builder = StateGraph(MergeQAState)

    # Add 3 agent nodes
    builder.add_node("agent1_extraction", agent1_extraction_for_merge)
    builder.add_node("agent_merge", agent_merge_qa_pairs)
    builder.add_node("agent3_reassembly", agent3_reassembly_for_merge)

    # Define sequential flow: Extract → Merge → Reassemble
    builder.add_edge(START, "agent1_extraction")
    builder.add_edge("agent1_extraction", "agent_merge")
    builder.add_edge("agent_merge", "agent3_reassembly")
    builder.add_edge("agent3_reassembly", END)

    logger.info("✓ Merge Q&A graph (3-agent) compiled")
    return builder.compile()


def build_qa_modifier_graph():
    """
    Build Q&A Modifier graph (no checkpointing)

    Returns:
        Compiled LangGraph
    """
    builder = StateGraph(QAModifierState)
    builder.add_node("supervisor", supervisor_agent_node)
    builder.add_node("simplifier", simplifier_agent_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_to_agent, {"simplifier": "simplifier", "end": END})
    builder.add_edge("simplifier", END)

    logger.info("Q&A Modifier graph compiled")
    return builder.compile()


class QAWorkflowManager:
    """
    Q&A Workflow Manager - Handles general, selective, and merge modifications

    Manages three separate LangGraph workflows:
    1. General Modifier (QAModifierState) - Simplifies entire topics
    2. Selective Modifier (SelectiveModificationState) - Modifies specific Q&A pairs within a topic
    3. Merge Modifier (MergeQAState) - Merges 2 Q&A pairs into 1
    """

    def __init__(self, general_graph=None, selective_graph=None, merge_graph=None):
        """
        Initialize workflow manager with pre-built graphs

        Args:
            general_graph: Compiled LangGraph for general modifications (uses QAModifierState) - Optional
            selective_graph: Compiled LangGraph for selective modifications (uses SelectiveModificationState) - Optional
            merge_graph: Compiled LangGraph for merging Q&A pairs (uses MergeQAState) - Optional
        """
        self.general_graph = general_graph
        self.selective_graph = selective_graph
        self.merge_graph = merge_graph

        active_graphs = []
        if general_graph:
            active_graphs.append("general")
        if selective_graph:
            active_graphs.append("selective")
        if merge_graph:
            active_graphs.append("merge")

        logger.info(f"QAWorkflowManager initialized with graphs: {', '.join(active_graphs)}")

    async def modify(self, user_request: str, current_data: List[Dict], thread_id: str = QA_THREAD_ID) -> Dict:
        """
        Modify Q&A pairs and persist in LangGraph checkpoint

        Uses QAModifierState with self.general_graph

        Args:
            user_request: User's modification request
            current_data: Current Q&A data
            thread_id: Thread ID for versioning (default: QA_THREAD_ID)

        Returns:
            Dict with modified_data, agent_response, error
        """
        # Get previous state to determine version
        config = {"configurable": {"thread_id": thread_id}}

        try:
            prev_state = await self.general_graph.aget_state(config)
            prev_version = prev_state.values.get("version", 0) if prev_state and prev_state.values else 0
        except:
            prev_version = 0

        # Create initial state
        initial_state = {
            "user_request": user_request,
            "current_data": current_data,
            "target_topic": "",
            "modification_type": "",
            "modified_data": [],
            "agent_response": "",
            "error": "",
            "version": prev_version + 1,
            "timestamp": datetime.now().isoformat()
        }

        # Run general modifier graph with checkpointer (uses QAModifierState)
        result = await self.general_graph.ainvoke(initial_state, config)

        logger.info(f"✓ Created checkpoint version {result.get('version')}")

        return {
            "modified_data": result.get("modified_data", []),
            "agent_response": result.get("agent_response", ""),
            "error": result.get("error", ""),
            "version": result.get("version", prev_version + 1)
        }

    async def modify_selective(
        self,
        user_request: str,
        topic_index: int,
        selected_qa_indices: List[int],
        selected_qa_pairs: List[Dict],
        all_data: List[Dict],
        thread_id: str = SELECTIVE_THREAD_ID
    ) -> Dict:
        """
        Modify specific Q&A pairs within a single topic using 3-agent workflow

        Uses SelectiveModificationState with self.selective_graph

        Workflow: Agent1 (Extract) → Agent2 (Modify with LLM) → Agent3 (Reassemble)

        Args:
            user_request: User's modification request
            topic_index: Index of the topic to modify
            selected_qa_indices: Indices of Q&A pairs to modify
            selected_qa_pairs: The actual Q&A pairs selected for modification (unused, indices are used instead)
            all_data: All current Q&A data
            thread_id: Thread ID for versioning (default: SELECTIVE_THREAD_ID)

        Returns:
            Dict with modified_data, agent_response, error
        """
        # Get previous state to determine version
        config = {"configurable": {"thread_id": thread_id}}

        try:
            prev_state = await self.selective_graph.aget_state(config)
            prev_version = prev_state.values.get("version", 0) if prev_state and prev_state.values else 0
        except:
            prev_version = 0

        logger.info("="*60)
        logger.info(f"🟢 STEP 3: Inside modify_selective() method")
        logger.info(f"🚀 Starting 3-agent selective modification workflow (v{prev_version + 1})")
        logger.info(f"  Topic index: {topic_index}")
        logger.info(f"  Selected indices: {selected_qa_indices}")
        logger.info(f"  User request: {user_request[:50]}...")
        logger.info("="*60)

        # Create initial state for 3-agent workflow
        initial_state = {
            "user_request": user_request,
            "topic_index": topic_index,
            "selected_qa_indices": selected_qa_indices,
            "all_data": all_data,
            "extracted_topic": {},
            "extracted_qa_pairs": [],
            "extraction_summary": "",
            "modified_qa_pairs": [],
            "modification_summary": "",
            "final_data": [],
            "agent_response": "",
            "version": prev_version + 1,
            "timestamp": "",
            "error": ""
        }

        try:
            # Run 3-agent graph: Agent1 → Agent2 → Agent3
            logger.info("🟠 STEP 4: About to invoke selective_graph.ainvoke()...")
            logger.info(f"  Graph type: {type(self.selective_graph)}")
            logger.info(f"  Initial state keys: {list(initial_state.keys())}")
            logger.info(f"  Config: {config}")
            logger.info(f"  ⚠️  SAVING TO CHECKPOINT: {thread_id}")
            result = await self.selective_graph.ainvoke(initial_state, config)

            logger.info(f"✓ 3-agent workflow completed (v{prev_version + 1})")
            logger.info(f"  - Agent 1: {result.get('extraction_summary', 'N/A')}")
            logger.info(f"  - Agent 2: {result.get('modification_summary', 'N/A')}")
            logger.info(f"  - Agent 3: {result.get('agent_response', 'N/A')}")

            return {
                "modified_data": result.get("final_data", all_data),
                "agent_response": result.get("agent_response", ""),
                "error": result.get("error", ""),
                "version": prev_version + 1
            }

        except Exception as e:
            logger.error(f"❌ 3-agent selective modification failed: {e}")
            return {
                "modified_data": all_data,
                "agent_response": f"Modification failed: {str(e)}",
                "error": str(e),
                "version": prev_version + 1
            }

    async def merge_qa_pairs(
        self,
        topic_index: int,
        selected_qa_indices: List[int],
        all_data: List[Dict],
        user_request: str = "Merge these Q&A pairs intelligently",
        thread_id: str = SELECTIVE_THREAD_ID
    ) -> Dict:
        """
        Merge 2 or more Q&A pairs into 1 using 3-agent workflow

        Uses MergeQAState with self.merge_graph

        Workflow: Agent1 (Extract N) → Merge Agent (LLM) → Agent3 (Append merged, keep originals)

        Args:
            topic_index: Index of the topic containing Q&A pairs
            selected_qa_indices: Indices of 2 or more Q&A pairs to merge
            all_data: All current Q&A data
            user_request: Optional merge instruction (default: intelligent merge)
            thread_id: Thread ID for versioning (default: SELECTIVE_THREAD_ID)

        Returns:
            Dict with modified_data, agent_response, error
        """
        # Validate at least 2 indices
        if len(selected_qa_indices) < 2:
            return {
                "modified_data": all_data,
                "agent_response": f"Error: Must select at least 2 Q&A pairs to merge (selected {len(selected_qa_indices)})",
                "error": "Invalid number of Q&A pairs",
                "version": 0
            }

        # Get previous state to determine version
        config = {"configurable": {"thread_id": thread_id}}

        try:
            prev_state = await self.merge_graph.aget_state(config)
            prev_version = prev_state.values.get("version", 0) if prev_state and prev_state.values else 0
        except:
            prev_version = 0

        logger.info("="*60)
        logger.info(f"🔀 MERGE Q&A: Starting 3-agent merge workflow (v{prev_version + 1})")
        logger.info(f"  Topic index: {topic_index}")
        logger.info(f"  Selected indices: {selected_qa_indices}")
        logger.info(f"  User request: {user_request[:50]}...")
        logger.info("="*60)

        # Create initial state for 3-agent workflow
        initial_state = {
            "user_request": user_request,
            "topic_index": topic_index,
            "selected_qa_indices": selected_qa_indices,
            "all_data": all_data,
            "extracted_topic": {},
            "extracted_qa_pairs": [],
            "extraction_summary": "",
            "merged_qa_pair": {},
            "merge_summary": "",
            "final_data": [],
            "agent_response": "",
            "version": prev_version + 1,
            "timestamp": "",
            "error": ""
        }

        try:
            # Run 3-agent graph: Agent1 → Merge Agent → Agent3
            logger.info("🔀 Invoking merge_graph.ainvoke()...")
            result = await self.merge_graph.ainvoke(initial_state, config)

            logger.info(f"✓ 3-agent merge workflow completed (v{prev_version + 1})")
            logger.info(f"  - Agent 1: {result.get('extraction_summary', 'N/A')}")
            logger.info(f"  - Merge Agent: {result.get('merge_summary', 'N/A')}")
            logger.info(f"  - Agent 3: {result.get('agent_response', 'N/A')}")

            return {
                "modified_data": result.get("final_data", all_data),
                "agent_response": result.get("agent_response", ""),
                "error": result.get("error", ""),
                "version": prev_version + 1
            }

        except Exception as e:
            logger.error(f"❌ 3-agent merge workflow failed: {e}")
            return {
                "modified_data": all_data,
                "agent_response": f"Merge failed: {str(e)}",
                "error": str(e),
                "version": prev_version + 1
            }

    async def get_current_state(self, thread_id: str = SELECTIVE_THREAD_ID) -> Optional[Dict]:
        """
        Get current state from LangGraph checkpoint

        Reads from SELECTIVE_THREAD_ID using selective_graph schema
        This supports both SelectiveModificationState and MergeQAState (both have 'final_data')
        """
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"  ⚠️  READING FROM CHECKPOINT: {thread_id}")
        try:
            state = await self.selective_graph.aget_state(config)
            if state and state.values:
                logger.info(f"  ✅ Found data in checkpoint (version: {state.values.get('version', 'unknown')})")
            else:
                logger.info(f"  ❌ No data found in checkpoint!")
            return state.values if state else None
        except Exception as e:
            logger.error(f"  ❌ Error reading checkpoint: {e}")
            return None

    async def get_history(self, thread_id: str = QA_THREAD_ID) -> List[Dict]:
        """Get checkpoint history (general modifier)"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            history = []
            async for state in self.general_graph.aget_state_history(config):
                if state.values:
                    history.append({
                        "version": state.values.get("version", 0),
                        "timestamp": state.values.get("timestamp", ""),
                        "user_request": state.values.get("user_request", ""),
                        "modification_type": state.values.get("modification_type", ""),
                        "agent_response": state.values.get("agent_response", "")
                    })
            return history
        except:
            return []

    async def rollback_to_version(self, version: int, thread_id: str = QA_THREAD_ID) -> Optional[List[Dict]]:
        """Rollback to specific version (general modifier)"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            # Find the checkpoint with matching version
            async for state in self.general_graph.aget_state_history(config):
                if state.values and state.values.get("version") == version:
                    # Update to this state
                    await self.general_graph.aupdate_state(config, state.values)
                    logger.info(f"✓ Rolled back to version {version}")
                    return state.values.get("modified_data", [])
            return None
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return None


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chatbot())

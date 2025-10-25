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
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Optional
from datetime import datetime
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

def build_chatbot_graph(checkpointer):
    """
    Build simple LangGraph workflow with checkpointer:

    START → llm_agent → END

    Args:
        checkpointer: AsyncSqliteSaver instance

    Returns:
        Compiled LangGraph with checkpointer
    """
    # Create graph builder
    builder = StateGraph(ChatbotState)

    # Add LLM agent node
    builder.add_node("llm_agent", llm_agent_node)

    # Define edges
    builder.add_edge(START, "llm_agent")
    builder.add_edge("llm_agent", END)

    # Compile graph with checkpointer
    graph = builder.compile(checkpointer=checkpointer)

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


def build_qa_modifier_graph(checkpointer):
    """
    Build Q&A Modifier graph with checkpointer

    Args:
        checkpointer: AsyncSqliteSaver instance

    Returns:
        Compiled LangGraph with checkpointer
    """
    builder = StateGraph(QAModifierState)
    builder.add_node("supervisor", supervisor_agent_node)
    builder.add_node("simplifier", simplifier_agent_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_to_agent, {"simplifier": "simplifier", "end": END})
    builder.add_edge("simplifier", END)

    logger.info("Q&A Modifier graph compiled with checkpointer")
    return builder.compile(checkpointer=checkpointer)


class QAModifier:
    """Q&A Modifier with Supervisor + Simplifier using LangGraph checkpoints"""

    def __init__(self, graph):
        """
        Initialize QA modifier with pre-built graph

        Args:
            graph: Compiled LangGraph with checkpointer
        """
        self.graph = graph
        logger.info("QAModifier initialized")

    async def modify(self, user_request: str, current_data: List[Dict], thread_id: str = QA_THREAD_ID) -> Dict:
        """
        Modify Q&A pairs and persist in LangGraph checkpoint

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
            prev_state = await self.graph.aget_state(config)
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

        # Run graph with checkpointer
        result = await self.graph.ainvoke(initial_state, config)

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
        thread_id: str = QA_THREAD_ID
    ) -> Dict:
        """
        Modify specific Q&A pairs within a single topic

        Args:
            user_request: User's modification request
            topic_index: Index of the topic to modify
            selected_qa_indices: Indices of Q&A pairs to modify
            selected_qa_pairs: The actual Q&A pairs selected for modification
            all_data: All current Q&A data
            thread_id: Thread ID for versioning

        Returns:
            Dict with modified_data, agent_response, error
        """
        # Get previous state to determine version
        config = {"configurable": {"thread_id": thread_id}}

        try:
            prev_state = await self.graph.aget_state(config)
            prev_version = prev_state.values.get("version", 0) if prev_state and prev_state.values else 0
        except:
            prev_version = 0

        try:
            # Initialize LLM
            llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                api_key=os.getenv('GROQ_API_KEY'),
                temperature=0.3,
                max_tokens=2000
            )

            # Format selected Q&A pairs for LLM (preserve original numbering)
            qa_text = "\n\n".join([f"Q{selected_qa_indices[i]+1}: {qa['question']}\nA{selected_qa_indices[i]+1}: {qa['answer']}"
                                   for i, qa in enumerate(selected_qa_pairs)])

            # Create prompt for selective modification
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

            # Apply selective modifications to the data
            modified_all_data = []
            for idx, topic_item in enumerate(all_data):
                if idx == topic_index:
                    # This is the topic to modify
                    modified_topic = topic_item.copy()
                    modified_qa_pairs = modified_topic.get("qa_pairs", []).copy()

                    # Replace selected Q&A pairs with modified versions
                    for selected_idx, modified_qa in zip(selected_qa_indices, modified_pairs):
                        if selected_idx < len(modified_qa_pairs):
                            modified_qa_pairs[selected_idx] = modified_qa

                    modified_topic["qa_pairs"] = modified_qa_pairs
                    modified_topic["qa_count"] = len(modified_qa_pairs)
                    modified_all_data.append(modified_topic)
                else:
                    # Keep other topics unchanged
                    modified_all_data.append(topic_item)

            topic_name = all_data[topic_index].get("topic", "Unknown Topic")
            agent_response = f"Modified {len(selected_qa_pairs)} Q&A pairs in '{topic_name}'"

            # Create state for checkpoint
            checkpoint_state = {
                "user_request": user_request,
                "current_data": all_data,
                "target_topic": topic_name,
                "modification_type": f"selective ({len(selected_qa_pairs)} Q&A pairs)",
                "modified_data": modified_all_data,
                "agent_response": agent_response,
                "error": "",
                "version": prev_version + 1,
                "timestamp": datetime.now().isoformat()
            }

            # Save to checkpoint
            await self.graph.aupdate_state(config, checkpoint_state)
            logger.info(f"✓ Created selective modification checkpoint version {prev_version + 1}")

            return {
                "modified_data": modified_all_data,
                "agent_response": agent_response,
                "error": "",
                "version": prev_version + 1
            }

        except Exception as e:
            logger.error(f"Selective modification failed: {e}")
            return {
                "modified_data": all_data,
                "agent_response": f"Modification failed: {str(e)}",
                "error": str(e),
                "version": prev_version + 1
            }

    async def get_current_state(self, thread_id: str = QA_THREAD_ID) -> Optional[Dict]:
        """Get current state from LangGraph checkpoint"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = await self.graph.aget_state(config)
            return state.values if state else None
        except:
            return None

    async def get_history(self, thread_id: str = QA_THREAD_ID) -> List[Dict]:
        """Get checkpoint history"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            history = []
            async for state in self.graph.aget_state_history(config):
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
        """Rollback to specific version"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            # Find the checkpoint with matching version
            async for state in self.graph.aget_state_history(config):
                if state.values and state.values.get("version") == version:
                    # Update to this state
                    await self.graph.aupdate_state(config, state.values)
                    logger.info(f"✓ Rolled back to version {version}")
                    return state.values.get("modified_data", [])
            return None
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return None


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chatbot())

#!/usr/bin/env python3
"""
Debug CLI for testing search algorithms and similarity scores.
Run: python menu.py <json_file>
"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple
from tabulate import tabulate

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from json_chatbot_engine import SimilaritySearchEngine, JSONChatbotEngine, QADataset


class SearchDebugger:
    """Debug wrapper around JSONQASimilaritySearch"""

    def __init__(self, json_path: str):
        self.json_path = json_path
        print(f"\n{'='*60}")
        print(f"Loading: {json_path}")
        print(f"{'='*60}\n")

        # Initialize search engine
        self.dataset = QADataset(json_path)
        self.search = SimilaritySearchEngine(self.dataset, json_path=json_path)
        self.chatbot = JSONChatbotEngine(json_path)

        # Try to load experimental algorithms
        try:
            from experimental_search import ExperimentalSearch
            self.experimental = ExperimentalSearch(self.search.dataset.all_qa_pairs)
            print("✅ Experimental algorithms loaded")
        except ImportError:
            self.experimental = None
            print("⚠️  experimental_search.py not found - experimental features disabled")

        print(f"\nLoaded {len(self.search.dataset.all_qa_pairs)} Q&A pairs")
        print(f"Thresholds:")
        print(f"  Semantic Ideal: {self.search.SIMILARITY_THRESHOLD_IDEAL}")
        print(f"  Semantic Minimal: {self.search.SIMILARITY_THRESHOLD}")
        print(f"  Answer Ideal: {self.search.ANSWER_THRESHOLD_IDEAL}")
        print(f"  Answer Minimal: {self.search.ANSWER_THRESHOLD_MINIMAL}")
        print(f"  Reranker Ideal: {self.search.RERANKER_THRESHOLD_IDEAL}")
        print(f"  Reranker Minimal: {self.search.RERANKER_THRESHOLD_MINIMAL}")

    def menu(self):
        """Main menu loop"""
        while True:
            print(f"\n{'='*60}")
            print("SEARCH DEBUG MENU")
            print(f"{'='*60}")
            print("\n[A] Search by phrase (show all algorithm scores)")
            print("[B] Compare phrase to specific answer (cosine similarity)")
            print("[C] Compare two phrases (semantic similarity)")
            print("[D] Raw embedding comparison (phrase vs all questions)")
            print("[E] BM25 keyword search only")
            print("[F] Test normalization (cofounder vs co-founder issue)")
            print("[G] Full pipeline trace (what chatbot does)")
            print("[H] List sample questions from data")
            print("[X] Experimental algorithms (if available)")
            print("[Q] Quit")

            choice = input("\nChoice: ").strip().upper()

            if choice == 'A':
                self.search_by_phrase()
            elif choice == 'B':
                self.compare_phrase_to_answer()
            elif choice == 'C':
                self.compare_two_phrases()
            elif choice == 'D':
                self.raw_embedding_search()
            elif choice == 'E':
                self.bm25_only()
            elif choice == 'F':
                self.test_normalization()
            elif choice == 'G':
                self.full_pipeline_trace()
            elif choice == 'H':
                self.list_sample_questions()
            elif choice == 'X':
                self.experimental_menu()
            elif choice == 'Q':
                print("Goodbye!")
                break
            else:
                print("Invalid choice")

    def search_by_phrase(self):
        """Option A: Search and show all algorithm scores"""
        query = input("\nEnter search phrase: ").strip()
        if not query:
            return

        print(f"\n{'='*60}")
        print(f"SEARCHING: '{query}'")
        print(f"{'='*60}")

        # 1. Pure semantic similarity
        print("\n[1] SEMANTIC SIMILARITY (Cosine)")
        semantic = self.search.similarity_search(query, top_k=5)
        self._print_results(semantic, "Semantic")

        # 2. Hybrid (BM25 + Semantic)
        print("\n[2] HYBRID SEARCH (BM25 + Semantic)")
        hybrid = self.search.hybrid_search(query, top_k=5)
        self._print_results(hybrid, "Hybrid")

        # 3. Cross-encoder reranking
        print("\n[3] CROSS-ENCODER RERANKING")
        if semantic.get('top_k_results'):
            reranked = self.search.rerank_with_cross_encoder(query, semantic['top_k_results'][:5])
            self._print_reranked(reranked)

        # 4. Answer-based search
        print("\n[4] ANSWER-BASED SEARCH")
        answer_search = self.search.similarity_search(query, search_answers=True, top_k=5)
        self._print_results(answer_search, "Answer-based")

        # 5. Topic search (if available)
        print("\n[5] TOPIC SEARCH")
        if hasattr(self.search, 'find_topic_for_question'):
            topic = self.search.find_topic_for_question(query)
            print(f"  Best topic: {topic.get('topic', 'None')}")
            print(f"  Score: {topic.get('score', 0):.4f}")
        else:
            print("  (not available in this engine)")

    def _print_results(self, result: Dict, label: str):
        """Print search results in table format"""
        # Extract data - handle different key names
        matches = result.get('candidates', result.get('top_k_results', []))
        scores = result.get('scores', [])
        best_score = result.get('score', 0)
        best_match = result.get('best_match')

        # Show top matches if available
        if matches:
            rows = []
            for i, item in enumerate(matches[:5]):
                q = item.question if hasattr(item, 'question') else str(item)
                q = q[:55] + ('...' if len(q) > 55 else '')
                s = scores[i] if i < len(scores) else best_score
                rows.append([i+1, f"{s:.4f}", q])
            print(tabulate(rows, headers=['Rank', 'Score', 'Question'], tablefmt='simple'))
            best_score = scores[0] if scores else best_score
        elif best_match:
            q = best_match.question if hasattr(best_match, 'question') else str(best_match)
            print(f"  → Best: [{best_score:.4f}] {q[:70]}")
        else:
            print(f"  No matches found")

        # Summary line
        ideal = self.search.SIMILARITY_THRESHOLD_IDEAL
        minimal = self.search.SIMILARITY_THRESHOLD
        status = "✅ IDEAL" if best_score >= ideal else ("⚠️ MINIMAL" if best_score >= minimal else "❌ BELOW")
        print(f"  {status} (score={best_score:.4f}, ideal>{ideal}, minimal>{minimal})")

    def _print_reranked(self, result: Dict):
        """Print reranked results"""
        candidates = result.get('ranked_candidates', [])
        best_score = result.get('score', 0)

        if candidates:
            rows = []
            for i, c in enumerate(candidates[:5]):
                q = c.question if hasattr(c, 'question') else c.get('question', str(c))
                q = q[:55] + ('...' if len(q) > 55 else '')
                s = c.get('reranker_score', 0) if isinstance(c, dict) else getattr(c, 'reranker_score', 0)
                rows.append([i+1, f"{s:.4f}", q])
            print(tabulate(rows, headers=['Rank', 'Reranker', 'Question'], tablefmt='simple'))
        else:
            print("  No candidates to rerank")

        ideal = self.search.RERANKER_THRESHOLD_IDEAL
        minimal = self.search.RERANKER_THRESHOLD_MINIMAL
        status = "✅ IDEAL" if best_score >= ideal else ("⚠️ MINIMAL" if best_score >= minimal else "❌ BELOW")
        print(f"  {status} (score={best_score:.4f}, ideal>{ideal}, minimal>{minimal})")

    def compare_phrase_to_answer(self):
        """Option B: Compare phrase to specific answer"""
        phrase = input("\nEnter phrase: ").strip()
        answer = input("Enter answer to compare: ").strip()
        if not phrase or not answer:
            return

        print(f"\n{'='*60}")
        print(f"COMPARING PHRASE TO ANSWER")
        print(f"{'='*60}")

        # Encode both
        phrase_emb = self.search.encoder.encode([phrase])[0]
        answer_emb = self.search.encoder.encode([answer])[0]

        # Cosine similarity
        cos_sim = np.dot(phrase_emb, answer_emb) / (np.linalg.norm(phrase_emb) * np.linalg.norm(answer_emb))

        print(f"\nPhrase: '{phrase}'")
        print(f"Answer: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
        print(f"\nCosine Similarity: {cos_sim:.4f}")
        print(f"Rating: {self._score_label(cos_sim)}")

    def compare_two_phrases(self):
        """Option C: Compare two phrases"""
        phrase1 = input("\nEnter first phrase: ").strip()
        phrase2 = input("Enter second phrase: ").strip()
        if not phrase1 or not phrase2:
            return

        print(f"\n{'='*60}")
        print(f"COMPARING TWO PHRASES")
        print(f"{'='*60}")

        # Semantic similarity
        emb1 = self.search.encoder.encode([phrase1])[0]
        emb2 = self.search.encoder.encode([phrase2])[0]
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        print(f"\nPhrase 1: '{phrase1}'")
        print(f"Phrase 2: '{phrase2}'")
        print(f"\nCosine Similarity: {cos_sim:.4f}")

        # Cross-encoder score (question-question similarity)
        if self.search.reranker:
            ce_score = self.search.reranker.predict([(phrase1, phrase2)])[0]
            print(f"Cross-Encoder Score: {ce_score:.4f}")

        print(f"\nRating: {self._score_label(cos_sim)}")

    def raw_embedding_search(self):
        """Option D: Raw embedding comparison"""
        query = input("\nEnter query: ").strip()
        if not query:
            return

        print(f"\n{'='*60}")
        print(f"RAW EMBEDDING SEARCH")
        print(f"{'='*60}")

        query_emb = self.search.encoder.encode([query])[0]

        # Compare to all questions
        scores = []
        for i, emb in enumerate(self.search.encoded_questions):
            cos_sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            scores.append((cos_sim, i))

        scores.sort(reverse=True)

        print(f"\nTop 10 matches by raw cosine similarity:")
        rows = []
        for score, idx in scores[:10]:
            q = self.search.dataset.all_qa_pairs[idx].question[:60]
            rows.append([f"{score:.4f}", q + ('...' if len(self.search.dataset.all_qa_pairs[idx].question) > 60 else '')])
        print(tabulate(rows, headers=['Score', 'Question'], tablefmt='simple'))

    def bm25_only(self):
        """Option E: BM25 keyword search only"""
        query = input("\nEnter query: ").strip()
        if not query:
            return

        print(f"\n{'='*60}")
        print(f"BM25 KEYWORD SEARCH")
        print(f"{'='*60}")

        # Tokenize query
        query_tokens = query.lower().split()
        print(f"Query tokens: {query_tokens}")

        # BM25 scores
        bm25_scores = self.search.bm25.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:10]

        print(f"\nTop 10 BM25 matches:")
        rows = []
        for idx in top_indices:
            score = bm25_scores[idx]
            q = self.search.dataset.all_qa_pairs[idx].question[:60]
            rows.append([f"{score:.4f}", q + ('...' if len(self.search.dataset.all_qa_pairs[idx].question) > 60 else '')])
        print(tabulate(rows, headers=['BM25 Score', 'Question'], tablefmt='simple'))

    def test_normalization(self):
        """Option F: Test text normalization issues"""
        print(f"\n{'='*60}")
        print(f"NORMALIZATION TEST")
        print(f"{'='*60}")

        # Test pairs that should be equivalent
        test_pairs = [
            ("cofounder", "co-founder"),
            ("CEO", "ceo"),
            ("who is the cofounder", "who is the co-founder"),
            ("healthcare", "health care"),
            ("email", "e-mail"),
        ]

        print("\nTesting phrase pairs that SHOULD be similar:")
        rows = []
        for p1, p2 in test_pairs:
            emb1 = self.search.encoder.encode([p1])[0]
            emb2 = self.search.encoder.encode([p2])[0]
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            # Cross-encoder
            ce_score = 0
            if self.search.reranker:
                ce_score = self.search.reranker.predict([(p1, p2)])[0]

            rows.append([p1, p2, f"{cos_sim:.4f}", f"{ce_score:.4f}"])

        print(tabulate(rows, headers=['Phrase 1', 'Phrase 2', 'Cosine Sim', 'Cross-Enc'], tablefmt='simple'))

        # Custom test
        print("\n" + "-"*40)
        custom = input("Enter custom phrase to test variations (or Enter to skip): ").strip()
        if custom:
            variations = [
                custom,
                custom.replace("-", ""),
                custom.replace("-", " "),
                custom.lower(),
                custom.upper(),
            ]

            print(f"\nSearching variations of '{custom}':")
            for var in variations:
                result = self.search.similarity_search(var, top_k=1)
                print(f"\n  '{var}':")
                print(f"    Best score: {result.get('score', 0):.4f}")
                if result.get('best_match'):
                    print(f"    Best match: {result['best_match'].question[:60]}")

    def full_pipeline_trace(self):
        """Option G: Full chatbot pipeline trace"""
        query = input("\nEnter query: ").strip()
        if not query:
            return

        print(f"\n{'='*60}")
        print(f"FULL PIPELINE TRACE")
        print(f"{'='*60}")

        # Run the full pipeline
        result = self.chatbot.answer(query)

        print(f"\nQuery: '{query}'")
        print(f"\n--- Pipeline Info ---")
        if result.get('pipeline_info'):
            info = result['pipeline_info']
            print(f"Stage: {info.get('stage', 'unknown')}")
            print(f"Matched by: {info.get('matched_by', 'unknown')}")
            print(f"Score: {info.get('score', 0):.4f}")
            print(f"Meets ideal: {info.get('meets_ideal', False)}")
            print(f"Meets minimal: {info.get('meets_minimal', False)}")

        print(f"\n--- Result ---")
        print(f"Question: {result.question[:80]}")
        print(f"Answer: {result.get('answer', '')[:200]}...")
        print(f"Topic: {result.get('topic', '')}")
        print(f"Source: {result.get('source', '')}")

    def list_sample_questions(self):
        """Option H: List sample questions"""
        print(f"\n{'='*60}")
        print(f"SAMPLE QUESTIONS FROM DATA")
        print(f"{'='*60}")

        # Show first 20 questions
        print("\nFirst 20 questions:")
        for i, qa in enumerate(self.search.dataset.all_qa_pairs[:20]):
            print(f"  {i+1}. {qa.question[:70]}")

        # Search for specific term
        term = input("\nSearch for questions containing (or Enter to skip): ").strip()
        if term:
            matches = [qa for qa in self.search.dataset.all_qa_pairs if term.lower() in qa.question.lower()]
            print(f"\nFound {len(matches)} questions containing '{term}':")
            for i, qa in enumerate(matches[:20]):
                print(f"  {i+1}. {qa.question[:70]}")

    def experimental_menu(self):
        """Option X: Experimental algorithms"""
        if not self.experimental:
            print("\n⚠️  experimental_search.py not found!")
            print("Create it with new algorithms to enable this menu.")
            return

        print(f"\n{'='*60}")
        print(f"EXPERIMENTAL ALGORITHMS")
        print(f"{'='*60}")

        self.experimental.menu(self.search)

    def _score_label(self, score: float) -> str:
        """Convert score to human label"""
        if score >= 0.8:
            return "★★★★★ Excellent"
        elif score >= 0.7:
            return "★★★★☆ Good"
        elif score >= 0.6:
            return "★★★☆☆ Fair"
        elif score >= 0.4:
            return "★★☆☆☆ Poor"
        else:
            return "★☆☆☆☆ Very Poor"


def main():
    if len(sys.argv) < 2:
        # Try to find a JSON file
        json_files = [f for f in os.listdir('.') if f.endswith('.json') and not f.endswith('_tree.json')]
        if json_files:
            print("Available JSON files:")
            for i, f in enumerate(json_files[:10]):
                print(f"  {i+1}. {f}")
            choice = input("\nSelect file number (or enter path): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(json_files):
                json_path = json_files[int(choice) - 1]
            else:
                json_path = choice
        else:
            print("Usage: python menu.py <json_file>")
            sys.exit(1)
    else:
        json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    debugger = SearchDebugger(json_path)
    debugger.menu()


if __name__ == "__main__":
    main()

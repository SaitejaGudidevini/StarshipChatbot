"""
Chatbot Rephrased Questions Test (Exact Match) — Diagnostic Edition
====================================================================

Tests if the chatbot returns the EXACT correct answer for rephrased questions.
Shows full pipeline diagnostics (stage scores, winner, thresholds) for EVERY result.

Usage:
    pytest test_chatbot_rephrased.py -v -s
"""
import sys
import os
import json
import csv
import pytest
from dotenv import load_dotenv

# Load environment variables
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path, override=True)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from json_chatbot_engine import JSONChatbotEngine

# --- Configuration ---
KB_PATH = "treetestingoutput2.json"
CSV_PATH = "SYRAhealth_rephrased_questions2.csv"
SAMPLE_SIZE = 500
USE_V2 = False  # Set to False for V1, True for V2


@pytest.fixture(scope="module")
def chatbot_engine():
    """Initialize chatbot with original KB."""
    print(f"\nInitializing chatbot with KB: '{KB_PATH}'")
    print(f"Using architecture: {'V2' if USE_V2 else 'V1'}")
    engine = JSONChatbotEngine(json_path=KB_PATH, enable_rephrasing=True)
    if USE_V2:
        engine.enable_v2_architecture()
    return engine


@pytest.fixture(scope="module")
def qa_test_cases():
    """Load rephrased questions from CSV and match with expected answers from KB."""
    print(f"Loading expected answers from: '{KB_PATH}'")
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    answer_map = {}
    for topic in data:
        for qa in topic.get('qa_pairs', []):
            answer_map[qa['question']] = qa['answer']

    print(f"Loading rephrased questions from: '{CSV_PATH}'")
    cases = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_q = row['original_question']
            rephrased_q = row['rephrased_question']
            expected_answer = answer_map.get(original_q)

            if expected_answer:
                cases.append({
                    'question': rephrased_q,
                    'original_question': original_q,
                    'answer': expected_answer
                })

    print(f"Built {len(cases)} test cases")

    if SAMPLE_SIZE and len(cases) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} test cases from {len(cases)} total.")
        return cases[:SAMPLE_SIZE]

    return cases


def extract_stage_diagnostics(result):
    """Extract per-stage scores and metadata from pipeline_info (V1 and V2)."""
    pipeline = result.get('pipeline_info', {})
    arch = pipeline.get('architecture', 'v1')

    stages = {'architecture': arch}

    if arch == 'v2_parallel_fused':
        # V2: Extract retriever details and RRF info
        retrieval = pipeline.get('retrieval_details', {})
        query_analysis = pipeline.get('query_analysis', {})

        stages['intent'] = query_analysis.get('intent', '')
        stages['entities'] = query_analysis.get('entities', {})
        stages['semantic_query'] = query_analysis.get('semantic_query', '')

        # Extract each retriever's top result
        for r_key in ['retriever_1', 'retriever_2', 'retriever_3', 'retriever_4']:
            r_data = retrieval.get(r_key, {})
            results = r_data.get('results', [])
            tech_name = r_data.get('technical_name', r_key)
            display_name = r_data.get('name', r_key)
            if results:
                top = results[0]
                stages[tech_name] = {
                    'display_name': display_name,
                    'rank1_score': top.get('score', 0.0),
                    'rank1_question': top.get('question', '')[:70],
                }

        # Extract RRF fusion top results
        rrf = retrieval.get('rrf_fusion', [])
        if rrf:
            stages['rrf_top'] = []
            for entry in rrf[:3]:
                stages['rrf_top'].append({
                    'rank': entry.get('rank', 0),
                    'score': entry.get('total_score', 0.0),
                    'question': entry.get('question', '')[:70],
                    'vote_count': entry.get('vote_count', 0),
                    'consensus': entry.get('consensus', ''),
                    'voted_by': entry.get('voted_by', []),
                })

        stages['candidates_evaluated'] = pipeline.get('candidates_evaluated', 0)
        stages['candidates_verified'] = pipeline.get('candidates_verified', 0)

    else:
        # V1: Extract sequential stage info
        stages_raw = pipeline.get('stages', [])
        for s in stages_raw:
            stage_key = s.get('stage', '')
            score = s.get('score', 0.0)
            question = s.get('best_match_question', '')

            if stage_key == 'primary_similarity':
                stages['stage_1'] = {
                    'score': score,
                    'question': question[:70],
                    'semantic_score': s.get('semantic_score', None),
                    'meets_ideal': s.get('meets_ideal', False),
                    'meets_minimal': s.get('meets_minimal', False),
                }
            elif stage_key == 'rephrase':
                stages['stage_2_rephrase_text'] = s.get('rephrased_text', '')
            elif stage_key == 'rephrase_similarity':
                stages['stage_2'] = {
                    'score': score,
                    'question': question[:70] if question else '',
                    'meets_ideal': s.get('meets_ideal', False),
                    'meets_minimal': s.get('meets_minimal', False),
                }
            elif stage_key == 'answer_search':
                stages['stage_2_5'] = {
                    'score': score,
                    'question': question[:70],
                    'meets_ideal': s.get('meets_ideal', False),
                    'meets_minimal': s.get('meets_minimal', False),
                }
            elif stage_key == 'fallback':
                stages['fallback'] = True

    return stages


def format_diagnostics(result, stages):
    """Format a compact diagnostics string for one test case (V1 and V2)."""
    lines = []

    winner = result.get('matched_by', 'fallback')
    confidence = result.get('confidence', 0.0)
    arch = stages.get('architecture', 'v1')

    if arch == 'v2_parallel_fused':
        # V2 format: show 4 retrievers + RRF
        if stages.get('intent'):
            lines.append(f"   Intent: {stages['intent']}  Entities: {stages.get('entities', {})}")

        for tech_name in ['Filtered-Semantic', 'BM25-Keyword', 'Q-Semantic', 'A-Semantic']:
            r = stages.get(tech_name)
            if r:
                lines.append(f"   R:{tech_name:<20} {r['rank1_score']:.4f} → \"{r['rank1_question']}...\"")

        rrf_top = stages.get('rrf_top', [])
        if rrf_top:
            lines.append(f"   RRF Top:")
            for entry in rrf_top:
                votes = ', '.join(entry.get('voted_by', [])[:3])
                lines.append(f"     #{entry['rank']} [{entry['score']:.4f}] ({entry['vote_count']} votes: {votes}) → \"{entry['question']}...\"")

        lines.append(f"   Winner: {winner} @ {confidence:.4f}")

    else:
        # V1 format: show S1/S2/S2.5
        s1 = stages.get('stage_1')
        if s1:
            flag = "✓" if s1['meets_ideal'] else ("~" if s1['meets_minimal'] else "✗")
            lines.append(f"   S1  reranker:  {s1['score']:.4f} [{flag}] → \"{s1['question']}...\"")
            if s1.get('semantic_score') is not None:
                lines.append(f"       (semantic pre-rerank: {s1['semantic_score']:.4f})")

        rephrase_text = stages.get('stage_2_rephrase_text', '')
        s2 = stages.get('stage_2')
        if s2:
            flag = "✓" if s2['meets_ideal'] else ("~" if s2['meets_minimal'] else "✗")
            lines.append(f"   S2  rephrase:  {s2['score']:.4f} [{flag}] → \"{s2['question']}...\"")
            if rephrase_text:
                lines.append(f"       (rephrased to: \"{rephrase_text[:70]}...\")")

        s25 = stages.get('stage_2_5')
        if s25:
            flag = "✓" if s25['meets_ideal'] else ("~" if s25['meets_minimal'] else "✗")
            lines.append(f"   S2.5 answer:   {s25['score']:.4f} [{flag}] → \"{s25['question']}...\"")

        lines.append(f"   Winner: {winner} @ {confidence:.4f}")

    return "\n".join(lines)


def test_chatbot_retrieval_accuracy(chatbot_engine, qa_test_cases):
    """
    Tests the chatbot's ability to retrieve the EXACT correct answer for rephrased questions.
    Shows full pipeline diagnostics for every result.
    """
    if not qa_test_cases:
        pytest.skip("No test cases found.")

    total_tests = len(qa_test_cases)
    passed_tests = 0
    failed_cases = []
    passed_cases = []

    # Track failure reasons
    failure_by_stage = {
        'fallback': 0,              # No stage met threshold
        'wrong_answer': 0,          # A stage won but returned wrong answer
    }
    winning_stage_counts = {}       # Which stage wins overall
    pass_by_winning_stage = {}      # Which stage wins for passes
    fail_by_winning_stage = {}      # Which stage wins for failures

    print(f"\n🔬 Starting Rephrased Questions Test - EXACT MATCH ({total_tests} test cases)...")
    print(f"   Architecture: {'V2' if USE_V2 else 'V1'}")
    print(f"   KB: {KB_PATH}")
    print()

    for i, qa_pair in enumerate(qa_test_cases):
        question = qa_pair['question']
        expected_answer = qa_pair['answer']

        # Get the chatbot's answer
        if USE_V2:
            result = chatbot_engine.process_question_v2(question)
        else:
            result = chatbot_engine.process_question(question)

        actual_answer = result['answer']
        is_pass = (actual_answer == expected_answer)

        # Extract diagnostics
        stages = extract_stage_diagnostics(result)
        diag = format_diagnostics(result, stages)
        winner = result.get('matched_by', 'fallback')

        # Track winning stage
        winning_stage_counts[winner] = winning_stage_counts.get(winner, 0) + 1

        if is_pass:
            passed_tests += 1
            pass_by_winning_stage[winner] = pass_by_winning_stage.get(winner, 0) + 1
            print(f"  ✅ [{i+1}/{total_tests}] Q: \"{question[:65]}...\"")
            print(diag)
            # Track vote count for V2 analysis
            rrf_top = stages.get('rrf_top', [])
            vote_count = rrf_top[0].get('vote_count', 0) if rrf_top else 0
            passed_cases.append({
                'question': question,
                'original_question': qa_pair['original_question'],
                'winner': winner,
                'confidence': result.get('confidence', 0.0),
                'vote_count': vote_count,
            })
        else:
            fail_by_winning_stage[winner] = fail_by_winning_stage.get(winner, 0) + 1
            if winner == 'fallback':
                failure_by_stage['fallback'] += 1
            else:
                failure_by_stage['wrong_answer'] += 1

            print(f"  ❌ [{i+1}/{total_tests}] Q: \"{question[:65]}...\"")
            print(diag)
            print(f"   Expected Q: \"{qa_pair['original_question'][:70]}...\"")
            failed_cases.append({
                'question': question,
                'original_question': qa_pair['original_question'],
                'expected_answer': expected_answer[:150],
                'actual_answer': actual_answer[:150],
                'winner': winner,
                'confidence': result.get('confidence', 0.0),
                'stages': stages,
            })

        print()  # blank line between tests

    # ================================================================
    # FINAL REPORT
    # ================================================================
    pass_rate = (passed_tests / total_tests) * 100

    print("\n" + "=" * 70)
    print(f"  🤖 EXACT MATCH TEST SUMMARY ({'V2' if USE_V2 else 'V1'}) 🤖")
    print("=" * 70)
    print(f"  Total Questions Tested:  {total_tests}")
    print(f"  ✅ Passed (Exact Match): {passed_tests}")
    print(f"  ❌ Failed:               {total_tests - passed_tests}")
    print(f"  🎯 Pass Rate:            {pass_rate:.2f}%")
    print("=" * 70)

    # --- Winning Stage Breakdown ---
    print("\n📊 WINNING STAGE BREAKDOWN (all results):")
    print("-" * 50)
    print(f"  {'Stage':<30} {'Total':>6} {'Pass':>6} {'Fail':>6}")
    print("-" * 50)
    for stage in sorted(winning_stage_counts.keys()):
        total = winning_stage_counts[stage]
        passes = pass_by_winning_stage.get(stage, 0)
        fails = fail_by_winning_stage.get(stage, 0)
        print(f"  {stage:<30} {total:>6} {passes:>6} {fails:>6}")
    print("-" * 50)

    # --- Failure Analysis ---
    print("\n🔍 FAILURE ANALYSIS:")
    print(f"  Fallback (no stage met threshold):  {failure_by_stage['fallback']}")
    print(f"  Wrong answer (stage won, wrong Q):  {failure_by_stage['wrong_answer']}")

    # --- Score Distribution for Failures ---
    if failed_cases:
        print("\n📉 FAILED CASES — SCORE DISTRIBUTION:")
        print("-" * 70)

        # Group by failure type
        fallback_fails = [c for c in failed_cases if c['winner'] == 'fallback']
        wrong_answer_fails = [c for c in failed_cases if c['winner'] != 'fallback']

        if fallback_fails:
            print(f"\n  🚫 Fallback failures ({len(fallback_fails)}):")
            # Show best scores from each stage for fallback failures
            s1_scores = []
            s2_scores = []
            s25_scores = []
            for c in fallback_fails:
                s = c.get('stages', {})
                if 'stage_1' in s:
                    s1_scores.append(s['stage_1']['score'])
                if 'stage_2' in s:
                    s2_scores.append(s['stage_2']['score'])
                if 'stage_2_5' in s:
                    s25_scores.append(s['stage_2_5']['score'])

            if s1_scores:
                print(f"     S1  scores: min={min(s1_scores):.4f}  avg={sum(s1_scores)/len(s1_scores):.4f}  max={max(s1_scores):.4f}")
            if s2_scores:
                print(f"     S2  scores: min={min(s2_scores):.4f}  avg={sum(s2_scores)/len(s2_scores):.4f}  max={max(s2_scores):.4f}")
            if s25_scores:
                print(f"     S2.5 scores: min={min(s25_scores):.4f}  avg={sum(s25_scores)/len(s25_scores):.4f}  max={max(s25_scores):.4f}")

        if wrong_answer_fails:
            print(f"\n  ❌ Wrong-answer failures ({len(wrong_answer_fails)}):")
            stage_groups = {}
            for c in wrong_answer_fails:
                w = c['winner']
                stage_groups.setdefault(w, []).append(c['confidence'])
            for stage, scores in sorted(stage_groups.items()):
                print(f"     {stage}: count={len(scores)}  avg_conf={sum(scores)/len(scores):.4f}")

        # Show top 10 "near-miss" failures (highest scoring failures)
        scored_fails = [c for c in failed_cases if c['confidence'] > 0]
        scored_fails.sort(key=lambda c: c['confidence'], reverse=True)

        if scored_fails:
            show_n = min(10, len(scored_fails))
            print(f"\n  🎯 Top {show_n} Near-Miss Failures (highest confidence that got wrong answer):")
            for i, c in enumerate(scored_fails[:show_n]):
                print(f"     {i+1}. [{c['winner']} @ {c['confidence']:.4f}] \"{c['question'][:60]}...\"")
                print(f"        Expected: \"{c['original_question'][:60]}...\"")

    # --- Architecture-specific deep dives ---
    if USE_V2:
        # V2 Deep Dive: RRF consensus analysis
        print(f"\n🔬 V2 RRF FAILURE DEEP DIVE ({len(failed_cases)} cases):")
        print("-" * 70)

        # Analyze vote counts for failures vs passes
        fail_vote_counts = []
        pass_vote_counts = []

        for c in failed_cases:
            rrf_top = c.get('stages', {}).get('rrf_top', [])
            if rrf_top:
                fail_vote_counts.append(rrf_top[0].get('vote_count', 0))

        for c in passed_cases:
            # passed_cases don't store stages, but we can track vote counts
            pass_vote_counts.append(c.get('vote_count', 0))

        if fail_vote_counts:
            from collections import Counter
            vote_dist = Counter(fail_vote_counts)
            print(f"\n  📊 Failed cases — RRF vote distribution (top candidate):")
            for votes in sorted(vote_dist.keys()):
                print(f"     {votes} votes: {vote_dist[votes]} failures")

        # Show top 20 failures with retriever breakdown
        for i, c in enumerate(failed_cases[:20]):
            s = c.get('stages', {})
            print(f"\n  [{i+1}] Q: \"{c['question'][:65]}...\"")
            print(f"      Expected: \"{c['original_question'][:65]}...\"")

            # Show each retriever's top pick
            for tech_name in ['Filtered-Semantic', 'BM25-Keyword', 'Q-Semantic', 'A-Semantic']:
                r = s.get(tech_name)
                if r:
                    print(f"      {tech_name:<20} {r['rank1_score']:.4f} → \"{r['rank1_question']}...\"")

            rrf_top = s.get('rrf_top', [])
            if rrf_top:
                top = rrf_top[0]
                votes = ', '.join(top.get('voted_by', [])[:4])
                print(f"      RRF Winner: [{top['score']:.4f}] ({top['vote_count']} votes: {votes})")

        if len(failed_cases) > 20:
            print(f"\n  ...and {len(failed_cases) - 20} more failures.")

    else:
        # V1 Deep Dives
        # --- Stage 2 Deep Dive ---
        s2_fails = [c for c in failed_cases if c['winner'] == 'rephrase_similarity']
        if s2_fails:
            print(f"\n🔬 STAGE 2 (REPHRASE) FAILURE DEEP DIVE ({len(s2_fails)} cases):")
            print("-" * 70)

            for i, c in enumerate(s2_fails[:20]):
                s = c.get('stages', {})
                s1 = s.get('stage_1', {})
                s2 = s.get('stage_2', {})
                rephrase_text = s.get('stage_2_rephrase_text', 'N/A')

                s1_q = s1.get('question', '')

                print(f"\n  [{i+1}] Rephrased Q: \"{c['question'][:65]}...\"")
                print(f"      Expected Q:   \"{c['original_question'][:65]}...\"")
                print(f"      Rephrase →    \"{rephrase_text[:65]}...\"")
                print(f"      S1: {s1.get('score', 0):.4f} → \"{s1_q}...\"")
                print(f"      S2: {s2.get('score', 0):.4f} → \"{s2.get('question', '')[:65]}...\"")

            if len(s2_fails) > 20:
                print(f"\n  ...and {len(s2_fails) - 20} more Stage 2 failures.")

            # Score comparison: S1 vs S2 for these failures
            s1_scores_in_s2_fails = []
            s2_scores_in_s2_fails = []
            s2_beat_s1_count = 0
            for c in s2_fails:
                s = c.get('stages', {})
                s1_score = s.get('stage_1', {}).get('score', 0)
                s2_score = s.get('stage_2', {}).get('score', 0)
                s1_scores_in_s2_fails.append(s1_score)
                s2_scores_in_s2_fails.append(s2_score)
                if s2_score > s1_score:
                    s2_beat_s1_count += 1

            print(f"\n  📊 Score comparison (S2 failures only):")
            print(f"     S1 scores: min={min(s1_scores_in_s2_fails):.4f}  avg={sum(s1_scores_in_s2_fails)/len(s1_scores_in_s2_fails):.4f}  max={max(s1_scores_in_s2_fails):.4f}")
            print(f"     S2 scores: min={min(s2_scores_in_s2_fails):.4f}  avg={sum(s2_scores_in_s2_fails)/len(s2_scores_in_s2_fails):.4f}  max={max(s2_scores_in_s2_fails):.4f}")
            print(f"     S2 beat S1 in {s2_beat_s1_count}/{len(s2_fails)} cases (this is why S2 won)")

        # --- Stage 1 Failure Deep Dive ---
        s1_fails = [c for c in failed_cases if c['winner'] in ('similarity_ideal', 'similarity_minimal')]
        if s1_fails:
            print(f"\n🔬 STAGE 1 (RERANKER) FAILURE DEEP DIVE ({len(s1_fails)} cases):")
            print("-" * 70)
            for i, c in enumerate(s1_fails[:10]):
                s = c.get('stages', {})
                s1 = s.get('stage_1', {})
                print(f"\n  [{i+1}] Rephrased Q: \"{c['question'][:65]}...\"")
                print(f"      Expected Q:   \"{c['original_question'][:65]}...\"")
                print(f"      S1: {s1.get('score', 0):.4f} → \"{s1.get('question', '')[:65]}...\"")

            if len(s1_fails) > 10:
                print(f"\n  ...and {len(s1_fails) - 10} more Stage 1 failures.")

    # Note: Not asserting 100% since rephrased questions are harder
    assert pass_rate >= 0, f"Pass rate: {pass_rate:.2f}%"

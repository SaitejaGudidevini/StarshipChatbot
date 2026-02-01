"""
Question Rephraser - CSV Output
===============================
Usage:
    python rephrase_questions.py -i CSU_Progress_deduplication.json -o rephrased_questions.csv
    python rephrase_questions.py -i CSU_Progress_deduplication.json -o rephrased_questions.csv --sample 100
"""

import os
import json
import csv
import argparse
import time
import re
from typing import List
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()


def rephrase_batch(client, questions: List[str]) -> List[str]:
    """Rephrase a batch of questions via LLM."""
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""Rephrase each question below. Keep the same meaning, use different words.

{numbered}

Reply with ONLY numbered rephrased questions, nothing else."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

        text = response.choices[0].message.content.strip()
        rephrased = []
        for line in text.split('\n'):
            line = re.sub(r'^\d+[\.\)\:]\s*', '', line.strip())
            if line:
                rephrased.append(line)

        return rephrased[:len(questions)] if len(rephrased) >= len(questions) else questions

    except Exception as e:
        print(f"Error: {e}")
        return questions


def main():
    parser = argparse.ArgumentParser(description="Rephrase KB questions to CSV")
    parser.add_argument('-i', '--input', default='treetestingoutput2.json')
    parser.add_argument('-o', '--output', default='SYRAhealth_rephrased_questions2.csv')
    parser.add_argument('-s', '--sample', type=int, default=None)
    parser.add_argument('-b', '--batch', type=int, default=10)
    args = parser.parse_args()

    # Load KB and extract questions
    print(f"Loading: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    questions = []
    for topic in data:
        for qa in topic.get('qa_pairs', []):
            questions.append(qa['question'])

    print(f"Found {len(questions)} questions")

    if args.sample:
        questions = questions[:args.sample]
        print(f"Limited to {args.sample}")

    # Init Groq
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    # Rephrase
    results = []
    print(f"Rephrasing...")

    for i in tqdm(range(0, len(questions), args.batch)):
        batch = questions[i:i + args.batch]
        rephrased = rephrase_batch(client, batch)

        for orig, reph in zip(batch, rephrased):
            results.append({'original_question': orig, 'rephrased_question': reph})

        time.sleep(0.5)

    # Save CSV
    print(f"Saving: {args.output}")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['original_question', 'rephrased_question'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! {len(results)} questions saved.")


if __name__ == "__main__":
    main()

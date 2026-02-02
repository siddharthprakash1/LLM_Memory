#!/usr/bin/env python3
"""
Run LongMemEval Benchmark with Visualization.

This script runs the LongMemEval benchmark on Memory V4 while streaming
results to the visualization dashboard in real-time.

Usage:
    # Start visualization server first:
    python benchmarks/longmemeval_viz.py
    
    # In another terminal, run the benchmark:
    python run_longmemeval.py --max-questions 5
    
    # Or test a specific question type:
    python run_longmemeval.py --types temporal-reasoning --max-questions 10
"""

import sys
import argparse
import requests
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.longmemeval_benchmark import LongMemEvalRunner


class LongMemEvalRunnerWithViz(LongMemEvalRunner):
    """
    Extended runner that sends updates to visualization server.
    """
    
    def __init__(self, viz_url: str = "http://localhost:5001", **kwargs):
        super().__init__(**kwargs)
        self.viz_url = viz_url
        self.viz_enabled = self._check_viz_server()
    
    def _check_viz_server(self) -> bool:
        """Check if visualization server is running."""
        try:
            response = requests.get(f"{self.viz_url}/api/status", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _notify_start(self, total_questions: int):
        """Notify viz server that benchmark is starting."""
        if not self.viz_enabled:
            return
        
        try:
            requests.post(
                f"{self.viz_url}/api/start",
                json={
                    'config': {
                        'model_name': self.model_name,
                        'ollama_url': self.ollama_url,
                        'memory_path': str(self.memory_path),
                    },
                    'total_questions': total_questions,
                },
                timeout=2
            )
        except:
            pass
    
    def _notify_progress(self, current: int, question_id: str, question_type: str, 
                         question_text: str = None, prediction: str = None, 
                         ground_truth: str = None, result=None, memory_stats=None):
        """Notify viz server of progress."""
        if not self.viz_enabled:
            return
        
        try:
            data = {
                'current': current,
                'question_id': question_id,
                'question_type': question_type,
                'question_text': question_text,
                'prediction': prediction,
                'ground_truth': ground_truth,
            }
            
            if result:
                # Convert result to dict
                data['result'] = {
                    'question_id': result.question_id,
                    'question_type': result.question_type,
                    'question': result.question,
                    'ground_truth': result.ground_truth,
                    'prediction': result.prediction,
                    'exact_match': result.exact_match,
                    'contains_match': result.contains_match,
                    'f1_score': result.f1_score,
                    'latency_ms': result.latency_ms,
                    'history_length': result.history_length,
                    'num_sessions': result.num_sessions,
                }
            
            if memory_stats:
                data['memory_stats'] = memory_stats
            
            requests.post(
                f"{self.viz_url}/api/update",
                json=data,
                timeout=2
            )
        except:
            pass
    
    def _notify_complete(self):
        """Notify viz server that benchmark is complete."""
        if not self.viz_enabled:
            return
        
        try:
            requests.post(
                f"{self.viz_url}/api/complete",
                json={},
                timeout=2
            )
        except:
            pass
    
    def _notify_error(self, error: str):
        """Notify viz server of error."""
        if not self.viz_enabled:
            return
        
        try:
            requests.post(
                f"{self.viz_url}/api/error",
                json={'error': error},
                timeout=2
            )
        except:
            pass
    
    def run(self, data_path: str):
        """Run benchmark with visualization."""
        print(f"\n{'#'*80}")
        print(f"# LongMemEval Benchmark - Memory V4")
        print(f"#")
        print(f"# Model: {self.model_name}")
        print(f"# Dataset: {data_path}")
        print(f"# Memory Path: {self.memory_path}")
        
        if self.viz_enabled:
            print(f"# Visualization: {self.viz_url} ✓")
        else:
            print(f"# Visualization: Not connected (start benchmarks/longmemeval_viz.py)")
        
        print(f"{'#'*80}\n")
        
        # Load dataset
        questions = self.load_dataset(data_path)
        print(f"Loaded {len(questions)} questions")
        
        # Notify start
        self._notify_start(len(questions))
        
        # Process each question
        for i, question_data in enumerate(questions):
            question_id = question_data['question_id']
            question_type = question_data['question_type']
            question_text = question_data['question']
            ground_truth = question_data['answer']
            
            print(f"\n[{i+1}/{len(questions)}] Processing {question_id} ({question_type})...")
            
            # Notify progress (start)
            self._notify_progress(
                i, question_id, question_type, 
                question_text=question_text,
                ground_truth=ground_truth
            )
            
            try:
                result = self.process_question(question_data)
                self.results.append(result)
                
                # Get memory stats from the last memory instance (if available)
                memory_stats = None
                # Note: In real implementation, you'd track the memory instance
                # For now, we'll skip memory stats to avoid complexity
                
                # Notify progress (with result)
                self._notify_progress(
                    i + 1, question_id, question_type,
                    question_text=question_text,
                    prediction=result.prediction,
                    ground_truth=ground_truth,
                    result=result,
                    memory_stats=memory_stats
                )
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                import traceback
                traceback.print_exc()
                
                # Add failed result
                failed_result = type('obj', (object,), {
                    'question_id': question_id,
                    'question_type': question_type,
                    'question': question_data['question'],
                    'ground_truth': question_data['answer'],
                    'prediction': "",
                    'exact_match': False,
                    'contains_match': False,
                    'f1_score': 0.0,
                    'latency_ms': 0.0,
                    'history_length': 0,
                    'num_sessions': 0,
                })()
                self.results.append(failed_result)
                
                self._notify_error(str(e))
        
        # Notify complete
        self._notify_complete()
        
        # Generate report
        report = self._generate_report()
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark with real-time visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 5 questions
  python run_longmemeval.py --max-questions 5
  
  # Test temporal reasoning questions only
  python run_longmemeval.py --types temporal-reasoning --max-questions 10
  
  # Full benchmark with custom model
  python run_longmemeval.py --model qwen2.5:32b
  
  # Start visualization first in another terminal:
  python benchmarks/longmemeval_viz.py
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json",
        help="Path to LongMemEval dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL"
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default="./longmemeval_memory",
        help="Path to store memory databases"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (for testing)"
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated list of question types to test (e.g., 'temporal-reasoning,multi-session')"
    )
    parser.add_argument(
        "--viz-url",
        type=str,
        default="http://localhost:5001",
        help="Visualization server URL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    # Parse question types
    question_types = None
    if args.types:
        question_types = [t.strip() for t in args.types.split(',')]
        print(f"Testing question types: {question_types}")
    
    # Create runner with visualization
    runner = LongMemEvalRunnerWithViz(
        model_name=args.model,
        ollama_url=args.ollama_url,
        memory_path=args.memory_path,
        max_questions=args.max_questions,
        question_types=question_types,
        viz_url=args.viz_url,
    )
    
    # Run benchmark
    try:
        report = runner.run(args.data)
        
        # Print and save report
        runner.print_report(report)
        report_path = runner.save_report(report, args.output_dir)
        
        print(f"\n{'='*80}")
        print("Benchmark completed successfully!")
        print(f"View results at: {args.viz_url}")
        print(f"Report saved to: {report_path}")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        if runner.results:
            report = runner._generate_report()
            runner.print_report(report)
            runner.save_report(report, args.output_dir)
    except Exception as e:
        print(f"\n\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

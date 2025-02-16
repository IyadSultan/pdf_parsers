import json
from collections import defaultdict, Counter
import statistics
from typing import Dict, List, Any

def analyze_test_results(file_path: str) -> Dict[str, Any]:
    """Analyze test results and evaluation metrics from the JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    analysis = {
        "overview": {},
        "metrics_summary": defaultdict(list),
        "detailed_analysis": defaultdict(list)
    }
    
    # Analyze each parser's results
    for parser_name, parser_data in data.items():
        if isinstance(parser_data, str) and "test_results=" in parser_data:
            try:
                # Extract metrics data
                test_results_str = parser_data.split("test_results=")[1]
                metrics = extract_metrics(test_results_str)
                
                # Organize metrics by type
                metrics_by_type = defaultdict(list)
                for metric in metrics:
                    metric_name = metric.get('name', 'unknown')
                    metrics_by_type[metric_name].append(metric)
                
                # Calculate summary statistics for this parser
                parser_summary = calculate_parser_metrics(metrics_by_type)
                analysis["overview"][parser_name] = parser_summary
                
                # Add to overall metrics summary
                for metric_name, metric_list in metrics_by_type.items():
                    analysis["metrics_summary"][metric_name].extend(metric_list)
                
                # Store detailed metrics
                analysis["detailed_analysis"][parser_name] = metrics
                
            except Exception as e:
                print(f"Error analyzing {parser_name}: {str(e)}")
    
    return analysis

def extract_metrics(text: str) -> List[Dict]:
    """Extract individual metric data blocks from text."""
    metrics = []
    current_pos = 0
    
    while True:
        start = text.find("MetricData(", current_pos)
        if start == -1:
            break
        end = find_matching_parenthesis(text, start)
        if end == -1:
            break
        metric_data = parse_metric_data(text[start:end+1])
        if metric_data:
            metrics.append(metric_data)
        current_pos = end + 1
    
    return metrics

def find_matching_parenthesis(text: str, start: int) -> int:
    """Find matching closing parenthesis."""
    count = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            count += 1
        elif text[i] == ')':
            count -= 1
            if count == 0:
                return i
    return -1

def parse_metric_data(metric_str: str) -> Dict:
    """Parse a MetricData string into a structured dictionary."""
    metric_dict = {}
    parts = metric_str.split(",")
    
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip().replace("MetricData(", "").strip()
            value = value.strip().strip("'")
            
            # Convert values to appropriate types
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.endswith('%'):
                try:
                    value = float(value.rstrip('%')) / 100
                except ValueError:
                    pass
            elif value.replace(".", "").isdigit():
                value = float(value)
            
            metric_dict[key] = value
    
    return metric_dict

def calculate_parser_metrics(metrics_by_type: Dict) -> Dict:
    """Calculate summary statistics for a parser's metrics."""
    summary = {}
    
    for metric_name, metrics in metrics_by_type.items():
        scores = [m.get('score', 0) for m in metrics]
        successes = [m.get('success', False) for m in metrics]
        
        summary[metric_name] = {
            "count": len(metrics),
            "success_rate": sum(successes) / len(successes) if successes else 0,
            "avg_score": statistics.mean(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "threshold": metrics[0].get('threshold', 0) if metrics else 0
        }
    
    return summary

def print_analysis(analysis: Dict):
    """Print formatted analysis results."""
    print("\n=== PARSER COMPARISON ===")
    print("-" * 80)
    
    # Compare parsers side by side
    metric_types = set()
    for parser_metrics in analysis["overview"].values():
        metric_types.update(parser_metrics.keys())
    
    for metric_type in sorted(metric_types):
        print(f"\n{metric_type}:")
        print("-" * 40)
        for parser_name, parser_metrics in analysis["overview"].items():
            if metric_type in parser_metrics:
                stats = parser_metrics[metric_type]
                print(f"\n{parser_name}:")
                print(f"  Success Rate: {stats['success_rate']:.1%}")
                print(f"  Average Score: {stats['avg_score']:.1%}")
                print(f"  Score Range: {stats['min_score']:.1%} - {stats['max_score']:.1%}")
                print(f"  Required Threshold: {stats['threshold']:.1%}")
                
                # Performance indicator
                if stats['avg_score'] >= stats['threshold']:
                    status = "✓ MEETS threshold"
                else:
                    gap = stats['threshold'] - stats['avg_score']
                    status = f"✗ BELOW threshold by {gap:.1%}"
                print(f"  Status: {status}")
    
    # Print overall statistics
    print("\n=== OVERALL STATISTICS ===")
    print("-" * 80)
    
    for parser_name, metrics in analysis["overview"].items():
        print(f"\n{parser_name}:")
        overall_success = sum(m['success_rate'] for m in metrics.values()) / len(metrics)
        overall_score = sum(m['avg_score'] for m in metrics.values()) / len(metrics)
        print(f"  Overall Success Rate: {overall_success:.1%}")
        print(f"  Overall Average Score: {overall_score:.1%}")

if __name__ == "__main__":
    file_path = r"C:\Users\isult\OneDrive\Documents\pdf_parsers\evaluation\evaluation_comparison_20250216_233536.json"
    results = analyze_test_results(file_path)
    print_analysis(results)
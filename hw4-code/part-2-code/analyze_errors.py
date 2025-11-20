"""
Error Analysis Script for Q6
Helps identify and categorize errors in generated SQL queries.
"""

import pickle
from collections import Counter, defaultdict


def load_files(nl_path, gt_sql_path, pred_sql_path, gt_records_path, pred_records_path):
    # Load all necessary files for error analysis
    # Load text files
    with open(nl_path, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    with open(gt_sql_path, 'r') as f:
        gt_sql = [line.strip() for line in f.readlines()]
    
    with open(pred_sql_path, 'r') as f:
        pred_sql = [line.strip() for line in f.readlines()]
    
    # Load record files
    with open(gt_records_path, 'rb') as f:
        gt_records, _ = pickle.load(f)
    
    with open(pred_records_path, 'rb') as f:
        pred_records, error_msgs = pickle.load(f)
    
    return nl_queries, gt_sql, pred_sql, gt_records, pred_records, error_msgs


def analyze_errors(nl_queries, gt_sql, pred_sql, gt_records, pred_records, error_msgs):
    # Perform comprehensive error analysis.
    
    # Returns:
        # Dictionary with error categories and examples

    error_analysis = {
        'syntax_errors': [],
        'semantic_errors': [],
        'missing_clauses': [],
        'wrong_tables': [],
        'wrong_aggregation': [],
        'partial_match': [],
        'correct': []
    }
    
    for i in range(len(nl_queries)):
        nl = nl_queries[i]
        gt = gt_sql[i]
        pred = pred_sql[i]
        gt_rec = set(gt_records[i])
        pred_rec = set(pred_records[i])
        error_msg = error_msgs[i]
        
        # Check if SQL is exactly correct
        if gt == pred:
            error_analysis['correct'].append({
                'idx': i,
                'nl': nl,
                'sql': pred
            })
            continue
        
        # Check for syntax errors (error message present)
        if error_msg:
            error_analysis['syntax_errors'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt,
                'pred_sql': pred,
                'error': error_msg
            })
            continue
        
        # Check for semantic errors (wrong results but valid SQL)
        if gt_rec != pred_rec:
            # Partial match (some overlap)
            overlap = len(gt_rec & pred_rec)
            if overlap > 0:
                error_analysis['partial_match'].append({
                    'idx': i,
                    'nl': nl,
                    'gt_sql': gt,
                    'pred_sql': pred,
                    'gt_count': len(gt_rec),
                    'pred_count': len(pred_rec),
                    'overlap': overlap
                })
            else:
                # Completely wrong
                error_analysis['semantic_errors'].append({
                    'idx': i,
                    'nl': nl,
                    'gt_sql': gt,
                    'pred_sql': pred,
                    'gt_count': len(gt_rec),
                    'pred_count': len(pred_rec)
                })
            
            # Check for specific error patterns
            gt_lower = gt.lower()
            pred_lower = pred.lower()
            
            # Missing WHERE clause
            if 'where' in gt_lower and 'where' not in pred_lower:
                error_analysis['missing_clauses'].append({
                    'idx': i,
                    'nl': nl,
                    'gt_sql': gt,
                    'pred_sql': pred,
                    'missing': 'WHERE'
                })
            
            # Missing GROUP BY
            elif 'group by' in gt_lower and 'group by' not in pred_lower:
                error_analysis['missing_clauses'].append({
                    'idx': i,
                    'nl': nl,
                    'gt_sql': gt,
                    'pred_sql': pred,
                    'missing': 'GROUP BY'
                })
            
            # Wrong aggregation function
            gt_aggs = set(['count', 'sum', 'avg', 'max', 'min']) & set(gt_lower.split())
            pred_aggs = set(['count', 'sum', 'avg', 'max', 'min']) & set(pred_lower.split())
            if gt_aggs != pred_aggs:
                error_analysis['wrong_aggregation'].append({
                    'idx': i,
                    'nl': nl,
                    'gt_sql': gt,
                    'pred_sql': pred,
                    'gt_agg': list(gt_aggs),
                    'pred_agg': list(pred_aggs)
                })
    
    return error_analysis


def print_error_summary(error_analysis):
    """Print a summary of errors."""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    
    total = sum(len(errors) for errors in error_analysis.values())
    
    print(f"\nTotal examples analyzed: {total}")
    print(f"Correct predictions: {len(error_analysis['correct'])} ({len(error_analysis['correct'])/total*100}%)")
    print()
    
    # Print error type statistics
    print("Error Type Distribution:")
    print("-" * 60)
    for error_type, errors in error_analysis.items():
        if error_type != 'correct' and len(errors) > 0:
            print(f"{error_type.replace('_', ' ').title()}: {len(errors)} ({len(errors)/total*100}%)")
    
    # Print detailed examples for each error type
    print("\n" + "=" * 60)
    print("DETAILED ERROR EXAMPLES")
    print("=" * 60)
    
    # Syntax errors
    if error_analysis['syntax_errors']:
        print("\n1. SYNTAX ERRORS (SQL doesn't execute)")
        print("-" * 60)
        for i, err in enumerate(error_analysis['syntax_errors'][:3]):  # Show first 3
            print(f"\nExample {i+1}:")
            print(f"NL Query: {err['nl']}")
            print(f"Ground Truth: {err['gt_sql']}")
            print(f"Predicted: {err['pred_sql']}")
            print(f"Error: {err['error']}")
    
    # Semantic errors
    if error_analysis['semantic_errors']:
        print("\n2. SEMANTIC ERRORS (Wrong results)")
        print("-" * 60)
        for i, err in enumerate(error_analysis['semantic_errors'][:3]):
            print(f"\nExample {i+1}:")
            print(f"NL Query: {err['nl']}")
            print(f"Ground Truth: {err['gt_sql']}")
            print(f"Predicted: {err['pred_sql']}")
            print(f"GT returned {err['gt_count']} records, Pred returned {err['pred_count']} records")
    
    # Missing clauses
    if error_analysis['missing_clauses']:
        print("\n3. MISSING CLAUSES")
        print("-" * 60)
        for i, err in enumerate(error_analysis['missing_clauses'][:3]):
            print(f"\nExample {i+1}:")
            print(f"NL Query: {err['nl']}")
            print(f"Ground Truth: {err['gt_sql']}")
            print(f"Predicted: {err['pred_sql']}")
            print(f"Missing: {err['missing']} clause")
    
    # Wrong aggregation
    if error_analysis['wrong_aggregation']:
        print("\n4. WRONG AGGREGATION FUNCTIONS")
        print("-" * 60)
        for i, err in enumerate(error_analysis['wrong_aggregation'][:3]):
            print(f"\nExample {i+1}:")
            print(f"NL Query: {err['nl']}")
            print(f"Ground Truth: {err['gt_sql']}")
            print(f"Predicted: {err['pred_sql']}")
            print(f"GT uses: {err['gt_agg']}, Pred uses: {err['pred_agg']}")


def generate_latex_table(error_analysis):
    """Generate LaTeX table for Table 5 in the report."""
    print("\n" + "=" * 60)
    print("LATEX TABLE 5 (Copy this to your report)")
    print("=" * 0)
    
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{p{2cm}p{4cm}p{5cm}p{2cm}}")
    print("\\hline")
    print("Error Type & Example & Error Description & Statistics \\\\")
    print("\\hline")
    
    # Syntax errors
    if error_analysis['syntax_errors']:
        err = error_analysis['syntax_errors'][0]
        print(f"Syntax Error & ")
        print(f"NL: \\textit{{{err['nl'][:50]}...}} ")
        print(f"Pred: \\texttt{{{err['pred_sql'][:40]}...}} & ")
        print(f"Generated SQL contains syntax errors that prevent execution & ")
        print(f"{len(error_analysis['syntax_errors'])}/{sum(len(v) for v in error_analysis.values())} \\\\")
        print("\\hline")
    
    # Semantic errors
    if error_analysis['semantic_errors']:
        err = error_analysis['semantic_errors'][0]
        print(f"Semantic Error & ")
        print(f"NL: \\textit{{{err['nl'][:50]}...}} ")
        print(f"Pred: \\texttt{{{err['pred_sql'][:40]}...}} & ")
        print(f"SQL executes but returns wrong records. GT: {err['gt_count']} records, Pred: {err['pred_count']} records & ")
        print(f"{len(error_analysis['semantic_errors'])}/{sum(len(v) for v in error_analysis.values())} \\\\")
        print("\\hline")
    
    # Missing clauses
    if error_analysis['missing_clauses']:
        err = error_analysis['missing_clauses'][0]
        print(f"Missing Clause & ")
        print(f"NL: \\textit{{{err['nl'][:50]}...}} ")
        print(f"Pred: \\texttt{{{err['pred_sql'][:40]}...}} & ")
        print(f"Query missing {err['missing']} clause, leading to overly broad results & ")
        print(f"{len(error_analysis['missing_clauses'])}/{sum(len(v) for v in error_analysis.values())} \\\\")
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\caption{Error analysis on development set.}")
    print("\\label{tab:error_analysis}")
    print("\\end{table}")


def main():
    # Run error analysis on dev set

    # File paths
    nl_path = 'data/dev.nl'
    gt_sql_path = 'data/dev.sql'
    pred_sql_path = 'results/t5_ft_ft_experiment_dev.sql'  # Adjust name as needed
    gt_records_path = 'records/ground_truth_dev.pkl'
    pred_records_path = 'records/t5_ft_ft_experiment_dev.pkl'  # Adjust name as needed
    
    print("Loading files...")
    nl_queries, gt_sql, pred_sql, gt_records, pred_records, error_msgs = load_files(
        nl_path, gt_sql_path, pred_sql_path, gt_records_path, pred_records_path
    )
    
    print("Analyzing errors...")
    error_analysis = analyze_errors(
        nl_queries, gt_sql, pred_sql, gt_records, pred_records, error_msgs
    )
    
    # Print summary
    print_error_summary(error_analysis)
    
    # Generate LaTeX table
    generate_latex_table(error_analysis)
    
    # Save detailed analysis to file
    output_file = 'error_analysis_detailed.txt'
    with open(output_file, 'w') as f:
        f.write("DETAILED ERROR ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        for error_type, errors in error_analysis.items():
            if error_type != 'correct':
                f.write(f"\n{error_type.upper()}\n")
                f.write("-" * 60 + "\n")
                for err in errors[:10]:  # Save first 10 of each type
                    f.write(f"\nIndex: {err['idx']}\n")
                    f.write(f"NL: {err['nl']}\n")
                    if 'gt_sql' in err:
                        f.write(f"GT SQL: {err['gt_sql']}\n")
                        f.write(f"Pred SQL: {err['pred_sql']}\n")
                    if 'error' in err:
                        f.write(f"Error: {err['error']}\n")
                    f.write("\n")
    
    print(f"\nDetailed analysis saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
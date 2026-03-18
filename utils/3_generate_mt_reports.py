"""
generate_mt_reports.py
Generate comprehensive reports for MT evaluation metrics: BLEU, chrF, COMET, and Average
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import re
from collections import defaultdict
from tqdm import tqdm

# ------------------------------------------------------------------
# Configure PNG export settings
# ------------------------------------------------------------------
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800
pio.templates.default = "plotly_white"

# Load language mapping from CSV
def load_language_mapping():
    """Load language mapping from CSV file"""
    mapping_path = os.path.join("language_mapping.csv")
    language_mapping = {}
    
    try:
        if os.path.exists(mapping_path):
            df = pd.read_csv(mapping_path)
            for _, row in df.iterrows():
                language_mapping[row['language_code']] = row['language_name']
            print(f"Loaded language mapping for {len(language_mapping)} languages")
        else:
            print(f"Warning: Language mapping file not found at {mapping_path}")
    except Exception as e:
        print(f"Error loading language mapping: {e}")
    
    return language_mapping

LANGUAGE_MAPPING = load_language_mapping()

def get_language_name(lang_code):
    """Get language name from code"""
    return LANGUAGE_MAPPING.get(lang_code, lang_code)

def get_language_display_name(language_pair):
    """Convert language pair code to display name using SOURCE language"""
    if '-' not in language_pair:
        return get_language_name(language_pair)
    
    source_lang, target_lang = language_pair.split('-', 1)
    source_name = get_language_name(source_lang)
    
    return source_name if source_name != source_lang else language_pair

def get_available_recipes(recipes_dir="/home/owusus/Documents/GitHub/nsanku/recipes"):
    """Get list of available recipe names"""
    recipes = []
    if os.path.exists(recipes_dir):
        for file in os.listdir(recipes_dir):
            if file.endswith(".py") and file != "__init__.py":
                recipes.append(file[:-3])
    return recipes

def extract_recipe_name_from_filename(filename, available_recipes):
    """Extract recipe/model name from filename"""
    name_without_ext = os.path.splitext(filename)[0]
    for recipe in available_recipes:
        if f"_{recipe}" in name_without_ext:
            return recipe
    match = re.search(r'_([^_]+)$', name_without_ext)
    if match:
        return match.group(1)
    return "unknown_recipe"

def combine_all_datasets(input_dir="/home/owusus/Documents/GitHub/nsanku/output_combined"):
    """
    Combine all CSV files from all directories into one main dataset
    """
    all_data = []
    recipes = get_available_recipes()
    
    print("Combining all datasets...")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            
            src, tgt = folder.split("-", 1)
            recipe = extract_recipe_name_from_filename(file, recipes)
            
            try:
                df = pd.read_csv(os.path.join(root, file))
                
                # Check if we have the required metric columns
                has_metrics = any(col in df.columns for col in ['bleu_score', 'chrf_score', 'comet_score', 'avg_score'])
                if not has_metrics:
                    continue
                
                # Add metadata columns
                df['language_pair'] = f"{src}-{tgt}"
                df['source_lang'] = src
                df['target_lang'] = tgt
                df['model'] = recipe
                df['file_path'] = os.path.join(root, file)
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
    if not all_data:
        print("No data files found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Language pairs: {combined_df['language_pair'].nunique()}")
    print(f"Models: {combined_df['model'].nunique()}")
    
    return combined_df

def create_horizontal_bar_chart(data_dict, title, x_label, filename, output_dir):
    """Create horizontal bar chart for model/language comparison"""
    if not data_dict:
        return
    
    # Sort by score (highest first)
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=x_label)
        ),
        text=[f'{v:.2f}' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='',
        height=max(400, len(labels) * 30),
        showlegend=False,
        font=dict(size=12)
    )
    
    # Save as HTML and PNG
    html_path = os.path.join(output_dir, f'{filename}.html')
    png_path = os.path.join(output_dir, f'{filename}.png')
    
    fig.write_html(html_path)
    fig.write_image(png_path)

def create_metric_comparison_chart(model_metrics, title, filename, output_dir):
    """
    Create grouped bar chart comparing different metrics for each model
    model_metrics: dict of {model_name: {metric_name: score}}
    """
    if not model_metrics:
        return
    
    models = list(model_metrics.keys())
    metrics = ['BLEU', 'chrF', 'Average']
    
    fig = go.Figure()
    
    colors = {
        'BLEU': '#1f77b4',
        'chrF': '#ff7f0e',
        'Average': '#d62728'
    }
    
    for metric in metrics:
        values = [model_metrics[model].get(metric, 0) for model in models]
        
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            marker_color=colors.get(metric, '#333333'),
            text=[f'{v:.1f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=600,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save as HTML and PNG
    html_path = os.path.join(output_dir, f'{filename}.html')
    png_path = os.path.join(output_dir, f'{filename}.png')
    
    fig.write_html(html_path)
    fig.write_image(png_path)

def collect_results(input_dir="/home/owusus/Documents/GitHub/nsanku/output_combined"):
    """
    Collect results for all models and language pairs
    IMPORTANT: For BLEU, we recalculate at corpus level across all files
    for each model+language pair combination
    """
    import sacrebleu
    
    # Storage for raw translation data (for corpus-level BLEU recalculation)
    translation_data = defaultdict(lambda: {'hypotheses': [], 'references': []})
    
    # Storage for chrF scores (can be averaged)
    chrf_results = {}
    
    print("Collecting results from processed data...")
    
    recipes = get_available_recipes()
    
    # First pass: collect all translations for corpus-level BLEU
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            
            src, tgt = folder.split("-", 1)
            language_pair = f"{src}-{tgt}"
            recipe = extract_recipe_name_from_filename(file, recipes)
            
            try:
                df = pd.read_csv(os.path.join(root, file))
                
                # Collect translations for corpus-level BLEU
                if 'translated' in df.columns and 'ref' in df.columns:
                    valid_mask = (
                        df['translated'].notna() & 
                        df['ref'].notna() & 
                        (df['translated'] != 'nan') & 
                        (df['ref'] != 'nan') &
                        (df['translated'].astype(str).str.strip() != '') &
                        (df['ref'].astype(str).str.strip() != '')
                    )
                    
                    if valid_mask.any():
                        key = (language_pair, recipe)
                        translation_data[key]['hypotheses'].extend(
                            df.loc[valid_mask, 'translated'].astype(str).str.strip().tolist()
                        )
                        translation_data[key]['references'].extend(
                            df.loc[valid_mask, 'ref'].astype(str).str.strip().tolist()
                        )
                
                # Collect chrF scores for averaging
                if 'chrf_score' in df.columns:
                    if language_pair not in chrf_results:
                        chrf_results[language_pair] = {}
                    
                    # Average chrF for this file
                    chrf_avg = df['chrf_score'].mean()
                    if recipe in chrf_results[language_pair]:
                        # If we have multiple files, collect them for averaging
                        if not isinstance(chrf_results[language_pair][recipe], list):
                            chrf_results[language_pair][recipe] = [chrf_results[language_pair][recipe]]
                        chrf_results[language_pair][recipe].append(chrf_avg)
                    else:
                        chrf_results[language_pair][recipe] = chrf_avg
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    # Second pass: calculate corpus-level BLEU for each model+language pair
    print("\nCalculating corpus-level BLEU scores for each model+language pair...")
    bleu_results = {}
    
    for (language_pair, recipe), data in tqdm(translation_data.items(), desc="Computing BLEU"):
        hypotheses = data['hypotheses']
        references = data['references']
        
        if hypotheses and references:
            try:
                refs = [[ref] for ref in references]
                bleu = sacrebleu.corpus_bleu(hypotheses, refs)
                
                if language_pair not in bleu_results:
                    bleu_results[language_pair] = {}
                bleu_results[language_pair][recipe] = bleu.score
                
            except Exception as e:
                print(f"  Error calculating BLEU for {language_pair}/{recipe}: {e}")
    
    # Average chrF scores if we have multiple files per model
    for lang_pair in chrf_results:
        for recipe in chrf_results[lang_pair]:
            if isinstance(chrf_results[lang_pair][recipe], list):
                chrf_results[lang_pair][recipe] = np.mean(chrf_results[lang_pair][recipe])
    
    # Combine results
    results = {
        'bleu': bleu_results,
        'chrf': chrf_results,
        'average': {}
    }
    
    # Calculate averages
    for language_pair in bleu_results:
        results['average'][language_pair] = {}
        for recipe in bleu_results[language_pair]:
            bleu_score = bleu_results[language_pair].get(recipe, 0)
            chrf_score = chrf_results.get(language_pair, {}).get(recipe, 0)
            results['average'][language_pair][recipe] = (bleu_score + chrf_score) / 2
    
    # Note: source_breakdown is not implemented in this version
    # since we're doing corpus-level calculation
    source_breakdown = {}
    
    print(f"\n✓ Calculated BLEU for {len(translation_data)} model+language combinations")
    
    return results, source_breakdown

def generate_language_specific_reports(results, source_breakdown, 
                                       output_dir="/home/owusus/Documents/GitHub/nsanku/reports_combined"):
    """
    Generate individual reports for each language pair with all metrics
    """
    print("Generating language-specific reports...")
    
    # Get all language pairs (from average results)
    language_pairs = results['average'].keys()
    
    for language_pair in language_pairs:
        print(f"Processing {language_pair}...")
        
        lang_output_dir = os.path.join(output_dir, language_pair)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        # Collect all metrics for this language pair
        model_metrics = {}
        
        for model in results['average'][language_pair].keys():
            model_metrics[model] = {
                'BLEU': results['bleu'][language_pair].get(model, 0),
                'chrF': results['chrf'][language_pair].get(model, 0),
                'Average': results['average'][language_pair].get(model, 0)
            }
        
        if not model_metrics:
            print(f"No model results found for {language_pair}")
            continue
        
        # Generate metric comparison chart
        create_metric_comparison_chart(
            model_metrics,
            f'MT Metrics Comparison for {language_pair}',
            'metrics_comparison',
            lang_output_dir
        )
        
        # Generate individual metric charts
        for metric_name, metric_key in [('BLEU', 'bleu'), ('chrF', 'chrf'), ('Average', 'average')]:
            if language_pair in results[metric_key]:
                create_horizontal_bar_chart(
                    results[metric_key][language_pair],
                    f'{metric_name} Scores for {language_pair}',
                    f'{metric_name} Score',
                    f'{metric_key}_scores',
                    lang_output_dir
                )
        
        # Generate CSV report
        report_data = []
        for model, metrics in sorted(model_metrics.items(), 
                                    key=lambda x: x[1]['Average'], reverse=True):
            report_data.append({
                'Model': model,
                'BLEU': f"{metrics['BLEU']:.2f}",
                'chrF': f"{metrics['chrF']:.2f}",
                'Average': f"{metrics['Average']:.2f}",
                'BLEU_raw': metrics['BLEU'],
                'chrF_raw': metrics['chrF'],
                'Average_raw': metrics['Average']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(os.path.join(lang_output_dir, 'detailed_report.csv'), index=False)
        
        # Generate summary
        best_model_by_avg = max(model_metrics.items(), key=lambda x: x[1]['Average'])
        
        summary = {
            'language_pair': language_pair,
            'timestamp': datetime.now().isoformat(),
            'models': {
                'bleu': results['bleu'][language_pair],
                'chrf': results['chrf'][language_pair],
                'average': results['average'][language_pair]
            },
            'best_model': best_model_by_avg[0],
            'best_model_scores': best_model_by_avg[1]
        }
        
        with open(os.path.join(lang_output_dir, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Text summary
        with open(os.path.join(lang_output_dir, 'summary.txt'), 'w') as f:
            f.write(f"MT Evaluation Results for {language_pair}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nModel Performance (sorted by Average score):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Model':<30} {'BLEU':>10} {'chrF':>10} {'Average':>10}\n")
            f.write("-" * 70 + "\n")
            for model, metrics in sorted(model_metrics.items(), 
                                        key=lambda x: x[1]['Average'], reverse=True):
                f.write(f"{model:<30} {metrics['BLEU']:>10.2f} {metrics['chrF']:>10.2f} "
                       f"{metrics['Average']:>10.2f}\n")
            f.write("-" * 70 + "\n")
            f.write(f"\nBest Model (by Average): {best_model_by_avg[0]}\n")
            f.write(f"  BLEU: {best_model_by_avg[1]['BLEU']:.2f}\n")
            f.write(f"  chrF: {best_model_by_avg[1]['chrF']:.2f}\n")
            f.write(f"  Average: {best_model_by_avg[1]['Average']:.2f}\n")

def generate_overall_summary(results, source_breakdown, 
                            output_dir="/home/owusus/Documents/GitHub/nsanku/reports_combined"):
    """Generate overall summary across all language pairs and metrics"""
    if not results['average']:
        return
    
    print("Generating overall summary...")
    
    # Calculate overall model performance (average across all language pairs)
    all_models = set()
    for lang_results in results['average'].values():
        all_models.update(lang_results.keys())
    
    model_performance = {
        'bleu': {},
        'chrf': {},
        'average': {}
    }
    
    for model in all_models:
        for metric_key in ['bleu', 'chrf', 'average']:
            scores = []
            for lang_results in results[metric_key].values():
                if model in lang_results:
                    scores.append(lang_results[model])
            if scores:
                model_performance[metric_key][model] = np.mean(scores)
    
    # Generate overall model performance charts
    for metric_name, metric_key in [('BLEU', 'bleu'), ('chrF', 'chrf'), ('Average', 'average')]:
        if model_performance[metric_key]:
            create_horizontal_bar_chart(
                model_performance[metric_key],
                f'Overall {metric_name} Performance Across All Languages',
                f'{metric_name} Score',
                f'overall_{metric_key}',
                output_dir
            )
    
    # Generate combined metrics chart for all models
    model_metrics_overall = {}
    for model in all_models:
        model_metrics_overall[model] = {
            'BLEU': model_performance['bleu'].get(model, 0),
            'chrF': model_performance['chrf'].get(model, 0),
            'Average': model_performance['average'].get(model, 0)
        }
    
    create_metric_comparison_chart(
        model_metrics_overall,
        'Overall MT Metrics Comparison Across All Languages',
        'overall_metrics_comparison',
        output_dir
    )
    
    # Calculate language performance
    language_performance = {}
    for language_pair in results['average'].keys():
        scores = list(results['average'][language_pair].values())
        if scores:
            display_name = get_language_display_name(language_pair)
            language_performance[display_name] = np.mean(scores)
    
    create_horizontal_bar_chart(
        language_performance,
        'Language Performance Across Models (Average Score)',
        'Average Score',
        'language_performance',
        output_dir
    )
    
    # Save overall summary
    best_model = max(model_performance['average'].items(), 
                    key=lambda x: x[1]) if model_performance['average'] else ("none", 0)
    best_language = max(language_performance.items(), 
                       key=lambda x: x[1]) if language_performance else ("none", 0)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_language_pairs': len(results['average']),
        'total_models': len(all_models),
        'model_performance': model_performance,
        'language_performance': language_performance,
        'best_overall_model': best_model[0],
        'best_overall_scores': {
            'BLEU': model_performance['bleu'].get(best_model[0], 0),
            'chrF': model_performance['chrf'].get(best_model[0], 0),
            'Average': best_model[1]
        },
        'best_language': best_language[0],
        'best_language_score': best_language[1]
    }
    
    with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create text summary
    with open(os.path.join(output_dir, 'overall_summary.txt'), 'w') as f:
        f.write("OVERALL MT EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Language Pairs: {summary['total_language_pairs']}\n")
        f.write(f"Total Models: {summary['total_models']}\n\n")
        
        f.write("BEST OVERALL MODEL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model: {summary['best_overall_model']}\n")
        f.write(f"  BLEU: {summary['best_overall_scores']['BLEU']:.2f}\n")
        f.write(f"  chrF: {summary['best_overall_scores']['chrF']:.2f}\n")
        f.write(f"  Average: {summary['best_overall_scores']['Average']:.2f}\n\n")
        
        f.write("MODEL RANKINGS (by Average Score)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6} {'Model':<30} {'BLEU':>10} {'chrF':>10} {'Average':>10}\n")
        f.write("-" * 70 + "\n")
        
        sorted_models = sorted(model_metrics_overall.items(), 
                             key=lambda x: x[1]['Average'], reverse=True)
        for rank, (model, metrics) in enumerate(sorted_models, 1):
            f.write(f"{rank:<6} {model:<30} {metrics['BLEU']:>10.2f} {metrics['chrF']:>10.2f} "
                   f"{metrics['Average']:>10.2f}\n")
    
    return summary

def generate_report(input_dir="/home/owusus/Documents/GitHub/nsanku/output_combined", 
                   output_dir="/home/owusus/Documents/GitHub/nsanku/reports_combined"):
    """Main function to generate all MT evaluation reports"""
    print("="*70)
    print("GENERATING MT EVALUATION REPORTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results
    results, source_breakdown = collect_results(input_dir)
    
    if not results['average']:
        print("No processed results found. Please run metric calculations first.")
        return
    
    # Generate reports
    generate_language_specific_reports(results, source_breakdown, output_dir)
    overall_summary = generate_overall_summary(results, source_breakdown, output_dir)
    
    print(f"\n{'='*70}")
    print("REPORTS GENERATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    if overall_summary:
        print(f"\nBest Overall Model: {overall_summary['best_overall_model']}")
        print(f"  BLEU: {overall_summary['best_overall_scores']['BLEU']:.2f}")
        print(f"  chrF: {overall_summary['best_overall_scores']['chrF']:.2f}")
        print(f"  Average: {overall_summary['best_overall_scores']['Average']:.2f}")
        print(f"\nBest Language: {overall_summary['best_language']} "
              f"({overall_summary['best_language_score']:.2f})")
    
    print(f"{'='*70}")
    
    return results, overall_summary

if __name__ == "__main__":
    generate_report()

"""
Script to compare sorted transactions with original transactions
Creates a DataFrame with both sorted and original transaction lists for analysis
"""
import os
import sys
import django
from datetime import datetime

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rb.settings")
django.setup()

import pandas as pd
from rbapp.models import Bank, RecoBankTransaction
from rbapp.views.banks.bt import BTSortedRecoBankTransactionsView


def create_comparison_dataframe(bank_code=None):
    """
    Create a DataFrame comparing sorted and original transactions
    
    Args:
        bank_code: Bank code to filter by. If None, uses first available bank.
    
    Returns:
        pandas.DataFrame with comparison data
    """
    # Get bank
    if bank_code:
        try:
            bank = Bank.objects.get(code=bank_code)
        except Bank.DoesNotExist:
            print(f"Bank with code '{bank_code}' not found")
            return None
    else:
        # Use first available bank
        bank = Bank.objects.first()
        if not bank:
            print("No banks found in database")
            return None
    
    print(f"Processing transactions for bank: {bank.name} (code: {bank.code})")
    
    # Create a mock request object
    class MockRequest:
        def __init__(self, bank_code):
            self.query_params = {'bank_code': bank_code}
    
    # Instantiate the view and call get method
    view = BTSortedRecoBankTransactionsView()
    request = MockRequest(bank.code)
    
    try:
        response = view.get(request)
        
        if response.status_code != 200:
            print(f"Error: {response.data}")
            return None
        
        data = response.data
        
        # Get sorted and original transactions
        sorted_txs = data.get('transactions', [])
        original_txs = data.get('original_transactions', [])
        
        print(f"Found {len(sorted_txs)} sorted transactions")
        print(f"Found {len(original_txs)} original transactions")
        
        # Create DataFrames
        df_sorted = pd.DataFrame(sorted_txs)
        df_original = pd.DataFrame(original_txs)
        
        # Add source column
        df_sorted['source'] = 'sorted'
        df_original['source'] = 'original'
        
        # Merge on ID to compare positions
        df_comparison = pd.merge(
            df_original[['id', 'original_position', 'operation_date', 'type', 
                        'internal_number', 'is_origine', 'label', 'amount']],
            df_sorted[['id', 'sorted_position', 'operation_date', 'type', 
                      'internal_number', 'is_origine', 'label', 'amount']],
            on='id',
            how='outer',
            suffixes=('_original', '_sorted')
        )
        
        # Calculate position difference
        df_comparison['position_diff'] = df_comparison['sorted_position'] - df_comparison['original_position']
        
        # Add both DataFrames side by side for detailed comparison
        # Rename position columns for clarity
        df_original_renamed = df_original.copy()
        df_original_renamed['position'] = df_original_renamed['original_position']
        df_original_renamed['source'] = 'original'
        
        df_sorted_renamed = df_sorted.copy()
        df_sorted_renamed['position'] = df_sorted_renamed['sorted_position']
        df_sorted_renamed['source'] = 'sorted'
        
        df_combined = pd.concat([
            df_original_renamed,
            df_sorted_renamed
        ], ignore_index=True)
        
        # Sort by source and position
        df_combined = df_combined.sort_values(['source', 'position'])
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'matching_results/sorted_transactions_comparison_{bank.code}_{timestamp}.csv'
        
        # Ensure directory exists
        os.makedirs('matching_results', exist_ok=True)
        
        # Save comparison DataFrame
        df_comparison.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nComparison DataFrame saved to: {output_file}")
        
        # Save combined DataFrame (original + sorted)
        combined_file = f'matching_results/sorted_transactions_combined_{bank.code}_{timestamp}.csv'
        df_combined.to_csv(combined_file, index=False, encoding='utf-8-sig')
        print(f"Combined DataFrame (original + sorted) saved to: {combined_file}")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total transactions: {len(df_comparison)}")
        print(f"Transactions with position change: {(df_comparison['position_diff'] != 0).sum()}")
        print(f"Max position change: {df_comparison['position_diff'].abs().max()}")
        print(f"Average position change: {df_comparison['position_diff'].abs().mean():.2f}")
        
        # Group by internal_number to see grouping
        grouped = df_comparison.groupby('internal_number_original').agg({
            'id': 'count',
            'position_diff': ['min', 'max', 'mean']
        }).round(2)
        print("\n=== Grouping Statistics (by internal_number) ===")
        print(grouped.head(20))
        
        return {
            'comparison': df_comparison,
            'combined': df_combined,
            'sorted': df_sorted,
            'original': df_original,
            'bank': bank.name,
            'bank_code': bank.code
        }
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare sorted and original transactions')
    parser.add_argument('--bank_code', type=str, help='Bank code to process (optional)')
    
    args = parser.parse_args()
    
    result = create_comparison_dataframe(args.bank_code)
    
    if result:
        print("\n=== DataFrames created successfully ===")
        print(f"Bank: {result['bank']} (code: {result['bank_code']})")
        print(f"Comparison DataFrame shape: {result['comparison'].shape}")
        print(f"Combined DataFrame shape: {result['combined'].shape}")
        print("\nYou can now explore the DataFrames:")
        print("  - result['comparison']: Position comparison")
        print("  - result['combined']: All transactions (original + sorted)")
        print("  - result['sorted']: Sorted transactions only")
        print("  - result['original']: Original transactions only")


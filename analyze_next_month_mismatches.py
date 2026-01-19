"""
Analyze the next month's tax comparison mismatches.
"""
import os
import django
from decimal import Decimal, ROUND_HALF_UP

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rb.settings')
django.setup()

from rbapp.models import (
    RecoCustomerTransaction, CustomerTaxRow, RecoBankTransaction, Comparison
)
from django.db.models import Sum

def analyze_mismatches():
    """
    Analyze the mismatches from the next month's results.
    """
    print(f"\n{'='*80}")
    print("ANALYZING NEXT MONTH'S TAX COMPARISON MISMATCHES")
    print(f"{'='*80}\n")
    
    # Get all AGIOS ESCOMPTE comparisons with mismatches
    mismatches = Comparison.objects.filter(
        tax_type__iexact='AGIOS ESCOMPTE',
        status='mismatch'
    ).order_by('matched_bank_transaction_id', 'customer_transaction_id')
    
    print(f"Found {mismatches.count()} AGIOS ESCOMPTE mismatch records\n")
    
    # Group by matched_bank_transaction_id
    groups = {}
    for comp in mismatches:
        bank_tx_id = comp.matched_bank_transaction_id
        if bank_tx_id not in groups:
            groups[bank_tx_id] = []
        groups[bank_tx_id].append(comp)
    
    print(f"Grouped into {len(groups)} unique bank transactions\n")
    
    for bank_tx_id, comps in groups.items():
        if not comps:
            continue
        
        first_comp = comps[0]
        customer_total = first_comp.customer_total_tax
        bank_tax = first_comp.bank_tax
        
        # Get customer transaction IDs
        cust_tx_ids = [c.customer_transaction_id for c in comps]
        first_cust_tx = RecoCustomerTransaction.objects.get(id=first_comp.customer_transaction_id)
        document_number = first_cust_tx.document_number
        
        # Get bank transaction
        try:
            bank_tx = RecoBankTransaction.objects.get(id=bank_tx_id)
        except RecoBankTransaction.DoesNotExist:
            print(f"❌ Bank transaction {bank_tx_id} not found!")
            continue
        
        # Calculate actual sum of individual customer_tax values
        individual_sum = sum([float(c.customer_tax) for c in comps])
        
        # Get all customer tax rows for this document_number
        cust_tax_rows = CustomerTaxRow.objects.filter(
            transaction__document_number=document_number,
            tax_type__iexact='AGIOS ESCOMPTE'
        )
        actual_tax_sum = cust_tax_rows.aggregate(total=Sum('tax_amount'))['total'] or 0
        
        # Get all bank transactions with same internal_number and type
        all_bank_txs = RecoBankTransaction.objects.filter(
            internal_number=bank_tx.internal_number,
            type__iexact='agios escompte'
        )
        bank_tax_sum = all_bank_txs.aggregate(total=Sum('amount'))['total'] or 0
        bank_tax_sum_abs = abs(Decimal(str(bank_tax_sum)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)) if bank_tax_sum else Decimal('0.000')
        
        # Calculate difference
        customer_total_decimal = Decimal(str(customer_total)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        bank_tax_decimal = Decimal(str(bank_tax)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        difference = abs(customer_total_decimal - bank_tax_decimal)
        pct_diff = (difference / bank_tax_decimal * 100) if bank_tax_decimal > 0 else 0
        
        print(f"{'='*80}")
        print(f"Bank Transaction ID: {bank_tx_id}")
        print(f"  - internal_number: {bank_tx.internal_number}")
        print(f"  - document_number: {document_number}")
        print(f"  - label: {bank_tx.label[:80] if bank_tx.label else 'N/A'}...")
        print(f"  - date_ref: {bank_tx.date_ref}")
        print(f"\nCustomer Side:")
        print(f"  - Number of customer transactions: {len(comps)}")
        print(f"  - customer_total_tax (from Comparison): {customer_total}")
        print(f"  - Sum of individual customer_tax values: {individual_sum:.3f}")
        print(f"  - Actual sum from CustomerTaxRow: {actual_tax_sum}")
        print(f"\nBank Side:")
        print(f"  - bank_tax (from Comparison): {bank_tax}")
        print(f"  - Number of bank transactions with same internal_number: {all_bank_txs.count()}")
        print(f"  - Sum of all bank transactions: {bank_tax_sum_abs}")
        print(f"\nComparison:")
        print(f"  - Customer total: {customer_total_decimal}")
        print(f"  - Bank amount: {bank_tax_decimal}")
        print(f"  - Difference: {difference:.3f}")
        print(f"  - Percentage difference: {pct_diff:.2f}%")
        
        # Check if there are other bank transaction types that might be related
        other_bank_txs = RecoBankTransaction.objects.filter(
            internal_number=bank_tx.internal_number
        ).exclude(type__iexact='agios escompte')
        
        if other_bank_txs.exists():
            print(f"\n[WARNING] Other bank transactions with same internal_number:")
            for other_tx in other_bank_txs:
                print(f"    - ID: {other_tx.id}, Type: {other_tx.type}, Amount: {other_tx.amount}, Label: {other_tx.label[:60] if other_tx.label else 'N/A'}")
        
        print(f"\n{'='*80}\n")

if __name__ == '__main__':
    analyze_mismatches()


"""
Script to manually calculate AGIOS ESCOMPTE for a document group and compare with bank amount.
"""
import os
import django
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rb.settings')
django.setup()

from rbapp.models import (
    RecoCustomerTransaction, CustomerTaxRow, RecoBankTransaction,
    Convention, TaxRule, ConventionParameter, Agency
)
from django.db.models import Sum

def calculate_agios_manual(document_number):
    """
    Manually calculate agios for all customer transactions with the same document_number
    and compare with the bank transaction amount.
    """
    print(f"\n{'='*80}")
    print(f"MANUAL AGIOS CALCULATION for document_number: {document_number}")
    print(f"{'='*80}\n")
    
    # Get all customer transactions with this document_number
    cust_txs = RecoCustomerTransaction.objects.filter(document_number=document_number)
    print(f"Found {cust_txs.count()} customer transactions with document_number={document_number}")
    
    if cust_txs.count() == 0:
        print("No customer transactions found!")
        return
    
    # Get the first transaction to find agency and bank
    first_tx = cust_txs.first()
    try:
        agency = Agency.objects.get(code=first_tx.account_number)
        bank = agency.bank
        print(f"Bank: {bank.name} (code: {bank.code})")
    except Agency.DoesNotExist:
        print(f"❌ Agency not found for account_number: {first_tx.account_number}")
        return
    
    # Get convention and tax rule
    convention = Convention.objects.filter(bank=bank, is_active=True).first()
    if not convention:
        print(f"❌ No active convention found for bank: {bank.name}")
        return
    
    print(f"Convention: {convention.name} (ID: {convention.id})")
    
    # Get AGIOS ESCOMPTE tax rule
    tax_rule = TaxRule.objects.filter(
        convention=convention,
        tax_type__iexact='AGIOS ESCOMPTE',
        payment_class__code=first_tx.payment_type,
        payment_status=first_tx.payment_status
    ).first()
    
    if not tax_rule:
        print(f"❌ No AGIOS ESCOMPTE tax rule found")
        return
    
    print(f"Tax Rule: {tax_rule.tax_type} ({tax_rule.calculation_type})")
    print(f"Formula: {tax_rule.formula}")
    
    # Get convention parameters
    params = {p.name: p.value for p in ConventionParameter.objects.all()}
    print(f"\nConvention Parameters:")
    for name, value in params.items():
        print(f"  - {name}: {value}")
    
    # Get existing customer tax rows
    existing_tax_rows = CustomerTaxRow.objects.filter(
        transaction__document_number=document_number,
        tax_type__iexact='AGIOS ESCOMPTE'
    )
    print(f"\nExisting CustomerTaxRow records: {existing_tax_rows.count()}")
    
    # Manual calculation
    print(f"\n{'='*80}")
    print("MANUAL CALCULATION:")
    print(f"{'='*80}\n")
    
    total_manual = Decimal('0.000')
    calculations = []
    
    for idx, tx in enumerate(cust_txs, 1):
        print(f"[{idx}/{cust_txs.count()}] Transaction ID={tx.id}")
        print(f"  - amount: {tx.amount}")
        print(f"  - accounting_date: {tx.accounting_date}")
        print(f"  - due_date: {tx.due_date}")
        print(f"  - payment_type: {tx.payment_type}")
        print(f"  - payment_status: {tx.payment_status}")
        
        # Get variables
        amount = float(tx.amount) if tx.amount else 0
        accounting_date = tx.accounting_date
        due_date = tx.due_date
        
        # Get parameters
        TMM = float(params.get('TMM', 0))
        Taux_convention = float(params.get('Taux_convention', 0))
        bank_days = int(params.get('bank_days', 0))
        
        print(f"  - Variables: amount={amount}, accounting_date={accounting_date}, due_date={due_date}")
        print(f"  - Parameters: TMM={TMM}, Taux_convention={Taux_convention}, bank_days={bank_days}")
        
        # Calculate days difference
        if accounting_date and due_date:
            days_diff = abs((due_date - accounting_date).days) + bank_days
            print(f"  - Days difference: ({due_date} - {accounting_date}) + {bank_days} = {days_diff} days")
            
            # Apply formula: (amount * ((due_date - accounting_date + bank_days) * (TMM + Taux_convention))) / 36000
            numerator = amount * days_diff * (TMM + Taux_convention)
            agios = numerator / 36000
            
            print(f"  - Calculation: ({amount} * {days_diff} * ({TMM} + {Taux_convention})) / 36000")
            print(f"  - Numerator: {numerator}")
            print(f"  - AGIOS: {agios}")
            
            agios_decimal = Decimal(str(agios)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
            total_manual += agios_decimal
            
            calculations.append({
                'tx_id': tx.id,
                'amount': amount,
                'days': days_diff,
                'agios': agios_decimal
            })
            
            print(f"  - AGIOS (rounded): {agios_decimal}")
        else:
            print(f"  - ⚠️ Missing dates, skipping calculation")
        
        print()
    
    print(f"{'='*80}")
    print(f"MANUAL CALCULATION SUMMARY:")
    print(f"{'='*80}")
    print(f"Total manual agios (sum of individual calculations): {total_manual}")
    print(f"Number of transactions calculated: {len(calculations)}")
    
    # Get existing customer tax rows sum
    existing_sum = existing_tax_rows.aggregate(total=Sum('tax_amount'))['total'] or 0
    existing_sum_decimal = Decimal(str(existing_sum)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    print(f"\nExisting CustomerTaxRow sum: {existing_sum_decimal}")
    print(f"Difference (manual vs existing): {abs(total_manual - existing_sum_decimal):.3f}")
    
    # Get bank transaction
    matched_bank_tx = first_tx.matched_bank_transaction
    if matched_bank_tx:
        bank_agios_tx = RecoBankTransaction.objects.filter(
            internal_number=matched_bank_tx.internal_number,
            type__iexact='agios escompte'
        ).first()
        
        if bank_agios_tx:
            bank_amount = abs(Decimal(str(bank_agios_tx.amount)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
            print(f"\nBank Transaction:")
            print(f"  - ID: {bank_agios_tx.id}")
            print(f"  - internal_number: {bank_agios_tx.internal_number}")
            print(f"  - amount: {bank_amount}")
            print(f"\nComparison:")
            print(f"  - Manual calculation: {total_manual}")
            print(f"  - Bank amount: {bank_amount}")
            print(f"  - Difference: {abs(total_manual - bank_amount):.3f}")
            print(f"  - Percentage difference: {(abs(total_manual - bank_amount) / bank_amount * 100) if bank_amount > 0 else 0:.2f}%")
        else:
            print(f"\n❌ No bank agios transaction found with internal_number={matched_bank_tx.internal_number}")
    else:
        print(f"\n❌ Customer transaction {first_tx.id} has no matched bank transaction")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    # Test with one of the document numbers from the investigation
    document_numbers = [
        'BEB24000094',  # Close match (14.702 difference)
        'BEB24000102',  # Medium difference (413.989)
        'BEB24000103',  # Large difference (1741.047)
    ]
    
    for doc_num in document_numbers:
        calculate_agios_manual(doc_num)
        print("\n" + "="*80 + "\n")

















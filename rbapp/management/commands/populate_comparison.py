from django.core.management.base import BaseCommand
from django.utils import timezone
from rbapp.models import CustomerTransaction, CustomerTaxRow, Comparison, BankTransaction
from decimal import Decimal

class Command(BaseCommand):
    help = 'Populate the Comparison table for tax reconciliation between customer and bank transactions.'

    def handle(self, *args, **options):
        # Step 1: Select Customer Transactions Matched to a Bank Transaction
        matched_customers = CustomerTransaction.objects.filter(matched_bank_transaction_id__isnull=False)
        count = 0
        for cust_tx in matched_customers:
            bank_tx = cust_tx.matched_bank_transaction
            if not bank_tx:
                continue
            internal_number = getattr(bank_tx, 'internal_number', None)
            # Step 3: Retrieve customer-side taxes
            cust_tax_rows = CustomerTaxRow.objects.filter(transaction=cust_tx)
            cust_taxes = {row.tax_type.strip().upper(): row.amount for row in cust_tax_rows}
            # Step 4: Retrieve bank-side taxes (assume BankTaxRow or similar, else use BankTransaction fields)
            # For this example, assume bank taxes are stored in CustomerTaxRow with transaction=bank_tx
            bank_tax_rows = CustomerTaxRow.objects.filter(transaction=bank_tx)
            bank_taxes = {row.tax_type.strip().upper(): row.amount for row in bank_tax_rows}
            # Step 5: Union of all tax types
            all_tax_types = set(cust_taxes.keys()) | set(bank_taxes.keys())
            for tax_type in all_tax_types:
                cust_val = cust_taxes.get(tax_type, Decimal('0.00'))
                bank_val = bank_taxes.get(tax_type, Decimal('0.00'))
                # Normalize nulls
                cust_val = cust_val or Decimal('0.00')
                bank_val = bank_val or Decimal('0.00')
                # Compare
                if cust_val == bank_val and cust_val != Decimal('0.00'):
                    status = 'matched'
                elif cust_val == Decimal('0.00') and bank_val == Decimal('0.00'):
                    continue  # skip both zero
                elif cust_val == Decimal('0.00') or bank_val == Decimal('0.00'):
                    status = 'missing'
                else:
                    status = 'mismatch'
                # Optionally, add more fields: difference, internal_number, etc.
                Comparison.objects.update_or_create(
                    customer_transaction=cust_tx,
                    tax_type=tax_type,
                    defaults={
                        'customer_tax': cust_val,
                        'bank_tax': bank_val,
                        'status': status,
                    }
                )
                count += 1
        self.stdout.write(self.style.SUCCESS(f'Populated {count} comparison rows.'))

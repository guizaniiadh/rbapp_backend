from django.core.management.base import BaseCommand
from django.db import transaction
from rbapp.models import BankTransaction, Tax, PaymentClass


class Command(BaseCommand):
    help = 'Classify bank transactions as tax transactions based on keywords in labels'

    def handle(self, *args, **options):
        # Get or create the "tax" payment class
        tax_payment_class, created = PaymentClass.objects.get_or_create(
            code='tax',
            defaults={'name': 'Tax'}
        )
        
        if created:
            self.stdout.write(
                self.style.SUCCESS(f'Created payment class: {tax_payment_class}')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f'Using existing payment class: {tax_payment_class}')
            )

        # Get all tax keywords from the Tax table
        tax_keywords = []
        for tax in Tax.objects.all():
            if tax.description and isinstance(tax.description, list):
                tax_keywords.extend(tax.description)
        
        self.stdout.write(f'Found tax keywords: {tax_keywords}')
        
        # Process all bank transactions
        updated_count = 0
        total_transactions = BankTransaction.objects.count()
        
        with transaction.atomic():
            for bank_transaction in BankTransaction.objects.all():
                label = bank_transaction.label.upper() if bank_transaction.label else ''
                
                # Check if any tax keyword is found in the label
                is_tax_transaction = any(keyword.upper() in label for keyword in tax_keywords)
                
                if is_tax_transaction:
                    # Update payment_class to tax
                    bank_transaction.payment_class = tax_payment_class
                    bank_transaction.save()
                    updated_count += 1
                    
                    self.stdout.write(
                        f'Updated transaction {bank_transaction.id}: "{bank_transaction.label}" -> TAX'
                    )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully processed {total_transactions} transactions. '
                f'Updated {updated_count} transactions as tax transactions.'
            )
        ) 
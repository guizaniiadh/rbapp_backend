from django.core.management.base import BaseCommand
from django.db.models import Sum
from rbapp.models import CustomerTransaction


class Command(BaseCommand):
    help = 'Calculate total_amount for CustomerTransaction records by grouping by document_number, description, and accounting_date'

    def handle(self, *args, **options):
        self.stdout.write('Starting total_amount calculation...')
        
        # Get all unique combinations of document_number, description, and accounting_date
        grouped_transactions = CustomerTransaction.objects.values(
            'document_number', 'description', 'accounting_date'
        ).annotate(
            total_sum=Sum('amount')
        )
        
        updated_count = 0
        
        for group in grouped_transactions:
            document_number = group['document_number']
            description = group['description']
            accounting_date = group['accounting_date']
            total_sum = group['total_sum']
            
            # Update all transactions in this group with the total amount
            updated = CustomerTransaction.objects.filter(
                document_number=document_number,
                description=description,
                accounting_date=accounting_date
            ).update(total_amount=total_sum)
            
            updated_count += updated
            
            self.stdout.write(
                f'Updated {updated} transactions with '
                f'document_number="{document_number}", '
                f'description="{description}", '
                f'accounting_date={accounting_date}, '
                f'total_amount={total_sum}'
            )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully updated {updated_count} CustomerTransaction records with total_amount values'
            )
        ) 
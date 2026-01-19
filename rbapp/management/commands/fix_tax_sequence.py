from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Fix the PostgreSQL sequence for Tax table to prevent duplicate key errors'

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            # Get the maximum ID from the Tax table
            cursor.execute("SELECT MAX(id) FROM rbapp_tax;")
            max_id = cursor.fetchone()[0] or 0
            
            # Reset the sequence to max_id + 1
            next_id = max_id + 1
            cursor.execute(f"SELECT setval('rbapp_tax_id_seq', {next_id}, false);")
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully reset rbapp_tax_id_seq to {next_id} (max_id was {max_id})'
                )
            )




#!/usr/bin/env python
"""
Fix PaymentStatus sequence to prevent duplicate key errors.
Run this if you're getting "duplicate key value violates unique constraint" errors.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rb.settings')
django.setup()

from django.db import connection, models
from rbapp.models import PaymentStatus

# Get the current max ID
max_id = PaymentStatus.objects.aggregate(max_id=models.Max('id'))['max_id'] or 0

# Reset the sequence to the max ID + 1
with connection.cursor() as cursor:
    cursor.execute(
        f"SELECT setval(pg_get_serial_sequence('rbapp_paymentstatus', 'id'), {max_id}, true);"
    )
    print(f"PaymentStatus sequence reset. Next ID will be: {max_id + 1}")


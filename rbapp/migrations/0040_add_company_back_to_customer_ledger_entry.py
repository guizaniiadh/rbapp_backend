# Generated manually to add company field back to CustomerLedgerEntry

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("rbapp", "0039_remove_customerledgerentry_company_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="customerledgerentry",
            name="company",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="customer_ledger_entries",
                to="rbapp.company",
                null=True,  # Allow null initially for existing records
                blank=True
            ),
        ),
    ]











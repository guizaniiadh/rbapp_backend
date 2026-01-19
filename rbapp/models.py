# Comparison model for tax reconciliation
from django.db import models
from decimal import Decimal

class Comparison(models.Model):
    customer_transaction = models.ForeignKey('RecoCustomerTransaction', on_delete=models.CASCADE, related_name='comparisons')
    tax_type = models.CharField(max_length=100)
    customer_tax = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True)
    bank_tax = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True)
    status = models.CharField(max_length=50)
    customer_total_tax = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True)
    matched_bank_transaction_id = models.IntegerField(null=True, blank=True, help_text="Bank transaction ID used for comparison")

    def __str__(self):
        return f"Comparison: {self.customer_transaction} - {self.tax_type}"
import json
from django.contrib.auth import get_user_model
from django.db import models
from django.contrib import admin
from django.contrib.auth.models import User

class PaymentClass(models.Model):
    code = models.CharField(max_length=20, primary_key=True)  # primary key, admin must enter manually
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.code} - {self.name}"

class PaymentStatus(models.Model):
    line = models.IntegerField()
    name = models.CharField(max_length=255)
    payment_class = models.ForeignKey(
        PaymentClass, on_delete=models.CASCADE, related_name='statuses')
    accounting_account = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        unique_together = ('line', 'payment_class')

    def __str__(self):
        return self.name

class PaymentIdentification(models.Model):
    line = models.AutoField(primary_key=True)
    description = models.TextField()
    payment_status = models.ForeignKey(
        PaymentStatus, on_delete=models.CASCADE, related_name='identifications')
    debit = models.BooleanField(default=False)
    credit = models.BooleanField(default=False)
    bank = models.ForeignKey('Bank', on_delete=models.CASCADE, related_name='payment_identifications')
    grouped = models.BooleanField(default=False)

    def __str__(self):
        return self.description

def bank_logo_upload_path(instance, filename):
    # Store in bank_logos/<bank_code>/<filename>
    return f'bank_logos/{instance.code}/{filename}'

class Bank(models.Model):
    code = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    address = models.TextField(blank=True, null=True)
    website = models.URLField(blank=True, null=True)
    logo = models.ImageField(upload_to=bank_logo_upload_path, blank=True, null=True, help_text="Bank logo image")
    statuses = models.ManyToManyField('PaymentStatus', related_name='banks', blank=True)
    beginning_balance_label = models.CharField(
        max_length=255, 
        blank=True, 
        null=True,
        help_text="Text to search for in the label column to identify beginning balance (e.g., 'SOLDE DEBUT PERIODE')"
    )

    def __str__(self):
        return self.name
    
    def get_url_slug(self):
        """
        Generate URL-friendly slug from bank code or name.
        Uses bank code directly for simplicity and robustness.
        
        Returns:
            str: URL slug (e.g., "1", "2", "4")
        """
        return self.code.lower()
    
    def get_view_module_name(self):
        """
        Get the view module name for this bank.
        Dynamically generates module name from bank name.
        
        Returns:
            str: Module name (e.g., "bt", "stb", "attijari", "zitouna")
        """
        # Generate module name from bank name (e.g., "ATTIJARI BANK" -> "attijari")
        import re
        slug = re.sub(r'[^a-z0-9]+', '', self.name.lower())
        return slug

class Agency(models.Model):
    bank = models.ForeignKey(Bank, on_delete=models.CASCADE, related_name='agencies')
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, primary_key=True)
    address = models.TextField(blank=True, null=True)
    city = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.bank.name} - {self.name}"

def ledger_entry_upload_path(instance, filename):
    # Store in ledger_entries/<agency_code>/<filename>
    return f'ledger_entries/{instance.agency.code}/{filename}'

class BankLedgerEntry(models.Model):
    agency = models.ForeignKey(Agency, on_delete=models.CASCADE, related_name='ledger_entries')
    file = models.FileField(upload_to=ledger_entry_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255, blank=True)
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='ledger_entries')

    def __str__(self):
        return self.name or self.file.name

def customer_ledger_entry_upload_path(instance, filename):
    # Store in customer_ledger_entries/<company_name>/<user_id>/<filename>
    company_name = instance.company.name.replace(' ', '_')
    return f'customer_ledger_entries/{company_name}/{instance.user.id}/{filename}'

def company_logo_upload_path(instance, filename):
    # Store in company_logos/<company_code>/<filename>
    return f'company_logos/{instance.code}/{filename}'

class CustomerLedgerEntry(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='customer_ledger_entries')
    company = models.ForeignKey('Company', on_delete=models.CASCADE, related_name='customer_ledger_entries')
    file = models.FileField(upload_to=customer_ledger_entry_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.name or self.file.name

class Company(models.Model):
    name = models.CharField(max_length=255, unique=True)
    code = models.CharField(max_length=20, primary_key=True)
    logo = models.ImageField(upload_to=company_logo_upload_path, blank=True, null=True, help_text="Company logo image")

    def __str__(self):
        return self.name

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    companies = models.ManyToManyField(Company, related_name='users', blank=True)

    def __str__(self):
        company_names = ", ".join([c.name for c in self.companies.all()])
        return f"{self.user.username} - {company_names}"
    
    @property
    def company(self):
        """Backward compatibility: return first company"""
        return self.companies.first()

from .models import Company

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('code', 'name', 'logo')
    search_fields = ('code', 'name')

from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'company')
    search_fields = ('user__username', 'company__name')

class Tax(models.Model):
    name = models.CharField(max_length=50)
    description = models.JSONField(default=list, blank=True)  # List of strings
    company = models.ForeignKey('Company', on_delete=models.CASCADE, related_name='taxes')
    bank = models.ForeignKey('Bank', on_delete=models.CASCADE, related_name='taxes')
    accounting_account = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        unique_together = [['name', 'company', 'bank']]

    def __str__(self):
        return self.name

class BankTransaction(models.Model):
    bank = models.ForeignKey('Bank', on_delete=models.CASCADE, related_name='transactions')  # Add this line
    bank_ledger_entry = models.ForeignKey(BankLedgerEntry, on_delete=models.CASCADE, related_name='transactions')
    import_batch_id = models.IntegerField(null=True, blank=True)
    operation_date = models.DateField()
    label = models.CharField(max_length=255)
    value_date = models.DateField()
    debit = models.FloatField(null=True, blank=True)
    credit = models.FloatField(null=True, blank=True)
    date_ref = models.DateField(blank=True, null=True, help_text="Extracted date reference in YYYY-MM-DD format")  # Extracted date reference
    ref = models.CharField(max_length=255, blank=True, null=True)  # Extracted reference number
    document_reference = models.CharField(max_length=255, blank=True)
    amount = models.DecimalField(max_digits=20, decimal_places=3)
    payment_class = models.ForeignKey(PaymentClass, on_delete=models.SET_NULL, null=True, blank=True)
    payment_status = models.ForeignKey(PaymentStatus, on_delete=models.SET_NULL, null=True, blank=True)
    accounting_account = models.CharField(max_length=100, blank=True, null=True)
    internal_number = models.CharField(max_length=255, blank=True, null=True, help_text="Internal reference number")
    type = models.CharField(max_length=50, blank=True, null=True, help_text="Transaction type classification")

    def __str__(self):
        return f"{self.label} - {self.amount}"

class CustomerTransaction(models.Model):
    customer_ledger_entry = models.ForeignKey(CustomerLedgerEntry, on_delete=models.CASCADE, related_name='transactions')
    import_batch_id = models.IntegerField(null=True, blank=True)
    account_number = models.CharField(max_length=50, blank=True, null=True, help_text="N° compte bancaire")
    accounting_date = models.DateField(help_text="Date comptabilisation")
    document_number = models.CharField(max_length=255, blank=True, null=True, help_text="N° document")
    description = models.CharField(max_length=255, help_text="Description")
    debit_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Montant débit")
    credit_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Montant crédit")
    external_doc_number = models.CharField(max_length=255, blank=True, null=True, help_text="N° doc. externe")
    due_date = models.DateField(null=True, blank=True, help_text="Date d'échéance")
    payment_type = models.CharField(max_length=100, blank=True, null=True, help_text="Type de règlement")
    payment_status = models.ForeignKey('PaymentStatus', on_delete=models.SET_NULL, null=True, blank=True, related_name='customer_transactions', help_text="Status from matched bank transaction")
    amount = models.DecimalField(max_digits=20, decimal_places=3, help_text="Calculated amount (debit - credit)")
    total_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Total amount (absolute value)")
    matched_bank_transaction = models.ForeignKey(BankTransaction, on_delete=models.SET_NULL, null=True, blank=True, related_name='matched_customer_transactions', help_text="Linked bank transaction through internal_number")

    def __str__(self):
        return f"{self.description} - {self.amount}"
    
    def save(self, *args, **kwargs):
        # Calculate total_amount before saving
        self.calculate_total_amount()
        # Set payment_status from matched_bank_transaction if present
        if self.matched_bank_transaction and self.matched_bank_transaction.payment_status:
            self.payment_status = self.matched_bank_transaction.payment_status
        super().save(*args, **kwargs)
    
    def calculate_total_amount(self):
        """Calculate total_amount by summing amounts of transactions with same document_number, description, and accounting_date"""
        from django.db.models import Sum
        
        # Get the sum of amounts for transactions with the same document_number, description, and accounting_date
        total_sum = CustomerTransaction.objects.filter(
            document_number=self.document_number,
            description=self.description,
            accounting_date=self.accounting_date
        ).aggregate(total=Sum('amount'))['total'] or 0
        
        # Add the current transaction's amount if it's not already included
        if self.pk is None:  # New transaction
            total_sum += self.amount or 0
        
        self.total_amount = total_sum


# Staging tables for reconciliation
class RecoBankTransaction(models.Model):
    bank = models.ForeignKey('Bank', on_delete=models.CASCADE, related_name='reco_transactions')
    bank_ledger_entry = models.ForeignKey(BankLedgerEntry, on_delete=models.CASCADE, related_name='reco_transactions')
    import_batch_id = models.IntegerField(null=True, blank=True)
    operation_date = models.DateField()
    label = models.CharField(max_length=255)
    value_date = models.DateField()
    debit = models.FloatField(null=True, blank=True)
    credit = models.FloatField(null=True, blank=True)
    date_ref = models.DateField(blank=True, null=True, help_text="Extracted date reference in YYYY-MM-DD format")
    ref = models.CharField(max_length=255, blank=True, null=True)
    document_reference = models.CharField(max_length=255, blank=True)
    amount = models.DecimalField(max_digits=20, decimal_places=3)
    payment_class = models.ForeignKey(PaymentClass, on_delete=models.SET_NULL, null=True, blank=True)
    payment_status = models.ForeignKey(PaymentStatus, on_delete=models.SET_NULL, null=True, blank=True)
    accounting_account = models.CharField(max_length=100, blank=True, null=True)
    internal_number = models.CharField(max_length=255, blank=True, null=True, help_text="Internal reference number")
    type = models.CharField(max_length=50, blank=True, null=True, help_text="Transaction type classification")

    def __str__(self):
        return f"[RECO] {self.label} - {self.amount}"


class RecoCustomerTransaction(models.Model):
    customer_ledger_entry = models.ForeignKey(CustomerLedgerEntry, on_delete=models.CASCADE, related_name='reco_transactions')
    import_batch_id = models.IntegerField(null=True, blank=True)
    account_number = models.CharField(max_length=50, blank=True, null=True, help_text="N° compte bancaire")
    accounting_date = models.DateField(help_text="Date comptabilisation")
    document_number = models.CharField(max_length=255, blank=True, null=True, help_text="N° document")
    description = models.CharField(max_length=255, help_text="Description")
    debit_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Montant débit")
    credit_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Montant crédit")
    external_doc_number = models.CharField(max_length=255, blank=True, null=True, help_text="N° doc. externe")
    due_date = models.DateField(null=True, blank=True, help_text="Date d'échéance")
    payment_type = models.CharField(max_length=100, blank=True, null=True, help_text="Type de règlement")
    payment_status = models.ForeignKey('PaymentStatus', on_delete=models.SET_NULL, null=True, blank=True, related_name='reco_customer_transactions', help_text="Status from matched bank transaction")
    amount = models.DecimalField(max_digits=20, decimal_places=3, help_text="Calculated amount (debit - credit)")
    total_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Total amount (absolute value)")
    matched_bank_transaction = models.ForeignKey(RecoBankTransaction, on_delete=models.SET_NULL, null=True, blank=True, related_name='matched_reco_customer_transactions', help_text="Linked reco bank transaction")

    def __str__(self):
        return f"[RECO] {self.description} - {self.amount}"

    def save(self, *args, **kwargs):
        # Calculate total_amount before saving
        self.recalculate_total_amount()
        # Set payment_status from matched_bank_transaction if present (mirror main model behavior)
        if self.matched_bank_transaction and self.matched_bank_transaction.payment_status:
            self.payment_status = self.matched_bank_transaction.payment_status
        super().save(*args, **kwargs)

    def recalculate_total_amount(self):
        """Calculate total_amount by summing amounts of transactions with same document_number, description, and accounting_date"""
        from django.db.models import Sum
        total_sum = RecoCustomerTransaction.objects.filter(
            document_number=self.document_number,
            description=self.description,
            accounting_date=self.accounting_date
        ).aggregate(total=Sum('amount'))['total'] or 0
        if self.pk is None:
            total_sum += self.amount or 0
        self.total_amount = total_sum


# Updated Convention model

class Convention(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    bank = models.ForeignKey('Bank', on_delete=models.CASCADE)
    company = models.ForeignKey('Company', on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)
    # Parameters moved to ConventionParameter

    class Meta:
        unique_together = ('company', 'bank')  # Ensure uniqueness

    def __str__(self):
        return f"Convention: {self.name} ({self.company.name} - {self.bank.name})"


# New ConventionParameter model
class ConventionParameter(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    value = models.JSONField(default=list, blank=True, help_text="List of strings or dates")

    def __str__(self):
        return f"{self.name}: {self.value}"


# Stores tax transactions for customer transactions
class CustomerTaxRow(models.Model):
    transaction = models.ForeignKey('RecoCustomerTransaction', on_delete=models.CASCADE, related_name='tax_rows')
    tax_type = models.CharField(max_length=50)  # "TVA", "com", "agios", etc.
    tax_amount = models.DecimalField(max_digits=15, decimal_places=3)
    total_tax_amount = models.DecimalField(max_digits=20, decimal_places=3, null=True, blank=True, help_text="Sum of customer tax for all transactions with the same document_number and tax type")
    applied_formula = models.TextField(null=True, blank=True)  # Store the formula used
    rate_used = models.FloatField(null=True, blank=True)       # If applicable


    def __str__(self):
        return f"{self.tax_type} - {self.tax_amount} for transaction {self.transaction_id}"


# TaxRule model for convention-based tax rules

class TaxRule(models.Model):
    convention = models.ForeignKey('Convention', on_delete=models.CASCADE, related_name='tax_rules')
    payment_class = models.ForeignKey('PaymentClass', on_delete=models.CASCADE)
    payment_status = models.ForeignKey('PaymentStatus', on_delete=models.CASCADE)
    tax_type = models.CharField(max_length=50)  # e.g. "TVA", "com", "agios"
    calculation_type = models.CharField(
        max_length=20,
        choices=[
            ('percentage', 'Percentage'),
            ('flat', 'Flat Fee'),
            ('formula', 'Formula'),
        ]
    )
    rate = models.FloatField(null=True, blank=True)         # Used for percentage or flat
    formula = models.TextField(null=True, blank=True)       # Used for custom formulas (e.g., agios)

    def __str__(self):
        return f"{self.tax_type} rule ({self.payment_class} / {self.payment_status})"

from django.contrib import admin
from .models import PaymentClass, PaymentStatus, PaymentIdentification, Bank, Agency, BankLedgerEntry, CustomerLedgerEntry, Tax, BankTransaction, CustomerTransaction

class PaymentStatusInline(admin.TabularInline):
    model = PaymentStatus
    extra = 1  # how many empty rows to show

@admin.register(PaymentClass)
class PaymentClassAdmin(admin.ModelAdmin):
    list_display = ('code', 'name')
    inlines = [PaymentStatusInline]

@admin.register(PaymentStatus)
class PaymentStatusAdmin(admin.ModelAdmin):
    list_display = ('id', 'line', 'name', 'accounting_account', 'payment_class')
    search_fields = ('name', 'accounting_account')

@admin.register(PaymentIdentification)
class PaymentIdentificationAdmin(admin.ModelAdmin):
    list_display = ('line', 'description', 'payment_status', 'debit', 'credit', 'bank')
    search_fields = ('description', 'bank__name', 'bank__code')

@admin.register(Bank)
class BankAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'address', 'website', 'logo', 'beginning_balance_label')
    search_fields = ('name', 'code')
    filter_horizontal = ('statuses',)
    fields = ('code', 'name', 'address', 'website', 'logo', 'beginning_balance_label', 'statuses')

@admin.register(Agency)
class AgencyAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'city', 'bank')
    search_fields = ('name', 'code', 'city')

@admin.register(BankLedgerEntry)
class BankLedgerEntryAdmin(admin.ModelAdmin):
    list_display = ('name', 'agency', 'user', 'uploaded_at', 'file')
    search_fields = ('name', 'agency__name', 'user__username')

@admin.register(CustomerLedgerEntry)
class CustomerLedgerEntryAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'uploaded_at', 'file')
    search_fields = ('name', 'user__username')

@admin.register(Tax)
class TaxAdmin(admin.ModelAdmin):
    list_display = ('name', 'accounting_account', 'get_description_display', 'company', 'bank')
    search_fields = ('name', 'description', 'accounting_account')
    
    def get_description_display(self, obj):
        """Display description as comma-separated list"""
        if obj.description and isinstance(obj.description, list):
            return ', '.join(obj.description)
        return ''
    get_description_display.short_description = 'Description'

@admin.register(BankTransaction)
class BankTransactionAdmin(admin.ModelAdmin):
    list_display = ('id', 'label', 'amount', 'operation_date', 'accounting_account', 'type')
    list_filter = ('operation_date', 'payment_class', 'payment_status', 'type')
    search_fields = ('label', 'ref', 'date_ref', 'type', 'accounting_account')

@admin.register(CustomerTransaction)
class CustomerTransactionAdmin(admin.ModelAdmin):
    list_display = ('id', 'description', 'amount', 'accounting_date', 'account_number', 'debit_amount', 'credit_amount')
    list_filter = ('accounting_date', 'customer_ledger_entry')
    search_fields = ('description', 'account_number', 'external_doc_number')

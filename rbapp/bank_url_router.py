"""
Dynamic bank URL router.

This module generates URL patterns for bank-specific views based on the Bank model
in the database. It maps bank codes to their corresponding view modules and creates
URLs like /api/{bank_code}/... for each bank.
"""
from django.urls import path
from django.apps import apps

# Mapping of view class names to URL slugs
VIEW_TO_URL_SLUG = {
    'PreprocessBankLedgerEntryView': 'preprocess-bank-ledger-entry',
    'ExtractBeginningBalanceView': 'extract-beginning-balance',
    'MatchCustomerBankTransactionsView': 'match-customer-bank-transactions',
    'ManualMatchCustomerBankTransactionsView': 'manual-match-customer-bank-transactions',
    'MatchTransactionView': 'match-transaction',
    'MatchBankTransactionTaxesView': 'match-bank-transaction-taxes',
    'ExtractCustomerTaxesView': 'extract-customer-taxes',
    'PreprocessCustomerLedgerEntryView': 'preprocess-customer-ledger-entry',
    'GetMatchingResultsView': 'get-matching-results',
    'TaxComparisonView': 'tax-comparison',
    'TaxManagementView': 'tax-management',
    'MatchTaxView': 'match-tax',
    'UnmatchedTransactionsView': 'unmatched-transactions',
    'SortedRecoBankTransactionsView': 'sorted-transactions',
    'AssignInternalNumberView': 'assign-internal-number',
    'SumMatchedBankTransactionsView': 'sum-matched-bank-transactions',
}

# Fallback mapping for known banks (used only when database is unavailable during initialization)
_DEFAULT_BANK_CODE_TO_PREFIX = {
    '1': 'BT',
    '2': 'Zitouna',
    '4': 'STB',
}


def get_bank_code_to_prefix():
    """
    Get bank code to prefix mapping from database dynamically.
    Uses Bank model's get_view_module_name() and converts to prefix format.
    """
    from django.db import connection
    connection.ensure_connection()
    
    Bank = apps.get_model('rbapp', 'Bank')
    banks = Bank.objects.all()
    
    mapping = {}
    for bank in banks:
        module_name = bank.get_view_module_name()  # Returns: 'bt', 'stb', 'attijari', 'zitouna', etc.
        # Convert module name to prefix format
        # Short names (2-3 chars): uppercase -> 'bt' -> 'BT', 'stb' -> 'STB'
        # Long names (4+ chars): capitalize -> 'attijari' -> 'Attijari', 'zitouna' -> 'Zitouna'
        if len(module_name) <= 3:
            prefix = module_name.upper()  # 'bt' -> 'BT', 'stb' -> 'STB'
        else:
            prefix = module_name.capitalize()  # 'attijari' -> 'Attijari', 'zitouna' -> 'Zitouna'
        
        mapping[bank.code] = prefix
    
    return mapping


def get_bank_view_class(prefix, view_name):
    """
    Dynamically import and return bank-specific view class.
    Uses module name from database to import the correct module.
    """
    try:
        # Get the module name from the bank code
        # We need to reverse lookup: prefix -> module_name
        # But we need the bank code to get the module name
        # For now, map prefix back to module name (dynamic approach)
        # Get module name from Bank model using prefix
        from django.db import connection
        connection.ensure_connection()
        
        Bank = apps.get_model('rbapp', 'Bank')
        banks = Bank.objects.all()
        
        # Find bank with matching prefix
        target_module_name = None
        for bank in banks:
            module_name = bank.get_view_module_name()
            # Generate prefix from module name (same logic as get_bank_code_to_prefix)
            if len(module_name) <= 3:
                bank_prefix = module_name.upper()
            else:
                bank_prefix = module_name.capitalize()
            if bank_prefix == prefix:
                target_module_name = module_name
                break
        
        if not target_module_name:
            return None
        
        # Dynamically import the module
        module_path = f'rbapp.views.banks.{target_module_name}'
        import importlib
        bank_module = importlib.import_module(module_path)
        
        full_class_name = f'{prefix}{view_name}'
        if hasattr(bank_module, full_class_name):
            return getattr(bank_module, full_class_name)
        return None
    except (ImportError, AttributeError):
        return None


def get_bank_url_patterns():
    url_patterns = []
    
    from django.db import connection
    connection.ensure_connection()
    
    Bank = apps.get_model('rbapp', 'Bank')
    banks = Bank.objects.all()
    
    # Get dynamic mapping from database
    BANK_CODE_TO_PREFIX = get_bank_code_to_prefix()
    
    for bank in banks:
        bank_code = bank.code
        prefix = BANK_CODE_TO_PREFIX.get(bank_code)
        
        if not prefix:
            continue
        
        for view_name, url_slug in VIEW_TO_URL_SLUG.items():
            view_class = get_bank_view_class(prefix, view_name)
            
            if view_class:
                url_patterns.append(
                    path(
                        f'api/{bank_code}/{url_slug}/',
                        view_class.as_view(),
                        name=f'{prefix.lower()}-{url_slug}'
                    )
                )
    
    return url_patterns

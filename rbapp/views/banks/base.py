"""
Base Classes for Bank-Specific Views

This module contains abstract base classes that all bank-specific views inherit from.
These base classes provide:
- Common HTTP handling (authentication, permissions, request/response)
- Common validation logic
- Template methods that delegate bank-specific logic to subclasses

Bank-specific views (bt.py, stb.py) inherit from these base classes and implement
the bank-specific methods for:
- Dataframe cleaning
- Reference extraction
- Matching algorithms
- Tax extraction
- Beginning balance extraction
"""

# TODO: Import statements will be added here
# from rest_framework.views import APIView
# from rest_framework import permissions
# from rest_framework.response import Response

# TODO: Base view classes to be implemented:
# - BasePreprocessBankLedgerEntryView
# - BaseExtractBeginningBalanceView
# - BaseMatchCustomerBankTransactionsView
# - BaseMatchTransactionView
# - BaseExtractCustomerTaxesView
# - BaseMatchTaxView
# - BaseMatchBankTransactionTaxesView
# - BaseGetMatchingResultsView
# - BaseUnmatchedTransactionsView
# - Any other bank-specific processing views



















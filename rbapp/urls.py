from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from rbapp.views.main import (
    RegisterAPIView, CurrentUserView,
    PaymentClassViewSet, PaymentStatusViewSet, PaymentIdentificationViewSet,
    BankViewSet, AgencyViewSet, BankLedgerEntryViewSet, CustomerLedgerEntryViewSet,
    UserViewSet, UserProfileViewSet, TaxViewSet, BankTransactionViewSet, 
    CustomerTransactionViewSet, CompanyViewSet,
    RecoBankTransactionViewSet, RecoCustomerTransactionViewSet,
    ConventionViewSet, TaxRuleViewSet, ConventionParameterViewSet,
    CustomerTaxRowViewSet
)

from rbapp.bank_url_router import get_bank_url_patterns

router = DefaultRouter()
router.register(r'payment-classes', PaymentClassViewSet, basename='paymentclass')
router.register(r'payment-statuses', PaymentStatusViewSet, basename='paymentstatus')
router.register(r'payment-identifications', PaymentIdentificationViewSet, basename='paymentidentification')
router.register(r'banks', BankViewSet, basename='bank')
router.register(r'agencies', AgencyViewSet, basename='agency')
router.register(r'bank-ledger-entries', BankLedgerEntryViewSet, basename='bankledgerentry')
router.register(r'customer-ledger-entries', CustomerLedgerEntryViewSet, basename='customerledgerentry')
router.register(r'users', UserViewSet, basename='user')
router.register(r'user-profiles', UserProfileViewSet, basename='userprofile')
router.register(r'taxes', TaxViewSet, basename='tax')
router.register(r'bank-transactions', BankTransactionViewSet, basename='banktransaction')
router.register(r'customer-transactions', CustomerTransactionViewSet, basename='customertransaction')
router.register(r'reco-bank-transactions', RecoBankTransactionViewSet, basename='recobanktransaction')
router.register(r'reco-customer-transactions', RecoCustomerTransactionViewSet, basename='recocustomertransaction')
router.register(r'companies', CompanyViewSet, basename='company')
router.register(r'taxrules', TaxRuleViewSet, basename='taxrule')
router.register(r'conventions', ConventionViewSet, basename='convention')
router.register(r'convention-parameters', ConventionParameterViewSet, basename='conventionparameter')
router.register(r'customer-tax-rows', CustomerTaxRowViewSet, basename='customertaxrow')

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/register/', RegisterAPIView.as_view(), name='register'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/current-user/', CurrentUserView.as_view(), name='current-user'),
]

# Helper function to get bank codes from database
def get_bank_codes():
    """Get all bank codes from database dynamically"""
    from django.db import connection
    from django.apps import apps
    connection.ensure_connection()
    Bank = apps.get_model('rbapp', 'Bank')
    return [bank.code for bank in Bank.objects.all()]

# Add nested bank code routes for customer-ledger-entries preprocess
# These support URLs like /api/1/customer-ledger-entries/{id}/preprocess/
# The ViewSet's preprocess action will extract the bank code from the path
for bank_code in get_bank_codes():
    urlpatterns.append(
        path(
            f'api/{bank_code}/customer-ledger-entries/<int:pk>/preprocess/',
            CustomerLedgerEntryViewSet.as_view({'post': 'preprocess'}),
            name=f'customerledgerentry-preprocess-bank-{bank_code}'
        )
    )
    # Add nested bank code routes for bank-ledger-entries preprocess
    urlpatterns.append(
        path(
            f'api/{bank_code}/bank-ledger-entries/<int:pk>/preprocess/',
            BankLedgerEntryViewSet.as_view({'post': 'preprocess'}),
            name=f'bankledgerentry-preprocess-bank-{bank_code}'
        )
    )

# Add bank-specific URL patterns from dynamic router
urlpatterns += get_bank_url_patterns()

# Add plural aliases for frontend compatibility
# Frontend uses /api/{bankCode}/match-transactions/ and /api/{bankCode}/match-taxes/
# but backend has /api/{bankCode}/match-transaction/ and /api/{bankCode}/match-tax/
from rbapp.bank_url_router import get_bank_view_class, get_bank_code_to_prefix
# Get dynamic mapping from database
BANK_CODE_TO_PREFIX = get_bank_code_to_prefix()
for bank_code in get_bank_codes():
    prefix = BANK_CODE_TO_PREFIX.get(bank_code)
    if prefix:
        # Add match-transactions (plural) -> match-transaction (singular)
        match_transaction_view = get_bank_view_class(prefix, 'MatchTransactionView')
        if match_transaction_view:
            urlpatterns.append(
                path(
                    f'api/{bank_code}/match-transactions/',
                    match_transaction_view.as_view(),
                    name=f'{prefix.lower()}-match-transactions'
                )
            )
        
        # Add match-taxes (plural) -> match-tax (singular)
        match_tax_view = get_bank_view_class(prefix, 'MatchTaxView')
        if match_tax_view:
            urlpatterns.append(
                path(
                    f'api/{bank_code}/match-taxes/',
                    match_tax_view.as_view(),
                    name=f'{prefix.lower()}-match-taxes'
                )
            )
        
        # Add sorted-reco-bank-transactions (frontend format) -> sorted-transactions (backend format)
        sorted_transactions_view = get_bank_view_class(prefix, 'SortedRecoBankTransactionsView')
        if sorted_transactions_view:
            urlpatterns.append(
                path(
                    f'api/{bank_code}/sorted-reco-bank-transactions/',
                    sorted_transactions_view.as_view(),
                    name=f'{prefix.lower()}-sorted-reco-bank-transactions'
                )
            )
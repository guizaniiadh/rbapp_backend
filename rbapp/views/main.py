"""
Main CRUD Views

This module contains all generic CRUD operations for entities that don't require
bank-specific logic. These views handle standard operations like:
- Bank, Agency, Company management
- User and UserProfile management
- Tax, PaymentClass, PaymentStatus, PaymentIdentification management
- Convention, ConventionParameter, TaxRule management
- Ledger entry and transaction CRUD operations
- Authentication views (Register, CurrentUser)

All bank-specific processing views are in views/banks/
"""

from rest_framework import viewsets, status, generics, permissions, parsers, serializers
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from django.db import transaction as db_transaction

from rbapp.models import (
    PaymentClass, PaymentStatus, PaymentIdentification, Bank, Agency,
    BankLedgerEntry, CustomerLedgerEntry, BankTransaction, CustomerTransaction,
    Tax, UserProfile, Company, Convention, TaxRule, CustomerTaxRow,
    RecoBankTransaction, RecoCustomerTransaction, ConventionParameter
)
from rbapp.serializers import (
    PaymentClassSerializer, PaymentStatusSerializer, PaymentIdentificationSerializer,
    BankSerializer, AgencySerializer, BankLedgerEntrySerializer, CustomerLedgerEntrySerializer,
    BankTransactionSerializer, CustomerTransactionSerializer, TaxSerializer,
    RegisterSerializer, UserProfileSerializer, UserListSerializer, UserDetailSerializer,
    CompanySerializer, CompanyWithUsersSerializer, CompanyStatsSerializer,
    ConventionSerializer, TaxRuleSerializer, CustomerTaxRowSerializer,
    RecoBankTransactionSerializer, RecoCustomerTransactionSerializer,
    ConventionParameterSerializer
)


# ============================================================================
# Authentication Views
# ============================================================================

class RegisterAPIView(generics.CreateAPIView):
    queryset = get_user_model().objects.all()
    serializer_class = RegisterSerializer


class CurrentUserView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Return current user information"""
        user = request.user
        try:
            user_profile = user.profile
            companies_data = []
            for company in user_profile.companies.all():
                companies_data.append({
                    'code': company.code,
                    'name': company.name,
                })
        except UserProfile.DoesNotExist:
            user_profile = None
            companies_data = []
        
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "date_joined": user.date_joined,
            "is_active": user.is_active,
            "is_staff": user.is_staff,
            "is_superuser": user.is_superuser,
        }
        
        if user_profile:
            user_data["profile"] = {
                "id": user_profile.id,
                "companies": companies_data,
                "company_count": len(companies_data),
            }
        
        return Response(user_data)


# ============================================================================
# CRUD ViewSets
# ============================================================================

class PaymentClassViewSet(viewsets.ModelViewSet):
    queryset = PaymentClass.objects.all()
    serializer_class = PaymentClassSerializer


class PaymentStatusViewSet(viewsets.ModelViewSet):
    queryset = PaymentStatus.objects.all()
    serializer_class = PaymentStatusSerializer


class PaymentIdentificationViewSet(viewsets.ModelViewSet):
    queryset = PaymentIdentification.objects.all()
    serializer_class = PaymentIdentificationSerializer


class BankViewSet(viewsets.ModelViewSet):
    queryset = Bank.objects.all()
    serializer_class = BankSerializer
    lookup_field = 'code'
    
    @action(detail=True, methods=['get'])
    def agencies(self, request, pk=None):
        """Get all agencies for a specific bank"""
        bank = self.get_object()
        agencies = bank.agencies.all()
        serializer = AgencySerializer(agencies, many=True)
        return Response(serializer.data)


class AgencyViewSet(viewsets.ModelViewSet):
    queryset = Agency.objects.all()
    serializer_class = AgencySerializer
    lookup_field = 'code'
    
    def get_queryset(self):
        """Filter agencies by bank if bank parameter is provided"""
        queryset = Agency.objects.all()
        bank_code = self.request.query_params.get('bank', None)
        if bank_code:
            queryset = queryset.filter(bank__code=bank_code)
        return queryset
    
    def update(self, request, *args, **kwargs):
        """Override update to handle code changes properly"""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()  # Get instance using old code from URL
        old_code = instance.code
        
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        
        new_code = serializer.validated_data.get('code')
        
        # If code is being changed
        if new_code and new_code != old_code:
            # Check if an agency with the new code already exists
            if Agency.objects.filter(code=new_code).exists():
                return Response(
                    {'code': ['An agency with this code already exists.']},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Update the code by creating new instance and updating references
            with db_transaction.atomic():
                # Create new instance with all validated data
                # For partial updates, merge existing instance data with validated data
                validated_data = serializer.validated_data.copy()
                
                # If partial update, include fields that weren't updated
                if partial:
                    for field in ['bank', 'name', 'address', 'city']:
                        if field not in validated_data:
                            validated_data[field] = getattr(instance, field)
                
                new_instance = Agency(**validated_data)
                new_instance.save()
                
                # Update all foreign key references
                BankLedgerEntry.objects.filter(agency=instance).update(agency=new_instance)
                
                # Delete the old instance
                instance.delete()
                
                instance = new_instance
        
        else:
            # Normal update (code not changed)
            self.perform_update(serializer)
            instance = serializer.instance
        
        if getattr(instance, '_prefetched_objects_cache', None):
            instance._prefetched_objects_cache = {}
        
        serializer = self.get_serializer(instance)
        return Response(serializer.data)


class BankLedgerEntryViewSet(viewsets.ModelViewSet):
    queryset = BankLedgerEntry.objects.all()
    serializer_class = BankLedgerEntrySerializer
    parser_classes = [parsers.JSONParser, parsers.MultiPartParser, parsers.FormParser]

    @action(detail=True, methods=['get'])
    def transactions(self, request, pk=None):
        """Return all bank transactions for this bank ledger entry."""
        entry = self.get_object()
        qs = entry.transactions.all().order_by('operation_date', 'id')
        serializer = BankTransactionSerializer(qs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def preprocess(self, request, pk=None):
        """
        Preprocess bank ledger entry.
        Routes to bank-specific preprocessing view based on bank code in URL or request.
        Supports URLs like /api/1/bank-ledger-entries/{id}/preprocess/ or /api/bank-ledger-entries/{id}/preprocess/?bank_code=1
        """
        # Get bank code from URL path (e.g., /api/1/bank-ledger-entries/...)
        # or from query parameter or request data
        bank_code = None
        
        # Try to extract from URL path - handle both nested and non-nested patterns
        path = request.path
        path_parts = [p for p in path.strip('/').split('/') if p]
        
        # Look for bank code in path (could be at position 1 if nested: /api/1/bank-ledger-entries/...)
        # or check query parameters and request data
        for i, part in enumerate(path_parts):
            if part.isdigit() and i == 1:  # Bank code would be second part if nested
                # Check if next part is 'bank-ledger-entries'
                if i + 1 < len(path_parts) and path_parts[i + 1] == 'bank-ledger-entries':
                    bank_code = part
                    break
        
        # Fallback: try query parameter or request data
        if not bank_code:
            bank_code = request.query_params.get('bank_code') or request.data.get('bank_code')
        
        if not bank_code:
            return Response({
                "error": "Bank code is required. Provide it in the URL path (e.g., /api/1/bank-ledger-entries/{id}/preprocess/) or as a query parameter (bank_code=1)."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Map bank code to prefix
        BANK_CODE_TO_PREFIX = {
            '1': 'BT',
            '2': 'Zitouna',
            '4': 'STB',
        }
        
        prefix = BANK_CODE_TO_PREFIX.get(bank_code)
        if not prefix:
            return Response({
                "error": f"Unsupported bank code: {bank_code}"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Import and call the bank-specific view
        try:
            if prefix == 'BT':
                from rbapp.views.banks.bt import BTPreprocessBankLedgerEntryView
                view = BTPreprocessBankLedgerEntryView()
            elif prefix == 'STB':
                from rbapp.views.banks.stb import STBPreprocessBankLedgerEntryView
                view = STBPreprocessBankLedgerEntryView()
            elif prefix == 'Zitouna':
                # Try to import Zitouna view, fall back to BT if not available
                try:
                    from rbapp.views.banks.attijari import ZitounaPreprocessBankLedgerEntryView
                    view = ZitounaPreprocessBankLedgerEntryView()
                except ImportError:
                    # Fallback to BT view for Zitouna if specific view doesn't exist
                    from rbapp.views.banks.bt import BTPreprocessBankLedgerEntryView
                    view = BTPreprocessBankLedgerEntryView()
            else:
                return Response({
                    "error": f"No preprocessing view found for bank code: {bank_code}"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Call the bank-specific view's post method
            return view.post(request, pk)
        except ImportError as e:
            return Response({
                "error": f"Could not import bank-specific view: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def perform_create(self, serializer):
        name = self.request.data.get('name')
        if not name:
            # Use the uploaded file's name if no name is provided
            uploaded_file = self.request.FILES.get('file')
            name = uploaded_file.name if uploaded_file else ''
        serializer.save(user=self.request.user, name=name)


class CustomerLedgerEntryViewSet(viewsets.ModelViewSet):
    queryset = CustomerLedgerEntry.objects.all()
    serializer_class = CustomerLedgerEntrySerializer
    parser_classes = [parsers.JSONParser, parsers.MultiPartParser, parsers.FormParser]

    @action(detail=True, methods=['get'])
    def transactions(self, request, pk=None):
        """Return all customer transactions for this ledger entry."""
        entry = self.get_object()
        qs = entry.transactions.all().order_by('accounting_date', 'id')
        serializer = CustomerTransactionSerializer(qs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def preprocess(self, request, pk=None):
        """
        Preprocess customer ledger entry.
        Routes to bank-specific preprocessing view based on bank code in URL or request.
        Supports URLs like /api/1/customer-ledger-entries/{id}/preprocess/ or /api/customer-ledger-entries/{id}/preprocess/?bank_code=1
        """
        # Get bank code from URL path (e.g., /api/1/customer-ledger-entries/...)
        # or from query parameter or request data
        bank_code = None
        
        # Try to extract from URL path - handle both nested and non-nested patterns
        path = request.path
        path_parts = [p for p in path.strip('/').split('/') if p]
        
        # Look for bank code in path (could be at position 1 if nested: /api/1/customer-ledger-entries/...)
        # or check query parameters and request data
        for i, part in enumerate(path_parts):
            if part.isdigit() and i == 1:  # Bank code would be second part if nested
                # Check if next part is 'customer-ledger-entries'
                if i + 1 < len(path_parts) and path_parts[i + 1] == 'customer-ledger-entries':
                    bank_code = part
                    break
        
        # Fallback: try query parameter or request data
        if not bank_code:
            bank_code = request.query_params.get('bank_code') or request.data.get('bank_code')
        
        if not bank_code:
            return Response({
                "error": "Bank code is required. Provide it in the URL path (e.g., /api/1/customer-ledger-entries/{id}/preprocess/) or as a query parameter (bank_code=1)."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Map bank code to prefix
        BANK_CODE_TO_PREFIX = {
            '1': 'BT',
            '2': 'Zitouna',
            '4': 'STB',
        }
        
        prefix = BANK_CODE_TO_PREFIX.get(bank_code)
        if not prefix:
            return Response({
                "error": f"Unsupported bank code: {bank_code}"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Import and call the bank-specific view
        try:
            if prefix == 'BT':
                from rbapp.views.banks.bt import BTPreprocessCustomerLedgerEntryView
                view = BTPreprocessCustomerLedgerEntryView()
            elif prefix == 'STB':
                from rbapp.views.banks.stb import STBPreprocessCustomerLedgerEntryView
                view = STBPreprocessCustomerLedgerEntryView()
            elif prefix == 'Zitouna':
                # Try to import Zitouna view, fall back to BT if not available
                try:
                    from rbapp.views.banks.attijari import ZitounaPreprocessCustomerLedgerEntryView
                    view = ZitounaPreprocessCustomerLedgerEntryView()
                except ImportError:
                    # Fallback to BT view for Zitouna if specific view doesn't exist
                    from rbapp.views.banks.bt import BTPreprocessCustomerLedgerEntryView
                    view = BTPreprocessCustomerLedgerEntryView()
            else:
                return Response({
                    "error": f"No preprocessing view found for bank code: {bank_code}"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Call the bank-specific view's post method
            return view.post(request, pk)
        except ImportError as e:
            return Response({
                "error": f"Could not import bank-specific view: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def perform_create(self, serializer):
        name = self.request.data.get('name')
        if not name:
            uploaded_file = self.request.FILES.get('file')
            name = uploaded_file.name if uploaded_file else ''
        
        # Get company_code from form data (preferred to avoid FK name collision)
        company_code = self.request.data.get('company_code')
        if not company_code:
            raise serializers.ValidationError("Company code is required. Provide it in the 'company_code' field.")
        
        try:
            company = Company.objects.get(code=company_code)
        except Company.DoesNotExist:
            raise serializers.ValidationError(f"Company with code '{company_code}' does not exist.")
        
        serializer.save(user=self.request.user, company=company, name=name)


class UserViewSet(viewsets.ModelViewSet):
    """Direct User API for frontend compatibility"""
    queryset = get_user_model().objects.all()
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return UserListSerializer
        return UserDetailSerializer
    
    def list(self, request, *args, **kwargs):
        """Get all users with their company information"""
        users = get_user_model().objects.all()
        users_data = []
        
        for user in users:
            try:
                profile = user.profile
                companies_data = []
                for company in profile.companies.all():
                    companies_data.append({
                        'code': company.code,
                        'name': company.name
                    })
                
                users_data.append({
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'is_active': user.is_active,
                    'is_staff': user.is_staff,
                    'is_superuser': user.is_superuser,
                    'date_joined': user.date_joined,
                    'last_login': user.last_login,
                    'companies': companies_data,
                    'company_count': profile.companies.count()
                })
            except UserProfile.DoesNotExist:
                # User without profile
                users_data.append({
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'is_active': user.is_active,
                    'is_staff': user.is_staff,
                    'is_superuser': user.is_superuser,
                    'date_joined': user.date_joined,
                    'last_login': user.last_login,
                    'companies': [],
                    'company_count': 0
                })
        
        return Response(users_data)
    
    def retrieve(self, request, *args, **kwargs):
        """Get specific user with company information"""
        user = self.get_object()
        try:
            profile = user.profile
            companies_data = []
            for company in profile.companies.all():
                companies_data.append({
                    'code': company.code,
                    'name': company.name
                })
            
            user_data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'is_active': user.is_active,
                'is_staff': user.is_staff,
                'is_superuser': user.is_superuser,
                'date_joined': user.date_joined,
                'last_login': user.last_login,
                'companies': companies_data,
                'company_count': profile.companies.count()
            }
        except UserProfile.DoesNotExist:
            user_data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'is_active': user.is_active,
                'is_staff': user.is_staff,
                'is_superuser': user.is_superuser,
                'date_joined': user.date_joined,
                'last_login': user.last_login,
                'companies': [],
                'company_count': 0
            }
        
        return Response(user_data)
    
    @action(detail=True, methods=['post'])
    def assign_to_company(self, request, pk=None):
        """Assign user to a company"""
        user = self.get_object()
        company_code = request.data.get('company_code')
        
        if not company_code:
            return Response({'error': 'company_code is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(code=company_code)
            user_profile, created = UserProfile.objects.get_or_create(user=user)
            
            # Add company to user's companies
            user_profile.companies.add(company)
            
            return Response({
                'message': f'User {user.username} assigned to company {company.name}',
                'user_id': user.id,
                'username': user.username,
                'company_code': company.code,
                'company_name': company.name
            })
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def remove_from_company(self, request, pk=None):
        """Remove user from a company"""
        user = self.get_object()
        company_code = request.data.get('company_code')
        
        if not company_code:
            return Response({'error': 'company_code is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(code=company_code)
            user_profile = user.profile
            
            # Remove company from user's companies
            user_profile.companies.remove(company)
            
            return Response({
                'message': f'User {user.username} removed from company {company.name}',
                'user_id': user.id,
                'username': user.username,
                'company_code': company.code,
                'company_name': company.name
            })
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User profile not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['get'])
    def companies(self, request, pk=None):
        """Get all companies for a user"""
        user = self.get_object()
        try:
            profile = user.profile
            companies_data = []
            
            for company in profile.companies.all():
                companies_data.append({
                    'code': company.code,
                    'name': company.name
                })
            
            return Response({
                'user_id': user.id,
                'username': user.username,
                'companies': companies_data
            })
        except UserProfile.DoesNotExist:
            return Response({
                'user_id': user.id,
                'username': user.username,
                'companies': []
            })


class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=True, methods=['post'])
    def assign_to_company(self, request, pk=None):
        """Assign user to a company"""
        user_profile = self.get_object()
        company_code = request.data.get('company_code')
        
        if not company_code:
            return Response({'error': 'company_code is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(code=company_code)
            user_profile.companies.add(company)
            
            return Response({
                'message': f'User {user_profile.user.username} assigned to company {company.name}',
                'user_id': user_profile.user.id,
                'company_code': company.code,
                'company_name': company.name,
            })
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def remove_from_company(self, request, pk=None):
        """Remove user from a company"""
        user_profile = self.get_object()
        company_code = request.data.get('company_code')
        
        if not company_code:
            return Response({'error': 'company_code is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            company = Company.objects.get(code=company_code)
            user_profile.companies.remove(company)
            
            return Response({
                'message': f'User {user_profile.user.username} removed from company {company.name}',
                'user_id': user_profile.user.id,
                'company_code': company.code,
                'company_name': company.name
            })
        except Company.DoesNotExist:
            return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['get'])
    def companies(self, request, pk=None):
        """Get all companies for a user"""
        user_profile = self.get_object()
        companies_data = []
        
        for company in user_profile.companies.all():
            companies_data.append({
                'code': company.code,
                'name': company.name,
            })
        
        return Response({
            'user_id': user_profile.user.id,
            'username': user_profile.user.username,
            'companies': companies_data,
        })


class TaxViewSet(viewsets.ModelViewSet):
    queryset = Tax.objects.all()
    serializer_class = TaxSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        bank_id = self.request.query_params.get('bank')
        company_id = self.request.query_params.get('company')
        if bank_id:
            queryset = queryset.filter(bank_id=bank_id)
        if company_id:
            queryset = queryset.filter(company_id=company_id)
        return queryset


class BankTransactionViewSet(viewsets.ModelViewSet):
    queryset = BankTransaction.objects.all()
    serializer_class = BankTransactionSerializer
    
    @action(detail=True, methods=['get'])
    def info(self, request, pk=None):
        """
        Get basic info about a bank transaction for debugging
        """
        try:
            bank_transaction = self.get_object()
            return Response({
                'id': bank_transaction.id,
                'label': bank_transaction.label,
                'amount': float(bank_transaction.amount),
                'internal_number': bank_transaction.internal_number,
                'type': bank_transaction.type,
                'operation_date': bank_transaction.operation_date.isoformat(),
                'exists': True
            })
        except BankTransaction.DoesNotExist:
            return Response({
                'error': 'Bank transaction not found',
                'bank_transaction_id': pk,
                'exists': False
            }, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['get'])
    def tax_rows(self, request, pk=None):
        """
        Get all tax rows related to a specific bank transaction
        """
        try:
            # Get the bank transaction
            bank_transaction = self.get_object()
            
            # Get all tax rows for customer transactions that are matched to this bank transaction
            tax_rows = CustomerTaxRow.objects.filter(
                transaction__matched_bank_transaction=bank_transaction
            ).select_related('transaction')
            
            # Build simple response
            tax_rows_data = []
            for tax_row in tax_rows:
                tax_rows_data.append({
                    'tax_row_id': tax_row.id,
                    'tax_type': tax_row.tax_type,
                    'tax_amount': float(tax_row.tax_amount),
                    'total_tax_amount': float(tax_row.total_tax_amount) if tax_row.total_tax_amount else None,
                    'applied_formula': tax_row.applied_formula,
                    'rate_used': tax_row.rate_used,
                    'customer_transaction_id': tax_row.transaction.id,
                    'customer_transaction_description': tax_row.transaction.description,
                    'customer_transaction_amount': float(tax_row.transaction.amount)
                })
            
            return Response({
                'bank_transaction_id': bank_transaction.id,
                'bank_transaction_label': bank_transaction.label,
                'bank_transaction_amount': float(bank_transaction.amount),
                'tax_rows': tax_rows_data,
                'total_tax_rows': len(tax_rows_data)
            })
            
        except BankTransaction.DoesNotExist:
            return Response({
                'error': 'Bank transaction not found',
                'bank_transaction_id': pk
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e),
                'bank_transaction_id': pk
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CustomerTransactionViewSet(viewsets.ModelViewSet):
    queryset = CustomerTransaction.objects.all()
    serializer_class = CustomerTransactionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset().select_related(
            'customer_ledger_entry', 
            'payment_status', 
            'matched_bank_transaction',
            'customer_ledger_entry__user',
            'customer_ledger_entry__company'
        )
        
        # Comprehensive filtering capabilities
        customer_ledger_entry_id = self.request.query_params.get('customer_ledger_entry')
        import_batch_id = self.request.query_params.get('import_batch_id')
        account_number = self.request.query_params.get('account_number')
        accounting_date = self.request.query_params.get('accounting_date')
        accounting_date_from = self.request.query_params.get('accounting_date_from')
        accounting_date_to = self.request.query_params.get('accounting_date_to')
        document_number = self.request.query_params.get('document_number')
        external_doc_number = self.request.query_params.get('external_doc_number')
        amount = self.request.query_params.get('amount')
        amount_min = self.request.query_params.get('amount_min')
        amount_max = self.request.query_params.get('amount_max')
        payment_type = self.request.query_params.get('payment_type')
        matched_bank_transaction_id = self.request.query_params.get('matched_bank_transaction')
        description = self.request.query_params.get('description')
        due_date = self.request.query_params.get('due_date')
        due_date_from = self.request.query_params.get('due_date_from')
        due_date_to = self.request.query_params.get('due_date_to')
        payment_status_id = self.request.query_params.get('payment_status')
        has_matched_bank_transaction = self.request.query_params.get('has_matched_bank_transaction')
        
        # Apply filters
        if customer_ledger_entry_id:
            queryset = queryset.filter(customer_ledger_entry_id=customer_ledger_entry_id)
        if import_batch_id:
            queryset = queryset.filter(import_batch_id=import_batch_id)
        if account_number:
            queryset = queryset.filter(account_number=account_number)
        if accounting_date:
            queryset = queryset.filter(accounting_date=accounting_date)
        if accounting_date_from:
            queryset = queryset.filter(accounting_date__gte=accounting_date_from)
        if accounting_date_to:
            queryset = queryset.filter(accounting_date__lte=accounting_date_to)
        if document_number:
            queryset = queryset.filter(document_number__icontains=document_number)
        if external_doc_number:
            queryset = queryset.filter(external_doc_number__icontains=external_doc_number)
        if amount:
            queryset = queryset.filter(amount=amount)
        if amount_min:
            queryset = queryset.filter(amount__gte=amount_min)
        if amount_max:
            queryset = queryset.filter(amount__lte=amount_max)
        if payment_type:
            queryset = queryset.filter(payment_type__icontains=payment_type)
        if matched_bank_transaction_id:
            queryset = queryset.filter(matched_bank_transaction_id=matched_bank_transaction_id)
        if description:
            queryset = queryset.filter(description__icontains=description)
        if due_date:
            queryset = queryset.filter(due_date=due_date)
        if due_date_from:
            queryset = queryset.filter(due_date__gte=due_date_from)
        if due_date_to:
            queryset = queryset.filter(due_date__lte=due_date_to)
        if payment_status_id:
            queryset = queryset.filter(payment_status_id=payment_status_id)
        if has_matched_bank_transaction is not None:
            if has_matched_bank_transaction.lower() == 'true':
                queryset = queryset.filter(matched_bank_transaction__isnull=False)
            elif has_matched_bank_transaction.lower() == 'false':
                queryset = queryset.filter(matched_bank_transaction__isnull=True)
        
        return queryset
    
    @action(detail=True, methods=['get'])
    def tax_rows(self, request, pk=None):
        """
        Get all tax rows for a specific customer transaction
        """
        try:
            customer_transaction = self.get_object()
            tax_rows = CustomerTaxRow.objects.filter(transaction=customer_transaction)
            
            tax_rows_data = []
            for tax_row in tax_rows:
                tax_rows_data.append({
                    'id': tax_row.id,
                    'tax_type': tax_row.tax_type,
                    'tax_amount': float(tax_row.tax_amount),
                    'total_tax_amount': float(tax_row.total_tax_amount) if tax_row.total_tax_amount else None,
                    'applied_formula': tax_row.applied_formula,
                    'rate_used': tax_row.rate_used
                })
            
            return Response({
                'customer_transaction_id': customer_transaction.id,
                'customer_transaction_description': customer_transaction.description,
                'customer_transaction_amount': float(customer_transaction.amount),
                'tax_rows': tax_rows_data,
                'total_tax_rows': len(tax_rows_data)
            })
            
        except CustomerTransaction.DoesNotExist:
            return Response({
                'error': 'Customer transaction not found',
                'customer_transaction_id': pk
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e),
                'customer_transaction_id': pk
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def matched_bank_transaction(self, request, pk=None):
        """
        Get the matched bank transaction for a customer transaction
        """
        try:
            customer_transaction = self.get_object()
            
            if not customer_transaction.matched_bank_transaction:
                return Response({
                    'message': 'No matched bank transaction found',
                    'customer_transaction_id': pk
                })
            
            bank_transaction = customer_transaction.matched_bank_transaction
            
            return Response({
                'customer_transaction_id': customer_transaction.id,
                'customer_transaction_description': customer_transaction.description,
                'customer_transaction_amount': float(customer_transaction.amount),
                'matched_bank_transaction': {
                    'id': bank_transaction.id,
                    'label': bank_transaction.label,
                    'amount': float(bank_transaction.amount),
                    'operation_date': bank_transaction.operation_date.isoformat(),
                    'internal_number': bank_transaction.internal_number,
                    'type': bank_transaction.type,
                    'ref': bank_transaction.ref
                }
            })
            
        except CustomerTransaction.DoesNotExist:
            return Response({
                'error': 'Customer transaction not found',
                'customer_transaction_id': pk
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e),
                'customer_transaction_id': pk
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RecoBankTransactionViewSet(viewsets.ModelViewSet):
    queryset = RecoBankTransaction.objects.all()
    serializer_class = RecoBankTransactionSerializer
    
    @action(detail=False, methods=['delete'])
    def empty(self, request):
        """
        Empty the RecoBankTransaction table using TRUNCATE for maximum speed.
        TRUNCATE is much faster than DELETE as it resets the table in one operation.
        First clears foreign key references from RecoCustomerTransaction to avoid constraint violations.
        """
        from django.db import connection
        from rbapp.models import RecoCustomerTransaction
        
        # Get count before deletion
        count = RecoBankTransaction.objects.count()
        
        # First, clear all foreign key references from RecoCustomerTransaction
        # This prevents foreign key constraint violations when truncating/deleting
        updated_count = RecoCustomerTransaction.objects.filter(
            matched_bank_transaction__isnull=False
        ).update(matched_bank_transaction=None)
        
        # Get table name from model
        table_name = RecoBankTransaction._meta.db_table
        
        # Use TRUNCATE TABLE for fastest possible operation (empties table in one click)
        # TRUNCATE is faster than DELETE because it's a DDL operation that resets the table
        try:
            with connection.cursor() as cursor:
                # TRUNCATE TABLE is the fastest way to empty a table
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
                method_used = 'TRUNCATE'
        except Exception as e:
            # Fallback to DELETE if TRUNCATE fails (e.g., due to other foreign key constraints)
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                method_used = 'DELETE'
        return Response({
            'message': f'Emptied RecoBankTransaction table. Deleted {count} rows. Cleared {updated_count} foreign key references.',
            'deleted_count': count,
            'cleared_references': updated_count,
            'method': method_used,
        }, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], url_path='unmatched')
    def unmatched(self, request):
        """
        Return the list of RecoBankTransaction rows that are NOT matched
        to any RecoCustomerTransaction (matched_reco_customer_transactions is empty).
        """
        unmatched_qs = RecoBankTransaction.objects.filter(
            matched_reco_customer_transactions__isnull=True
        ).distinct()
        serializer = self.get_serializer(unmatched_qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], url_path='unmatched-count')
    def unmatched_count(self, request):
        """
        Return the number of RecoBankTransaction rows that are NOT matched
        to any RecoCustomerTransaction (matched_reco_customer_transactions is empty).
        """
        # A bank transaction is "unmatched" if no RecoCustomerTransaction points to it
        unmatched_qs = RecoBankTransaction.objects.filter(
            matched_reco_customer_transactions__isnull=True
        ).distinct()
        count = unmatched_qs.count()

        return Response(
            {
                'unmatched_reco_bank_transactions': count,
            },
            status=status.HTTP_200_OK,
        )
    
    @action(detail=True, methods=['get'], url_path='taxes')
    def taxes(self, request, pk=None):
        """
        Get all tax transactions (RecoBankTransaction rows) associated with this RecoBankTransaction.
        Taxes are identified by having the same internal_number but different type.
        """
        try:
            bank_transaction = self.get_object()
            
            if not bank_transaction.internal_number:
                return Response({
                    'bank_transaction_id': bank_transaction.id,
                    'internal_number': None,
                    'message': 'This transaction has no internal_number, so no taxes can be associated.',
                    'taxes': []
                }, status=status.HTTP_200_OK)
            
            # Get all RecoBankTransaction rows with the same internal_number
            # Exclude the current transaction itself and filter out 'origine' type
            tax_transactions = RecoBankTransaction.objects.filter(
                internal_number=bank_transaction.internal_number
            ).exclude(id=bank_transaction.id).exclude(type='origine')
            
            taxes_data = []
            for tax_tx in tax_transactions:
                taxes_data.append({
                    'id': tax_tx.id,
                    'type': tax_tx.type,
                    'amount': float(tax_tx.amount),
                    'operation_date': tax_tx.operation_date.strftime('%Y-%m-%d') if tax_tx.operation_date else None,
                    'label': tax_tx.label,
                    'ref': tax_tx.ref,
                    'internal_number': tax_tx.internal_number
                })
            
            serializer = self.get_serializer(bank_transaction)
            return Response({
                'bank_transaction': serializer.data,
                'taxes': taxes_data,
                'total_taxes': len(taxes_data),
                'internal_number': bank_transaction.internal_number
            }, status=status.HTTP_200_OK)
            
        except RecoBankTransaction.DoesNotExist:
            return Response({
                'error': 'RecoBankTransaction not found',
                'bank_transaction_id': pk
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': str(e),
                'bank_transaction_id': pk
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'], url_path='with-taxes')
    def with_taxes(self, request):
        """
        Get all RecoBankTransaction rows with their associated taxes.
        Returns transactions grouped by internal_number with their taxes.
        """
        try:
            # Get all transactions that have an internal_number
            transactions_with_internal = RecoBankTransaction.objects.filter(
                internal_number__isnull=False
            ).exclude(internal_number='')
            
            # Group by internal_number
            result = []
            processed_internal_numbers = set()
            
            for transaction in transactions_with_internal:
                internal_num = transaction.internal_number
                
                # Skip if we've already processed this internal_number
                if internal_num in processed_internal_numbers:
                    continue
                
                processed_internal_numbers.add(internal_num)
                
                # Get all transactions with this internal_number
                all_related = RecoBankTransaction.objects.filter(
                    internal_number=internal_num
                )
                
                # Separate main transaction (type='origine' or first one) and taxes
                main_transaction = all_related.filter(type='origine').first()
                if not main_transaction:
                    # If no 'origine', use the first one as main
                    main_transaction = all_related.order_by('id').first()
                
                # Get taxes (all others with same internal_number)
                tax_transactions = all_related.exclude(id=main_transaction.id).exclude(type='origine')
                
                taxes_data = []
                for tax_tx in tax_transactions:
                    taxes_data.append({
                        'id': tax_tx.id,
                        'type': tax_tx.type,
                        'amount': float(tax_tx.amount),
                        'operation_date': tax_tx.operation_date.strftime('%Y-%m-%d') if tax_tx.operation_date else None,
                        'label': tax_tx.label,
                        'ref': tax_tx.ref,
                        'internal_number': tax_tx.internal_number
                    })
                
                main_serializer = self.get_serializer(main_transaction)
                result.append({
                    'bank_transaction': main_serializer.data,
                    'taxes': taxes_data,
                    'total_taxes': len(taxes_data),
                    'internal_number': internal_num
                })
            
            return Response({
                'transactions_with_taxes': result,
                'total_transactions': len(result)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RecoCustomerTransactionViewSet(viewsets.ModelViewSet):
    queryset = RecoCustomerTransaction.objects.all()
    serializer_class = RecoCustomerTransactionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = super().get_queryset().select_related(
            'customer_ledger_entry', 
            'payment_status', 
            'matched_bank_transaction',
            'customer_ledger_entry__user',
            'customer_ledger_entry__company'
        )
        params = self.request.query_params
        if params.get('customer_ledger_entry'):
            queryset = queryset.filter(customer_ledger_entry_id=params.get('customer_ledger_entry'))
        if params.get('import_batch_id'):
            queryset = queryset.filter(import_batch_id=params.get('import_batch_id'))
        if params.get('account_number'):
            queryset = queryset.filter(account_number=params.get('account_number'))
        if params.get('accounting_date'):
            queryset = queryset.filter(accounting_date=params.get('accounting_date'))
        if params.get('accounting_date_from'):
            queryset = queryset.filter(accounting_date__gte=params.get('accounting_date_from'))
        if params.get('accounting_date_to'):
            queryset = queryset.filter(accounting_date__lte=params.get('accounting_date_to'))
        if params.get('document_number'):
            queryset = queryset.filter(document_number__icontains=params.get('document_number'))
        if params.get('external_doc_number'):
            queryset = queryset.filter(external_doc_number__icontains=params.get('external_doc_number'))
        if params.get('amount'):
            queryset = queryset.filter(amount=params.get('amount'))
        if params.get('amount_min'):
            queryset = queryset.filter(amount__gte=params.get('amount_min'))
        if params.get('amount_max'):
            queryset = queryset.filter(amount__lte=params.get('amount_max'))
        if params.get('payment_type'):
            queryset = queryset.filter(payment_type__icontains=params.get('payment_type'))
        if params.get('matched_bank_transaction'):
            queryset = queryset.filter(matched_bank_transaction_id=params.get('matched_bank_transaction'))
        if params.get('description'):
            queryset = queryset.filter(description__icontains=params.get('description'))
        if params.get('due_date'):
            queryset = queryset.filter(due_date=params.get('due_date'))
        if params.get('due_date_from'):
            queryset = queryset.filter(due_date__gte=params.get('due_date_from'))
        if params.get('due_date_to'):
            queryset = queryset.filter(due_date__lte=params.get('due_date_to'))
        if params.get('payment_status'):
            queryset = queryset.filter(payment_status_id=params.get('payment_status'))
        has_matched = params.get('has_matched_bank_transaction')
        if has_matched is not None:
            if has_matched.lower() == 'true':
                queryset = queryset.filter(matched_bank_transaction__isnull=False)
            elif has_matched.lower() == 'false':
                queryset = queryset.filter(matched_bank_transaction__isnull=True)
        return queryset
    
    @action(detail=False, methods=['delete'])
    def empty(self, request):
        """
        Empty the RecoCustomerTransaction table using TRUNCATE for maximum speed.
        TRUNCATE is much faster than DELETE as it resets the table in one operation.
        """
        from django.db import connection
        
        # Get count before deletion
        count = RecoCustomerTransaction.objects.count()
        
        # Get table name from model
        table_name = RecoCustomerTransaction._meta.db_table
        
        # Use TRUNCATE TABLE for fastest possible operation (empties table in one click)
        # TRUNCATE is faster than DELETE because it's a DDL operation that resets the table
        try:
            with connection.cursor() as cursor:
                # TRUNCATE TABLE is the fastest way to empty a table
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
                method_used = 'TRUNCATE'
        except Exception as e:
            # Fallback to DELETE if TRUNCATE fails (e.g., due to foreign key constraints)
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                method_used = 'DELETE'
        
        return Response({
            'message': f'Emptied RecoCustomerTransaction table. Deleted {count} rows.',
            'deleted_count': count,
            'method': method_used
        }, status=status.HTTP_200_OK)


class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action and query parameters"""
        if self.action == 'list' and self.request.query_params.get('include_users'):
            return CompanyWithUsersSerializer
        elif self.action == 'retrieve' and self.request.query_params.get('include_users'):
            return CompanyWithUsersSerializer
        elif self.action == 'stats':
            return CompanyStatsSerializer
        return CompanySerializer
    
    @action(detail=True, methods=['get'])
    def users(self, request, pk=None):
        """Get all users for a specific company"""
        company = self.get_object()
        user_profiles = company.users.all()
        users_data = []
        for profile in user_profiles:
            users_data.append({
                'id': profile.user.id,
                'username': profile.user.username,
                'email': profile.user.email,
                'first_name': profile.user.first_name,
                'last_name': profile.user.last_name,
                'is_active': profile.user.is_active,
                'is_staff': profile.user.is_staff,
                'is_superuser': profile.user.is_superuser,
                'date_joined': profile.user.date_joined,
                'last_login': profile.user.last_login,
                'all_companies': [{'code': c.code, 'name': c.name} for c in profile.companies.all()]
            })
        return Response(users_data)
    
    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """Get company statistics"""
        company = self.get_object()
        return Response({
            'code': company.code,
            'name': company.name,
            'total_users': company.users.count(),
            'active_users': company.users.filter(user__is_active=True).count(),
            'inactive_users': company.users.filter(user__is_active=False).count(),
            'staff_users': company.users.filter(user__is_staff=True).count(),
        })
    
    @action(detail=True, methods=['post'])
    def assign_user(self, request, pk=None):
        """Assign a user to this company"""
        company = self.get_object()
        user_id = request.data.get('user_id')
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = get_user_model().objects.get(id=user_id)
            user_profile, created = UserProfile.objects.get_or_create(user=user)
            
            # Add company to user's companies
            user_profile.companies.add(company)
            
            return Response({
                'message': f'User {user.username} assigned to company {company.name}',
                'user_id': user.id,
                'username': user.username,
                'company_code': company.code,
                'company_name': company.name
            })
        except get_user_model().DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def remove_user(self, request, pk=None):
        """Remove a user from this company"""
        company = self.get_object()
        user_id = request.data.get('user_id')
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = get_user_model().objects.get(id=user_id)
            user_profile = user.profile
            
            # Remove company from user's companies
            user_profile.companies.remove(company)
            
            return Response({
                'message': f'User {user.username} removed from company {company.name}',
                'user_id': user.id,
                'username': user.username,
                'company_code': company.code,
                'company_name': company.name
            })
        except get_user_model().DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User profile not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['get'])
    def active_users(self, request, pk=None):
        """Get only active users for a specific company"""
        company = self.get_object()
        active_profiles = company.users.filter(user__is_active=True)
        users_data = []
        for profile in active_profiles:
            users_data.append({
                'id': profile.user.id,
                'username': profile.user.username,
                'email': profile.user.email,
                'first_name': profile.user.first_name,
                'last_name': profile.user.last_name,
                'is_staff': profile.user.is_staff,
                'date_joined': profile.user.date_joined,
                'last_login': profile.user.last_login,
                'all_companies': [{'code': c.code, 'name': c.name} for c in profile.companies.all()]
            })
        return Response(users_data)


class ConventionViewSet(viewsets.ModelViewSet):
    queryset = Convention.objects.all()
    serializer_class = ConventionSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        bank_id = self.request.query_params.get('bank')
        company_id = self.request.query_params.get('company')
        if bank_id:
            queryset = queryset.filter(bank_id=bank_id)
        if company_id:
            queryset = queryset.filter(company_id=company_id)
        return queryset
    
    @action(detail=True, methods=['get'])
    def tax_rules(self, request, pk=None):
        """Get all tax rules for a specific convention"""
        convention = self.get_object()
        tax_rules = convention.tax_rules.all().select_related('payment_class', 'payment_status')
        serializer = TaxRuleSerializer(tax_rules, many=True)
        return Response(serializer.data)


class TaxRuleViewSet(viewsets.ModelViewSet):
    queryset = TaxRule.objects.all()
    serializer_class = TaxRuleSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset().select_related('convention', 'payment_class', 'payment_status')
        convention_id = self.request.query_params.get('convention_id')
        bank_id = self.request.query_params.get('bank')
        company_id = self.request.query_params.get('company')
        if convention_id:
            queryset = queryset.filter(convention_id=convention_id)
        if bank_id:
            queryset = queryset.filter(convention__bank_id=bank_id)
        if company_id:
            queryset = queryset.filter(convention__company_id=company_id)
        return queryset


class ConventionParameterViewSet(viewsets.ModelViewSet):
    queryset = ConventionParameter.objects.all()
    serializer_class = ConventionParameterSerializer


class CustomerTaxRowViewSet(viewsets.ModelViewSet):
    queryset = CustomerTaxRow.objects.all()
    serializer_class = CustomerTaxRowSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset().select_related('transaction')
        transaction_id = self.request.query_params.get('transaction')
        tax_type = self.request.query_params.get('tax_type')
        if transaction_id:
            queryset = queryset.filter(transaction_id=transaction_id)
        if tax_type:
            queryset = queryset.filter(tax_type=tax_type)
        return queryset
    
    @action(detail=False, methods=['delete'])
    def empty(self, request):
        """
        Empty the CustomerTaxRow table using TRUNCATE for maximum speed.
        TRUNCATE is much faster than DELETE as it resets the table in one operation.
        """
        from django.db import connection
        
        # Get count before deletion
        count = CustomerTaxRow.objects.count()
        
        # Get table name from model
        table_name = CustomerTaxRow._meta.db_table
        
        # Use TRUNCATE TABLE for fastest possible operation (empties table in one click)
        # TRUNCATE is faster than DELETE because it's a DDL operation that resets the table
        try:
            with connection.cursor() as cursor:
                # TRUNCATE TABLE is the fastest way to empty a table
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
                method_used = 'TRUNCATE'
        except Exception as e:
            # Fallback to DELETE if TRUNCATE fails (e.g., due to foreign key constraints)
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                method_used = 'DELETE'
        
        return Response({
            'message': f'Emptied CustomerTaxRow table. Deleted {count} rows.',
            'deleted_count': count,
            'method': method_used
        }, status=status.HTTP_200_OK)


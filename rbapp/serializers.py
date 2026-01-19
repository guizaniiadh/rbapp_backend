from rest_framework import serializers
from .models import ConventionParameter

class ConventionParameterSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConventionParameter
        fields = ['id', 'name', 'value']
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import (
    PaymentClass, PaymentStatus, PaymentIdentification, Bank, Agency, 
    BankLedgerEntry, CustomerLedgerEntry, Company, UserProfile, Tax, BankTransaction, CustomerTransaction,
    Convention, TaxRule, CustomerTaxRow, RecoBankTransaction, RecoCustomerTransaction
)
class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = get_user_model()
        fields = ['username', 'email', 'password']

    def create(self, validated_data):
        user = get_user_model().objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        return user



class PaymentClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentClass
        fields = ['code', 'name']


class PaymentStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentStatus
        fields = ['id', 'line', 'name', 'accounting_account', 'payment_class']
        read_only_fields = ['id']  # ID is auto-generated, should not be set during creation


class PaymentIdentificationSerializer(serializers.ModelSerializer):
    bank = serializers.PrimaryKeyRelatedField(queryset=Bank.objects.all())
    line = serializers.IntegerField(read_only=True)  # Read-only for responses

    def validate(self, data):
        description = data.get('description')
        bank = data.get('bank')
        debit = data.get('debit', False)
        credit = data.get('credit', False)

        # Check for duplicate description and bank
        qs = PaymentIdentification.objects.filter(
            description=description,
            bank=bank
        )
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError(
                "A payment identification with this description and bank already exists."
            )

        # Check that debit and credit are not the same
        if debit == credit:
            raise serializers.ValidationError(
                "Debit and credit cannot have the same value. One must be True and the other False."
            )

        return data

    class Meta:
        model = PaymentIdentification
        fields = ['line', 'description', 'payment_status', 'debit', 'credit', 'bank', 'grouped']
        read_only_fields = ['line']  # Explicitly mark line as read-only


class BankSerializer(serializers.ModelSerializer):
    statuses = serializers.PrimaryKeyRelatedField(queryset=PaymentStatus.objects.all(), many=True, required=False)
    logo_url = serializers.SerializerMethodField()
    
    def get_logo_url(self, obj):
        if obj.logo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.logo.url)
            return obj.logo.url
        return None

    class Meta:
        model = Bank
        fields = ['code', 'name', 'address', 'website', 'logo', 'logo_url', 'statuses', 'beginning_balance_label']

class AgencySerializer(serializers.ModelSerializer):
    class Meta:
        model = Agency
        fields = ['code', 'bank', 'name', 'address', 'city']

class BankLedgerEntrySerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = BankLedgerEntry
        fields = ['id', 'agency', 'file', 'uploaded_at', 'name', 'user']

class CustomerLedgerEntrySerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(read_only=True)
    # Preferred: accept company_code as input to avoid clashing with FK field name
    company_code = serializers.CharField(max_length=20, write_only=True, required=True)
    # Expose company_code in responses as well
    company_code_display = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = CustomerLedgerEntry
        fields = ['id', 'user', 'company_code', 'company_code_display', 'file', 'uploaded_at', 'name']
        read_only_fields = ['user']

    def validate_company_code(self, value):
        """Validate that the company code exists"""
        try:
            Company.objects.get(code=value)
        except Company.DoesNotExist:
            raise serializers.ValidationError(f"Company with code '{value}' does not exist.")
        return value

    def get_company_code_display(self, obj):
        """Return the company code for display"""
        try:
            return obj.company.code if obj.company else None
        except Exception:
            return None

    def create(self, validated_data):
        """Remove non-model fields before model create."""
        validated_data.pop('company_code', None)
        return super().create(validated_data)

class UserListSerializer(serializers.ModelSerializer):
    """Serializer for user list view"""
    companies = serializers.SerializerMethodField()
    company_count = serializers.SerializerMethodField()
    
    def get_companies(self, obj):
        try:
            profile = obj.profile
            companies_data = []
            for company in profile.companies.all():
                companies_data.append({
                    'code': company.code,
                    'name': company.name
                })
            return companies_data
        except UserProfile.DoesNotExist:
            return []
    
    
    def get_company_count(self, obj):
        try:
            return obj.profile.companies.count()
        except UserProfile.DoesNotExist:
            return 0
    
    class Meta:
        model = get_user_model()
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'is_active', 'is_staff', 'is_superuser', 'date_joined', 'last_login', 'companies', 'company_count']

class UserDetailSerializer(serializers.ModelSerializer):
    """Serializer for user detail view"""
    companies = serializers.SerializerMethodField()
    company_count = serializers.SerializerMethodField()
    
    def get_companies(self, obj):
        try:
            profile = obj.profile
            companies_data = []
            for company in profile.companies.all():
                companies_data.append({
                    'code': company.code,
                    'name': company.name
                })
            return companies_data
        except UserProfile.DoesNotExist:
            return []
    
    
    def get_company_count(self, obj):
        try:
            return obj.profile.companies.count()
        except UserProfile.DoesNotExist:
            return 0
    
    class Meta:
        model = get_user_model()
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'is_active', 'is_staff', 'is_superuser', 'date_joined', 'last_login', 'companies', 'company_count']

class UserProfileSerializer(serializers.ModelSerializer):
    companies = serializers.PrimaryKeyRelatedField(queryset=Company.objects.all(), many=True, required=False)
    company = serializers.SerializerMethodField()  # For backward compatibility
    companies_info = serializers.SerializerMethodField()  # Detailed company information
    
    def get_company(self, obj):
        """Backward compatibility: return first company"""
        first_company = obj.companies.first()
        return first_company.code if first_company else None
    
    def get_companies_info(self, obj):
        """Return detailed information about all companies"""
        companies = obj.companies.all()
        return [
            {
                'code': company.code,
                'name': company.name
            }
            for company in companies
        ]
    
    class Meta:
        model = UserProfile
        fields = ['id', 'user', 'companies', 'company', 'companies_info']

class CompanyAssignmentSerializer(serializers.Serializer):
    """Serializer for assigning/removing companies from users"""
    company_codes = serializers.ListField(
        child=serializers.CharField(max_length=20),
        allow_empty=False,
        help_text="List of company codes to assign/remove"
    )
    
    def validate_company_codes(self, value):
        """Validate that all company codes exist"""
        existing_companies = Company.objects.filter(code__in=value)
        existing_codes = set(existing_companies.values_list('code', flat=True))
        invalid_codes = set(value) - existing_codes
        
        if invalid_codes:
            raise serializers.ValidationError(f"Invalid company codes: {', '.join(invalid_codes)}")
        
        return value

class BulkCompanyAssignmentSerializer(serializers.Serializer):
    """Serializer for bulk company assignments"""
    user_ids = serializers.ListField(
        child=serializers.IntegerField(),
        allow_empty=False,
        help_text="List of user IDs"
    )
    company_codes = serializers.ListField(
        child=serializers.CharField(max_length=20),
        allow_empty=False,
        help_text="List of company codes to assign"
    )
    action = serializers.ChoiceField(
        choices=['assign', 'remove'],
        help_text="Action to perform: assign or remove companies"
    )
    
    def validate_user_ids(self, value):
        """Validate that all user IDs exist"""
        existing_users = get_user_model().objects.filter(id__in=value)
        existing_ids = set(existing_users.values_list('id', flat=True))
        invalid_ids = set(value) - existing_ids
        
        if invalid_ids:
            raise serializers.ValidationError(f"Invalid user IDs: {', '.join(map(str, invalid_ids))}")
        
        return value
    
    def validate_company_codes(self, value):
        """Validate that all company codes exist"""
        existing_companies = Company.objects.filter(code__in=value)
        existing_codes = set(existing_companies.values_list('code', flat=True))
        invalid_codes = set(value) - existing_codes
        
        if invalid_codes:
            raise serializers.ValidationError(f"Invalid company codes: {', '.join(invalid_codes)}")
        
        return value

class TaxSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tax
        fields = ['id', 'name', 'description', 'accounting_account', 'company', 'bank']
    
    def validate_description(self, value):
        """Validate that description is a list of strings"""
        if not isinstance(value, list):
            raise serializers.ValidationError("Description must be a list of strings")
        
        for item in value:
            if not isinstance(item, str):
                raise serializers.ValidationError("All items in description must be strings")
        
        return value
    
    def validate(self, attrs):
        """Validate that tax name is unique per company/bank combination"""
        name = attrs.get('name')
        company = attrs.get('company')
        bank = attrs.get('bank')
        
        # Get the instance if we're updating (for partial updates)
        instance = self.instance
        
        # If fields are not in attrs, get from instance (for partial updates)
        if name is None and instance:
            name = instance.name
        if company is None and instance:
            company = instance.company
        if bank is None and instance:
            bank = instance.bank
        
        # Only validate if we have all required fields
        if name and company and bank:
            # Check if a tax with the same name, company, and bank already exists
            query = Tax.objects.filter(name=name, company=company, bank=bank)
            
            # If updating, exclude the current instance
            if instance:
                query = query.exclude(pk=instance.pk)
            
            if query.exists():
                raise serializers.ValidationError({
                    'name': f'A tax with the name "{name}" already exists for this company and bank combination.'
                })
        
        return attrs

class BankTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = BankTransaction
        fields = ['id', 'bank_ledger_entry', 'import_batch_id', 'operation_date', 'label', 'value_date', 'debit', 'credit', 'date_ref', 'ref', 'document_reference', 'amount', 'payment_class', 'payment_status', 'accounting_account', 'internal_number', 'type']

class CustomerTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerTransaction
        fields = ['id', 'customer_ledger_entry', 'import_batch_id', 'account_number', 'accounting_date', 'description', 'debit_amount', 'credit_amount', 'external_doc_number', 'due_date', 'payment_type', 'amount', 'total_amount', 'matched_bank_transaction']

class CompanySerializer(serializers.ModelSerializer):
    logo_url = serializers.SerializerMethodField()
    
    def get_logo_url(self, obj):
        if obj.logo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.logo.url)
            return obj.logo.url
        return None
    
    class Meta:
        model = Company
        fields = ['code', 'name', 'logo', 'logo_url']

class CompanyWithUsersSerializer(serializers.ModelSerializer):
    users = serializers.SerializerMethodField()
    user_count = serializers.SerializerMethodField()
    active_user_count = serializers.SerializerMethodField()
    
    def get_users(self, obj):
        """Get all users for this company with their details"""
        user_profiles = obj.users.all()
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
            })
        return users_data
    
    def get_user_count(self, obj):
        """Get total number of users in this company"""
        return obj.users.count()
    
    def get_active_user_count(self, obj):
        """Get number of active users in this company"""
        return obj.users.filter(user__is_active=True).count()
    
    def get_logo_url(self, obj):
        if obj.logo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.logo.url)
            return obj.logo.url
        return None
    
    logo_url = serializers.SerializerMethodField()
    
    class Meta:
        model = Company
        fields = ['code', 'name', 'logo', 'logo_url', 'users', 'user_count', 'active_user_count']

class CompanyStatsSerializer(serializers.ModelSerializer):
    """Serializer for company statistics only"""
    total_users = serializers.SerializerMethodField()
    active_users = serializers.SerializerMethodField()
    inactive_users = serializers.SerializerMethodField()
    staff_users = serializers.SerializerMethodField()
    logo_url = serializers.SerializerMethodField()
    
    def get_total_users(self, obj):
        return obj.users.count()
    
    def get_active_users(self, obj):
        return obj.users.filter(user__is_active=True).count()
    
    def get_inactive_users(self, obj):
        return obj.users.filter(user__is_active=False).count()
    
    def get_staff_users(self, obj):
        return obj.users.filter(user__is_staff=True).count()
    
    def get_logo_url(self, obj):
        if obj.logo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.logo.url)
            return obj.logo.url
        return None
    
    class Meta:
        model = Company
        fields = ['code', 'name', 'logo', 'logo_url', 'total_users', 'active_users', 'inactive_users', 'staff_users']

class ConventionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Convention
        fields = '__all__'


# Serializer for TaxRule
class TaxRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaxRule
        fields = '__all__'


# Serializer for CustomerTaxRow
class CustomerTaxRowSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerTaxRow
        fields = '__all__'

# Reco serializers
class RecoBankTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecoBankTransaction
        fields = ['id', 'bank_ledger_entry', 'import_batch_id', 'operation_date', 'label', 'value_date', 'debit', 'credit', 'date_ref', 'ref', 'document_reference', 'amount', 'payment_class', 'payment_status', 'accounting_account', 'internal_number', 'type', 'bank']

class RecoCustomerTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecoCustomerTransaction
        fields = ['id', 'customer_ledger_entry', 'import_batch_id', 'account_number', 'accounting_date', 'document_number', 'description', 'debit_amount', 'credit_amount', 'external_doc_number', 'due_date', 'payment_type', 'payment_status', 'amount', 'total_amount', 'matched_bank_transaction']


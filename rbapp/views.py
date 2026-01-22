
from decimal import Decimal, ROUND_HALF_UP
from .models import Comparison, RecoBankTransaction, RecoCustomerTransaction
# API to compare customer and bank taxes for matched transactions and populate the Comparison table
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

class TaxComparisonView(APIView):
    def put(self, request):
        """
        For each row in the Comparison table, set customer_total_tax to the sum of tax_amount for all CustomerTaxRow entries with transaction_id == customer_transaction_id.
        """
        from rbapp.models import Comparison, CustomerTaxRow
        from django.db.models import Sum
        updated = 0
        for comp in Comparison.objects.all():
            total_tax = CustomerTaxRow.objects.filter(transaction_id=comp.customer_transaction_id).aggregate(total=Sum('tax_amount'))['total'] or 0
            if comp.customer_total_tax != total_tax:
                comp.customer_total_tax = total_tax
                comp.save(update_fields=['customer_total_tax'])
                updated += 1
        return Response({'message': f'Updated {updated} comparison rows with customer_total_tax.'})
    """
    API endpoint to compare customer and bank taxes for matched transactions and populate the Comparison table.
    POST: Triggers the reconciliation process and returns the results.
    """
    def post(self, request):
        import logging
        logger = logging.getLogger(__name__)
        
        from rbapp.models import RecoCustomerTransaction, CustomerTaxRow, Comparison, RecoBankTransaction
        from django.db.models import Sum
        matched_customers = RecoCustomerTransaction.objects.filter(matched_bank_transaction_id__isnull=False)
        results = []
        count = 0
        
        logger.info(f"TaxComparisonView.post: Processing {matched_customers.count()} matched customer transactions")
        
        for cust_tx in matched_customers:
            if not isinstance(cust_tx, RecoCustomerTransaction):
                continue
            bank_tx = cust_tx.matched_bank_transaction
            if not bank_tx:
                logger.warning(f"TaxComparisonView: Customer transaction {cust_tx.id} has matched_bank_transaction_id={cust_tx.matched_bank_transaction_id} but bank_tx is None")
                continue
            
            logger.debug(f"TaxComparisonView: Processing customer_tx={cust_tx.id}, matched_bank_tx={bank_tx.id}, bank_tx.type={bank_tx.type}, bank_tx.internal_number={bank_tx.internal_number}")
            
            # --- original logic ---
            cust_tax_rows = CustomerTaxRow.objects.filter(transaction=cust_tx)
            for row in cust_tax_rows:
                tax_type = row.tax_type.strip().upper()
                # Calculate customer_tax and bank_tax as before
                customer_tax = row.tax_amount.quantize(Decimal('0.001')) if isinstance(row.tax_amount, Decimal) else round(float(row.tax_amount), 3)
                
                # Find matching bank tax
                bank_tax = None
                if bank_tx.internal_number:
                    logger.debug(f"TaxComparisonView: Looking for bank tax - customer_tx={cust_tx.id}, tax_type={tax_type}, internal_number={bank_tx.internal_number}")
                    
                    # Check all transactions with this internal_number and type
                    bank_tx_candidates = RecoBankTransaction.objects.filter(
                        internal_number=bank_tx.internal_number,
                        type__iexact=tax_type
                    )
                    logger.debug(f"TaxComparisonView: Found {bank_tx_candidates.count()} bank transactions with internal_number={bank_tx.internal_number}, type={tax_type}")
                    
                    if bank_tx_candidates.count() > 0:
                        for candidate in bank_tx_candidates:
                            logger.debug(f"TaxComparisonView: Candidate bank_tx: id={candidate.id}, type={candidate.type}, amount={candidate.amount}, internal_number={candidate.internal_number}")
                    
                    bank_tx_row = bank_tx_candidates.order_by('-id').first()
                    if bank_tx_row:
                        val = bank_tx_row.amount
                        bank_tax = val.quantize(Decimal('0.001')) if isinstance(val, Decimal) else round(float(val), 3)
                        logger.info(f"TaxComparisonView: ✅ Found bank_tax for customer_tx={cust_tx.id}, tax_type={tax_type}: bank_tx_id={bank_tx_row.id}, amount={bank_tax}")
                    else:
                        # Additional investigation: check if agios transactions exist at all with this internal_number
                        if tax_type == 'AGIOS':
                            all_agios_with_internal = RecoBankTransaction.objects.filter(
                                internal_number=bank_tx.internal_number,
                                type__iexact='agios'
                            )
                            all_agios_any_internal = RecoBankTransaction.objects.filter(type__iexact='agios')
                            logger.warning(f"TaxComparisonView: ❌ No bank_tax found for customer_tx={cust_tx.id}, tax_type={tax_type}, internal_number={bank_tx.internal_number}")
                            logger.warning(f"  - Agios transactions with this internal_number: {all_agios_with_internal.count()}")
                            logger.warning(f"  - Total agios transactions in DB: {all_agios_any_internal.count()}")
                            if all_agios_any_internal.count() > 0:
                                sample_agios = all_agios_any_internal[:3]
                                for ag in sample_agios:
                                    logger.warning(f"    Sample agios: id={ag.id}, amount={ag.amount}, internal_number={ag.internal_number}, date_ref={ag.date_ref}")
                        else:
                            logger.warning(f"TaxComparisonView: ❌ No bank_tax found for customer_tx={cust_tx.id}, tax_type={tax_type}, internal_number={bank_tx.internal_number}")
                            bank_tax = Decimal('0.000')
                else:
                    logger.warning(f"TaxComparisonView: ❌ bank_tx.internal_number is None for customer_tx={cust_tx.id}, bank_tx_id={bank_tx.id}, tax_type={tax_type}")
                    bank_tax = Decimal('0.000')
                
                # Ensure bank_tax is not None
                if bank_tax is None:
                    logger.warning(f"TaxComparisonView: bank_tax is None for customer_tx={cust_tx.id}, tax_type={tax_type}, setting to 0.000")
                    bank_tax = Decimal('0.000')
                
                abs_cust_val = abs(Decimal(str(customer_tax))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                abs_bank_val = abs(Decimal(str(bank_tax or 0))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                
                if abs_cust_val == abs_bank_val and abs_cust_val != Decimal('0.000'):
                    status_str = 'matched'
                elif abs_cust_val == Decimal('0.000') and abs_bank_val == Decimal('0.000'):
                    continue
                elif abs_cust_val == Decimal('0.000') or abs_bank_val == Decimal('0.000'):
                    status_str = 'missing'
                    if tax_type == 'AGIOS' and abs_bank_val == Decimal('0.000'):
                        logger.warning(f"TaxComparisonView: ⚠️ MISSING bank_tax for AGIOS - customer_tx={cust_tx.id}, customer_tax={abs_cust_val}, bank_tax={abs_bank_val}, matched_bank_tx_id={bank_tx.id}, internal_number={bank_tx.internal_number}")
                else:
                    status_str = 'mismatch'
                
                obj, created = Comparison.objects.update_or_create(
                    customer_transaction=cust_tx,
                    tax_type=tax_type,
                    defaults={
                        'customer_tax': abs_cust_val,
                        'bank_tax': abs_bank_val,
                        'status': status_str,
                        'customer_total_tax': row.total_tax_amount,
                        'matched_bank_transaction_id': getattr(bank_tx, 'id', None),
                    }
                )
                
                if tax_type == 'AGIOS' and abs_bank_val == Decimal('0.000') and abs_cust_val != Decimal('0.000'):
                    logger.warning(f"TaxComparisonView: Created/Updated Comparison with missing bank_tax: id={obj.id}, customer_tx={cust_tx.id}, tax_type={tax_type}, customer_tax={abs_cust_val}, bank_tax={abs_bank_val}, matched_bank_tx_id={bank_tx.id}")
                results.append({
                    'customer_transaction_id': cust_tx.id,
                    'matched_bank_transaction_id': getattr(bank_tx, 'id', None),
                    'internal_number': getattr(bank_tx, 'internal_number', None),
                    'tax_type': tax_type,
                    'customer_tax': format(abs_cust_val, '.3f'),
                    'bank_tax': format(abs_bank_val, '.3f'),
                    'status': status_str,
                    'customer_total_tax': format(row.total_tax_amount, '.3f') if row.total_tax_amount is not None else None,
                })
                count += 1
        return Response({
            'message': f'Populated {count} comparison rows.',
            'results': results
        }, status=status.HTTP_200_OK)
    
    def delete(self, request):
        """
        Empty the Comparison table using TRUNCATE for maximum speed.
        TRUNCATE is much faster than DELETE as it resets the table in one operation.
        """
        from rbapp.models import Comparison
        from django.db import connection
        
        # Get count before deletion
        count = Comparison.objects.count()
        
        # Get table name from model
        table_name = Comparison._meta.db_table
        
        # Use TRUNCATE TABLE for fastest possible operation (empties table in one click)
        # TRUNCATE is faster than DELETE because it's a DDL operation that resets the table
        try:
            with connection.cursor() as cursor:
                # TRUNCATE TABLE is the fastest way to empty a table
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
        except Exception as e:
            # Fallback to DELETE if TRUNCATE fails (e.g., due to foreign key constraints)
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
        
        return Response({
            'message': f'Emptied Comparison table. Deleted {count} rows.',
            'deleted_count': count,
            'method': 'TRUNCATE' if count > 0 else 'N/A'
        }, status=status.HTTP_200_OK)
from .models import ConventionParameter
from .serializers import ConventionParameterSerializer

from rest_framework import viewsets
import os
import glob
import pandas as pd
from datetime import datetime

class ConventionParameterViewSet(viewsets.ModelViewSet):
    queryset = ConventionParameter.objects.all()
    serializer_class = ConventionParameterSerializer
# TaxRule CRUD API
# (Moved below imports to avoid NameError)
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from .models import (
    PaymentClass, PaymentStatus, PaymentIdentification, Bank, Agency, 
    BankLedgerEntry, CustomerLedgerEntry, BankTransaction, CustomerTransaction, Tax, UserProfile, Company, Convention, TaxRule, CustomerTaxRow,
    RecoBankTransaction, RecoCustomerTransaction
)
from .serializers import (
    PaymentClassSerializer, PaymentStatusSerializer, PaymentIdentificationSerializer,
    BankSerializer, AgencySerializer, BankLedgerEntrySerializer, CustomerLedgerEntrySerializer,
    BankTransactionSerializer, CustomerTransactionSerializer, TaxSerializer, RegisterSerializer, UserProfileSerializer, 
    UserListSerializer, UserDetailSerializer, CompanySerializer, CompanyWithUsersSerializer, CompanyStatsSerializer, ConventionSerializer, TaxRuleSerializer, CustomerTaxRowSerializer,
    RecoBankTransactionSerializer, RecoCustomerTransactionSerializer
)
from django.db.models import Q, Max
import json
from datetime import timedelta
import ast
import pandas as pd
import numpy as np
import os
import re
from decimal import Decimal, ROUND_DOWN
from django.db import transaction as db_transaction
from rest_framework import parsers, serializers
from rest_framework import permissions
from rest_framework.decorators import action
from .serializers import (
    RegisterSerializer, UserProfileSerializer, CompanySerializer
)
from .models import UserProfile, Company
from rest_framework import generics
from django.conf import settings
import shutil
import tempfile
from rapidfuzz import fuzz
from datetime import datetime

# === 1. Score functions ===

# Relaxed date scoring: ±3 days gets high scores, smaller penalty

# --- Holiday-aware business days calculation ---
def get_holiday_dates():
    from .models import ConventionParameter
    import re
    holidays_param = ConventionParameter.objects.filter(name='holidays').first()
    holidays = set()
    if holidays_param and holidays_param.value:
        for entry in holidays_param.value:
            if isinstance(entry, str):
                entry = entry.strip()
                # Range: "YYYY-MM-DD - YYYY-MM-DD"
                m = re.match(r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})", entry)
                if m:
                    start = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                    end = datetime.strptime(m.group(2), "%Y-%m-%d").date()
                    for i in range((end - start).days + 1):
                        holidays.add(start + timedelta(days=i))
                else:
                    # Single date
                    try:
                        holidays.add(datetime.strptime(entry, "%Y-%m-%d").date())
                    except Exception:
                        pass
    return holidays

def business_days_excluding_holidays(start_date, end_date, holidays):
    # Always go from min to max
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    day_count = 0
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5 and curr not in holidays:  # Mon-Fri and not a holiday
            day_count += 1
        curr += timedelta(days=1)
    return day_count

def score_date_tolerance(source_date_str, target_date_str):
    try:
        source_date = datetime.strptime(source_date_str, "%Y-%m-%d").date()
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        holidays = get_holiday_dates()
        business_days = business_days_excluding_holidays(source_date, target_date, holidays)
        day_diff = abs(business_days)

        if day_diff == 0:
            return 100
        elif day_diff == 1:
            return 95
        elif day_diff == 2:
            return 90
        elif day_diff == 3:
            return 85
        elif day_diff <= 5:
            return 75
        elif day_diff <= 8:
            return 60
        else:
            return 0
    except Exception:
        return 0

# Exact match on amount (no tolerance)
def score_amount_exact(amount1, amount2):
    try:
        # Convert to float and compare with sign
        return 100 if float(amount1) == float(amount2) else 0
    except Exception:
        return 0

# Reference match: max between substring and fuzzy partial ratio
def score_reference_fuzzy(ref1, ref2):
    try:
        if pd.isna(ref1) or pd.isna(ref2):
            return 0
            
        ref1_str = str(ref1)
        ref2_str = str(ref2)
        
        ref1_digits = re.sub(r'\D', '', ref1_str)
        ref2_digits = re.sub(r'\D', '', ref2_str)

        len_ref1 = len(ref1_digits)
        min_match_len = int(len_ref1 * 0.7) if len_ref1 else 0
        score_substring = 0

        if len_ref1 and len(ref2_digits) >= min_match_len:
            for i in range(len(ref2_digits) - min_match_len + 1):
                sub = ref2_digits[i:i + min_match_len]
                if sub in ref1_digits:
                    score_substring = 100
                    break

        score_fuzzy = fuzz.partial_ratio(ref1_str, ref2_str)

        return max(score_substring, score_fuzzy)
    except Exception:
        return 0

# Payment type matching
def score_payment_type_fuzzy(payment_type1, payment_type2):
    try:
        if pd.isna(payment_type1) or pd.isna(payment_type2):
            return 0
            
        payment_type1_str = str(payment_type1).strip()
        payment_type2_str = str(payment_type2).strip()
        
        if not payment_type1_str or not payment_type2_str:
            return 0
            
        return fuzz.token_sort_ratio(payment_type1_str, payment_type2_str)
    except Exception:
        return 0

# Helper function to safely convert amount to float, replacing inf/nan with None
def safe_float(value, default=0):
    """
    Safely convert a value to float, replacing inf, -inf, and nan with None.
    """
    import math
    try:
        if value is None:
            return default
        result = float(value)
        if math.isinf(result) or math.isnan(result):
            return None
        return result
    except (ValueError, TypeError):
        return default

# Helper function to sanitize DataFrame for JSON serialization
def sanitize_dataframe_for_json(df):
    """
    Replace inf, -inf, and nan values in DataFrame with None (which becomes null in JSON).
    This prevents JSON serialization errors.
    """
    import numpy as np
    df_clean = df.copy()
    # Replace inf, -inf, and nan with None
    df_clean = df_clean.replace([np.inf, -np.inf, np.nan], None)
    return df_clean

# Document number matching (alphanumeric like "CHP250000001")
def score_document_number_fuzzy(doc_num1, doc_num2):
    try:
        if pd.isna(doc_num1) or pd.isna(doc_num2):
            return 0
            
        doc_num1_str = str(doc_num1).strip()
        doc_num2_str = str(doc_num2).strip()
        
        if not doc_num1_str or not doc_num2_str:
            return 0
            
        # Exact match for alphanumeric document numbers
        if doc_num1_str == doc_num2_str:
            return 100
            
        # Fuzzy matching for similar document numbers
        return fuzz.partial_ratio(doc_num1_str, doc_num2_str)
    except Exception:
        return 0

def clean_bank_dataframe(df):
    column_mapping = {
        'Date Opération': ['Opération', 'Date opération', 'Date', 'DATE OPERATION'],
        'Libellé': ['Intitulé', "Libellé de l'opération", 'Libellé Opération', 'LIBELLE'],
        'Date Valeur': ['Valeur', 'Date de valeur', 'Date valeur', 'DATE VALEUR'],
        'Débit': ['DEBIT (TND)', 'Débit', 'Débit (TND)'],
        'Crédit': ['CREDIT (TND)', 'Crédit', 'Crédit (TND)'],
        'Montant': ['Montant', 'MONTANT', 'Amount'],
        'Sens Opération': ['Sens', 'Sens Opération', 'SENS OPERATION'],
        'Référence': ['Référence', 'Reference', 'REFERENCE'],
        'Référence Dossier': ['REFERENCE DOSSIER']
    }

    reverse_mapping = {}
    for standard_col, aliases in column_mapping.items():
        for alias in aliases:
            reverse_mapping[alias.strip().lower()] = standard_col

    header_row_index = None
    for i in range(len(df)):
        row = df.iloc[i]
        normalized = [str(cell).strip().lower() for cell in row]
        match_count = sum(1 for cell in normalized if cell in reverse_mapping)
        if match_count >= 2:
            header_row_index = i
            break

    if header_row_index is None:
        raise ValueError("Aucune ligne contenant des noms de colonnes valides n'a été trouvée.")

    df = pd.DataFrame(df.values[header_row_index + 1:], columns=df.iloc[header_row_index])
    df = df.dropna(axis=1, how='all')

    df = df.rename(columns={
        col: reverse_mapping[str(col).strip().lower()]
        for col in df.columns
        if str(col).strip().lower() in reverse_mapping
    })

    if 'Montant' in df.columns:
        df['Montant'] = pd.to_numeric(df['Montant'], errors='coerce')

    if 'Sens Opération' in df.columns and 'Montant' in df.columns:
        df['Débit'] = np.where(df['Sens Opération'].str.upper() == 'D', df['Montant'], 0)
        df['Crédit'] = np.where(df['Sens Opération'].str.upper() == 'C', df['Montant'], 0)

    desired_columns = [
        'Date Opération', 'Libellé', 'Date Valeur',
        'Débit', 'Crédit', 'Référence', 'Référence Dossier'
    ]
    df = df[[col for col in desired_columns if col in df.columns]]

    return df.reset_index(drop=True)

def normalize_amount_column(series: pd.Series) -> pd.Series:
    try:
        # Handle NaN values first
        series = series.fillna('')
        
        # Convert to string safely
        cleaned = series.astype(str)
        
        # Apply string operations
        cleaned = cleaned.str.replace(r'\s+', '', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        
        # Convert to numeric, handling errors
        return pd.to_numeric(cleaned, errors='coerce').round(3)
    except Exception as e:
        # If there's an error, try a more robust approach
        try:
            # Convert to numeric directly, handling errors
            return pd.to_numeric(series, errors='coerce').round(3)
        except Exception as e2:
            # If all else fails, return zeros
            return pd.Series([0.0] * len(series))

def extract_largest_number(text):
    if pd.isna(text):
        return None
    matches = re.findall(r'\d{5,}', text)
    if not matches:
        return None
    return max(matches, key=len)

# Helper to validate day and month ranges
def is_valid_date(d, m):
    try:
        day, month = int(d), int(m)
        return 1 <= day <= 31 and 1 <= month <= 12
    except:
        return False

def extract_info(libelle):
    libelle = str(libelle)
    date_ref = None
    used_date_number = None  # to avoid reusing in ref

    # Match format: DD MM YYYY
    m3 = re.search(r'(\d{2})\s+(\d{2})\s+(20\d{2})', libelle)
    if m3 and is_valid_date(m3.group(1), m3.group(2)):
        try:
            from datetime import datetime
            date_ref = datetime.strptime(f"{m3.group(1)} {m3.group(2)} {m3.group(3)}", "%d %m %Y").date()
        except ValueError:
            date_ref = None
        used_date_number = f"{m3.group(1)}{m3.group(2)}{m3.group(3)[-2:]}"
    else:
        # Match format: DD MM YY
        m2 = re.search(r'(\d{2})\s+(\d{2})\s+(\d{2})', libelle)
        if m2 and is_valid_date(m2.group(1), m2.group(2)):
            try:
                from datetime import datetime
                # Assume 20xx for 2-digit years
                year = f"20{m2.group(3)}" if int(m2.group(3)) < 50 else f"19{m2.group(3)}"
                date_ref = datetime.strptime(f"{m2.group(1)} {m2.group(2)} {year}", "%d %m %Y").date()
            except ValueError:
                date_ref = None
            used_date_number = f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
        else:
            # Match compact: DDMMYY
            m1 = re.search(r'(\d{2})(\d{2})(\d{2})', libelle)
            if m1 and is_valid_date(m1.group(1), m1.group(2)):
                try:
                    from datetime import datetime
                    # Assume 20xx for 2-digit years
                    year = f"20{m1.group(3)}" if int(m1.group(3)) < 50 else f"19{m1.group(3)}"
                    date_ref = datetime.strptime(f"{m1.group(1)} {m1.group(2)} {year}", "%d %m %Y").date()
                except ValueError:
                    date_ref = None
                used_date_number = f"{m1.group(1)}{m1.group(2)}{m1.group(3)}"

    # Match reference (7+ digit numbers), excluding date number
    ref_match = re.findall(r'\d{5,}', libelle)
    ref = None
    for r in ref_match:
        if used_date_number is None or used_date_number not in r:
            ref = r
            break

    # Return None if no date_ref or ref found
    return pd.Series([date_ref if date_ref else None, ref if ref else None])

class RegisterAPIView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer




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
            from django.db import transaction
            from rbapp.models import BankLedgerEntry
            
            with transaction.atomic():
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
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    @action(detail=True, methods=['get'])
    def transactions(self, request, pk=None):
        """Return all bank transactions for this bank ledger entry."""
        entry = self.get_object()
        qs = entry.transactions.all().order_by('operation_date', 'id')
        from .serializers import BankTransactionSerializer
        serializer = BankTransactionSerializer(qs, many=True)
        return Response(serializer.data)

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
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    @action(detail=True, methods=['get'])
    def transactions(self, request, pk=None):
        """Return all customer transactions for this ledger entry."""
        entry = self.get_object()
        qs = entry.transactions.all().order_by('accounting_date', 'id')
        from .serializers import CustomerTransactionSerializer
        serializer = CustomerTransactionSerializer(qs, many=True)
        return Response(serializer.data)

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
    queryset = User.objects.all()
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return UserListSerializer
        return UserDetailSerializer
    
    def list(self, request, *args, **kwargs):
        """Get all users with their company information"""
        users = User.objects.all()
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
            
            # Set as primary company if it's the first one or if requested
            
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
        """
        from django.db import connection
        
        # Get count before deletion
        count = RecoBankTransaction.objects.count()
        
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
            # Fallback to DELETE if TRUNCATE fails (e.g., due to foreign key constraints)
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                method_used = 'DELETE'
        return Response({
            'message': f'Emptied RecoBankTransaction table. Deleted {count} rows.',
            'deleted_count': count,
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
    
    @action(detail=True, methods=['post'])
    def assign_user(self, request, pk=None):
        """Assign a user to this company"""
        company = self.get_object()
        user_id = request.data.get('user_id')
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            from django.contrib.auth.models import User
            user = User.objects.get(id=user_id)
            user_profile, created = UserProfile.objects.get_or_create(user=user)
            
            # Add company to user's companies
            user_profile.companies.add(company)
            
            # Set as primary if it's the first company or if requested
            
            return Response({
                'message': f'User {user.username} assigned to company {company.name}',
                'user_id': user.id,
                'username': user.username,
                'company_code': company.code,
                'company_name': company.name,
            })
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['post'])
    def remove_user(self, request, pk=None):
        """Remove a user from this company"""
        company = self.get_object()
        user_id = request.data.get('user_id')
        
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            from django.contrib.auth.models import User
            user = User.objects.get(id=user_id)
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
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User profile not found'}, status=status.HTTP_404_NOT_FOUND)

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

class PreprocessBankLedgerEntryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        try:
            ledger_entry = BankLedgerEntry.objects.get(pk=pk)
            file_path = ledger_entry.file.path
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.xlsx':
                df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
            elif ext == '.xls':
                df_raw = pd.read_excel(file_path, header=None, engine='xlrd')
            else:
                return Response({"error": "Unsupported file extension"}, status=400)

            # Clean the dataframe
            df_cleaned = clean_bank_dataframe(df_raw)
            
            # Make sure 'Libellé' column is string
            if 'Libellé' in df_cleaned.columns:
                df_cleaned['Libellé'] = df_cleaned['Libellé'].astype(str)
            
            # Normalize columns
            target_columns = ['Date Opération', 'Libellé', 'Date Valeur', 'Débit', 'Crédit']
            df_cleaned = df_cleaned[[col for col in target_columns if col in df_cleaned.columns]]
            
            # Normalize amounts
            if 'Crédit' in df_cleaned.columns:
                df_cleaned['Crédit'] = normalize_amount_column(df_cleaned['Crédit'])
            if 'Débit' in df_cleaned.columns:
                df_cleaned['Débit'] = normalize_amount_column(df_cleaned['Débit'])
            
            # Normalize dates
            if 'Date Opération' in df_cleaned.columns:
                df_cleaned['Date Opération'] = pd.to_datetime(df_cleaned['Date Opération'], dayfirst=True, errors='coerce')
            if 'Date Valeur' in df_cleaned.columns:
                df_cleaned['Date Valeur'] = pd.to_datetime(df_cleaned['Date Valeur'], dayfirst=True, errors='coerce')
            
            # Fill NaN values
            df_cleaned['Crédit'] = df_cleaned['Crédit'].fillna(0)
            df_cleaned['Débit'] = df_cleaned['Débit'].fillna(0)
            
            # Extract references and dates using the new function
            if 'Libellé' in df_cleaned.columns:
                extracted = df_cleaned['Libellé'].apply(extract_info)
                extracted.columns = ['date_ref', 'ref']
                df_cleaned = pd.concat([df_cleaned, extracted], axis=1)
            
            # Get the next import batch id (on reco table)
            last_batch = RecoBankTransaction.objects.aggregate(max_id=Max('import_batch_id'))['max_id'] or 0
            import_batch_id = last_batch + 1

            # Get the bank from the agency linked to the ledger entry
            bank = ledger_entry.agency.bank

            # Save transactions to reco database
            created_transactions = []
            for _, row in df_cleaned.iterrows():
                credit = row.get('Crédit', 0) or 0
                debit = row.get('Débit', 0) or 0
                amount = credit - debit

                transaction = RecoBankTransaction(
                    bank_ledger_entry=ledger_entry,
                    bank=bank,  # <-- Automatically set the bank here
                    import_batch_id=import_batch_id,
                    operation_date=row.get('Date Opération'),
                    label=row.get('Libellé', ''),
                    value_date=row.get('Date Valeur'),
                    debit=debit if debit != 0 else None,
                    credit=credit if credit != 0 else None,
                    date_ref=row.get('date_ref'),
                    ref=row.get('ref'),
                    document_reference='',
                    amount=amount
                )
                created_transactions.append(transaction)

            RecoBankTransaction.objects.bulk_create(created_transactions)

            return Response({
                "message": f"Successfully processed {len(created_transactions)} transactions",
                "import_batch_id": import_batch_id,
                "transactions_count": len(created_transactions)
            })

        except BankLedgerEntry.DoesNotExist:
            return Response({"error": "Bank ledger entry not found"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

class ExtractBeginningBalanceView(APIView):
    """
    View to extract beginning balance from RecoBankTransaction using the bank's configured beginning_balance_label
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk=None):
        try:
            # Get bank_ledger_entry_id from URL path (pk), query params, or use latest
            bank_ledger_entry_id = pk or request.query_params.get('bank_ledger_entry_id')
            
            if bank_ledger_entry_id:
                try:
                    ledger_entry = BankLedgerEntry.objects.get(pk=bank_ledger_entry_id)
                except BankLedgerEntry.DoesNotExist:
                    return Response({"error": "Bank ledger entry not found"}, status=404)
                transactions = RecoBankTransaction.objects.filter(bank_ledger_entry=ledger_entry)
                selected_ledger_entry = ledger_entry
            else:
                # Use latest bank ledger entry
                latest_ledger_entry = BankLedgerEntry.objects.order_by('-id').first()
                if not latest_ledger_entry:
                    return Response({"error": "No bank ledger entries found"}, status=404)
                transactions = RecoBankTransaction.objects.filter(bank_ledger_entry=latest_ledger_entry)
                selected_ledger_entry = latest_ledger_entry
            
            # Get the bank from the ledger entry's agency
            bank = selected_ledger_entry.agency.bank
            
            # Get the beginning balance label from the bank configuration
            beginning_balance_label = bank.beginning_balance_label
            
            if not beginning_balance_label:
                return Response({
                    "error": "Beginning balance label not configured",
                    "message": f"Please configure 'beginning_balance_label' for bank '{bank.name}' in the parameters page",
                    "bank_code": bank.code,
                    "bank_name": bank.name
                }, status=400)
            
            # Search for transaction with label containing the configured label (case-insensitive)
            beginning_balance_tx = transactions.filter(
                label__icontains=beginning_balance_label
            ).first()
            
            if not beginning_balance_tx:
                return Response({
                    "error": "Beginning balance transaction not found",
                    "message": f"No transaction with label containing '{beginning_balance_label}' found",
                    "bank_ledger_entry_id": selected_ledger_entry.id,
                    "search_label": beginning_balance_label
                }, status=404)
            
            # Extract debit value, or use amount if debit is null/zero
            beginning_balance = None
            if beginning_balance_tx.debit and beginning_balance_tx.debit != 0:
                beginning_balance = float(beginning_balance_tx.debit)
            elif beginning_balance_tx.amount:
                beginning_balance = float(beginning_balance_tx.amount)
            else:
                return Response({
                    "error": "Beginning balance value not found",
                    "message": "Transaction found but debit and amount are both null/zero"
                }, status=404)
            
            return Response({
                "beginning_balance": beginning_balance,
                "transaction_id": beginning_balance_tx.id,
                "transaction_label": beginning_balance_tx.label,
                "operation_date": beginning_balance_tx.operation_date.isoformat() if beginning_balance_tx.operation_date else None,
                "bank_ledger_entry_id": beginning_balance_tx.bank_ledger_entry.id,
                "bank_code": bank.code,
                "bank_name": bank.name,
                "search_label_used": beginning_balance_label,
                "source": "debit" if beginning_balance_tx.debit and beginning_balance_tx.debit != 0 else "amount"
            }, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)

def clean_customer_accounting_dataframe(df_comptable):
    """
    Clean customer accounting dataframe by finding the header row and extracting relevant columns.
    """
    # List of columns to keep (strip to clean accidental spaces)
    columns_to_keep = [
        'N° compte bancaire',
        'Date comptabilisation',
        'N° document',
        'Description',
        'Montant débit',
        'Montant crédit',
        'N° doc. externe',
        'Date d\'échéance',
        'Type de règlement'
    ]
    columns_to_keep_clean = [col.strip() for col in columns_to_keep]

    # Find the index of the row that contains the columns
    header_row_index = None
    for i, row in df_comptable.iterrows():
        row_cleaned = [str(cell).strip() for cell in row]
        match_count = sum(1 for cell in row_cleaned if cell in columns_to_keep_clean)
        print(f"Row {i}: {row_cleaned[:5]}... (matches: {match_count})")
        if match_count >= 3:  # Detection threshold
            header_row_index = i
            print(f"Found header row at index {i}")
            break

    if header_row_index is None:
        raise ValueError("Expected columns not found in the file.")

    # Reload dataframe from header row
    df = pd.DataFrame(df_comptable.values[header_row_index + 1:], columns=df_comptable.iloc[header_row_index])
    
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Keep only columns present in columns_to_keep
    available_columns = [col for col in columns_to_keep_clean if col in df.columns]
    print(f"Available columns in dataframe: {list(df.columns)}")
    print(f"Columns to keep: {columns_to_keep_clean}")
    print(f"Columns that will be kept: {available_columns}")
    
    df = df[available_columns]

    return df.reset_index(drop=True)

class PreprocessCustomerLedgerEntryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        try:
            ledger_entry = CustomerLedgerEntry.objects.get(pk=pk)
            file_path = ledger_entry.file.path
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.xlsx':
                df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
            elif ext == '.xls':
                df_raw = pd.read_excel(file_path, header=None, engine='xlrd')
            else:
                return Response({"error": "Unsupported file extension"}, status=400)

            # Clean the dataframe
            df_cleaned = clean_customer_accounting_dataframe(df_raw)
            
            # Debug: Print found columns
            print(f"Found columns in Excel file: {list(df_cleaned.columns)}")
            expected_columns = ['N° compte bancaire', 'Date comptabilisation', 'N° document', 'Description', 'Montant débit', 'Montant crédit', 'N° doc. externe', 'Date d\'échéance', 'Type de règlement']
            print(f"Expected columns: {expected_columns}")
            
            # Check if N° document column exists
            if 'N° document' not in df_cleaned.columns:
                print(f"WARNING: 'N° document' column not found! Available columns: {list(df_cleaned.columns)}")
                # Try to find similar column names
                for col in df_cleaned.columns:
                    if 'document' in col.lower() or 'doc' in col.lower():
                        print(f"Found similar column: {col}")
            
            # Get the agency code from request data or use a default
            # Try to get from JSON data first, then from form data, then from raw data
            agency_code = None
            
            # Try JSON data
            if request.content_type == 'application/json':
                agency_code = request.data.get('agency_code') or request.data.get('code')
            # Try form data
            elif request.content_type == 'application/x-www-form-urlencoded':
                agency_code = request.POST.get('agency_code') or request.POST.get('code')
            # Try raw data (text/plain)
            elif request.content_type == 'text/plain':
                try:
                    import json
                    raw_data = request.body.decode('utf-8')
                    parsed_data = json.loads(raw_data)
                    agency_code = parsed_data.get('agency_code') or parsed_data.get('code')
                except:
                    # If JSON parsing fails, try to extract from raw text
                    raw_data = request.body.decode('utf-8')
                    if '"code"' in raw_data or '"agency_code"' in raw_data:
                        # Try to extract JSON-like content
                        import re
                        code_match = re.search(r'"code"\s*:\s*"([^"]+)"', raw_data)
                        agency_match = re.search(r'"agency_code"\s*:\s*"([^"]+)"', raw_data)
                        if code_match:
                            agency_code = code_match.group(1)
                        elif agency_match:
                            agency_code = agency_match.group(1)
            
            # Use default if no agency code provided
            if not agency_code:
                agency_code = '53-5'  # Default agency code
            
            # Filter rows by agency code (N° compte bancaire)
            if 'N° compte bancaire' in df_cleaned.columns:
                # Filter to only include rows where account number matches agency code
                df_filtered = df_cleaned[df_cleaned['N° compte bancaire'] == agency_code].copy()
                
                if df_filtered.empty:
                    return Response({
                        "error": f"No transactions found for agency code '{agency_code}'",
                        "agency_code": agency_code,
                        "total_rows_before_filter": len(df_cleaned)
                    }, status=404)
                
                df_cleaned = df_filtered
            else:
                return Response({"error": "Column 'N° compte bancaire' not found in the file"}, status=400)
            
            # Rename columns to match our model
            df_final = df_cleaned.rename(columns={
                'Montant débit': 'Crédit',
                'Montant crédit': 'Débit'
            })
            
            # Normalize amount columns
            if 'Crédit' in df_final.columns:
                df_final['Crédit'] = normalize_amount_column(df_final['Crédit'])
            if 'Débit' in df_final.columns:
                df_final['Débit'] = normalize_amount_column(df_final['Débit'])
            
            # Normalize dates
            if 'Date comptabilisation' in df_final.columns:
                df_final['Date comptabilisation'] = pd.to_datetime(df_final['Date comptabilisation'], dayfirst=True, errors='coerce').dt.date
            if 'Date d\'échéance' in df_final.columns:
                df_final['Date d\'échéance'] = pd.to_datetime(df_final['Date d\'échéance'], dayfirst=True, errors='coerce').dt.date
            
            # Fill NaN values
            df_final['Crédit'] = df_final['Crédit'].fillna(0)
            df_final['Débit'] = df_final['Débit'].fillna(0)
            
            # Get the next import batch id (on reco table)
            last_batch = RecoCustomerTransaction.objects.aggregate(max_id=Max('import_batch_id'))['max_id'] or 0
            import_batch_id = last_batch + 1
            
            # Save transactions to reco database
            created_transactions = []
            for _, row in df_final.iterrows():
                # Calculate amount: negative for debit, positive for credit
                debit = row.get('Débit', 0) or 0
                credit = row.get('Crédit', 0) or 0
                
                if debit > 0:
                    amount = -debit  # Negative for debit transactions
                elif credit > 0:
                    amount = credit   # Positive for credit transactions
                else:
                    amount = 0       # Zero if both are zero
                
                # Handle date conversion safely
                accounting_date = row.get('Date comptabilisation')
                if pd.isna(accounting_date):
                    accounting_date = None
                
                due_date = row.get('Date d\'échéance')
                if pd.isna(due_date):
                    due_date = None
                
                # Clean payment type by removing prefix pattern and city names
                payment_type = row.get('Type de règlement', '')
                if payment_type:
                    # Convert to string if it's not already
                    payment_type = str(payment_type)
                    
                    # Remove pattern like "09- ", "01- ", etc.
                    import re
                    payment_type = re.sub(r'^\d{2}-\s*', '', payment_type)
                    # Remove "TUNIS" or "ZARZIS" if found
                    payment_type = re.sub(r'\b(TUNIS|ZARZIS)\b', '', payment_type, flags=re.IGNORECASE)
                    # Replace "TRAITES" with "EFFETS"
                    payment_type = re.sub(r'\bTRAITES\b', 'EFFETS', payment_type, flags=re.IGNORECASE)
                    # Replace "EFFETS CLT" with "CLT EFFET"
                    payment_type = re.sub(r'\bEFFETS CLT\b', 'CLT EFFET', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bVIREMENTS FRS\b', 'FRS VIR', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bVIREMENTS CLT\b', 'CLT VIR', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bEFFETS FRS\b', 'FRS EFFET', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bCHEQUES CLT\b', 'CLT CHQ', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bCHEQUES FRS\b', 'FRS CHQ', payment_type, flags=re.IGNORECASE)

                    # Clean up extra spaces
                    payment_type = re.sub(r'\s+', ' ', payment_type).strip()
                
                transaction = RecoCustomerTransaction(
                    customer_ledger_entry=ledger_entry,
                    import_batch_id=import_batch_id,
                    account_number=str(row.get('N° compte bancaire', '')),
                    accounting_date=accounting_date,
                    document_number=str(row.get('N° document', '')),
                    description=str(row.get('Description', '')),
                    debit_amount=debit if debit != 0 else 0,
                    credit_amount=credit if credit != 0 else 0,
                    external_doc_number=str(row.get('N° doc. externe', '')),
                    due_date=due_date,
                    payment_type=payment_type,  # Use the cleaned payment_type variable
                    amount=amount
                )
                created_transactions.append(transaction)
            
            # Bulk create transactions
            RecoCustomerTransaction.objects.bulk_create(created_transactions)
            
            # Calculate total_amount for all transactions in this batch
            from django.db.models import Sum
            
            # Get unique combinations of document_number, description, and accounting_date (reco)
            unique_groups = set()
            for transaction in created_transactions:
                group_key = (transaction.document_number, transaction.description, transaction.accounting_date)
                unique_groups.add(group_key)
            
            # Calculate and update total_amount for each unique group
            for document_number, description, accounting_date in unique_groups:
                total_sum = RecoCustomerTransaction.objects.filter(
                    document_number=document_number,
                    description=description,
                    accounting_date=accounting_date
                ).aggregate(total=Sum('amount'))['total'] or 0
                
                # Update the total_amount for all transactions in this group
                RecoCustomerTransaction.objects.filter(
                    document_number=document_number,
                    description=description,
                    accounting_date=accounting_date
                ).update(total_amount=total_sum)
            
            return Response({
                "message": f"Successfully processed {len(created_transactions)} customer transactions for agency code '{agency_code}'",
                "import_batch_id": import_batch_id,
                "transactions_count": len(created_transactions),
                "agency_code": agency_code,
                "total_rows_before_filter": len(df_cleaned),
                "filtered_rows": len(df_final)
            })
            
        except CustomerLedgerEntry.DoesNotExist:
            return Response({"error": "Customer ledger entry not found"}, status=404)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)

class GetMatchingResultsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            import os
            import pandas as pd
            from datetime import datetime
            
            # Get the output directory
            output_dir = os.path.join(settings.BASE_DIR, 'matching_results')
            
            if not os.path.exists(output_dir):
                return Response({
                    "error": "No matching results found. Run the matching endpoint first."
                }, status=404)
            
            # Get all CSV files in the directory
            csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            
            if not csv_files:
                return Response({
                    "error": "No matching result files found."
                }, status=404)
            
            # Sort files by timestamp (newest first)
            csv_files.sort(reverse=True)
            
            # Get the most recent files
            latest_files = {}
            for file in csv_files:
                if 'all_matches' in file:
                    latest_files['all_matches'] = file
                elif 'high_matches' in file:
                    latest_files['high_matches'] = file
                elif 'low_matches' in file:
                    latest_files['low_matches'] = file
            
            # Check if "one to many" filter is requested
            one_to_many = request.query_params.get('one_to_many', '').lower() == 'true'
            
            # Read the dataframes
            results = {}
            for file_type, filename in latest_files.items():
                file_path = os.path.join(output_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    
                    # If one_to_many is requested, sort rows to show one-to-many relationships at the top
                    if one_to_many and 'bank_transaction_id' in df.columns:
                        # Count occurrences of each bank_transaction_id
                        bank_transaction_counts = df['bank_transaction_id'].value_counts()
                        # Create a column indicating if it's a one-to-many relationship (count > 1)
                        df['is_one_to_many'] = df['bank_transaction_id'].map(bank_transaction_counts) > 1
                        # Sort: one-to-many rows first (True before False), then by bank_transaction_id
                        df = df.sort_values(['is_one_to_many', 'bank_transaction_id'], ascending=[False, True])
                        # Remove the temporary column
                        df = df.drop(columns=['is_one_to_many'])
                    
                    results[file_type] = {
                        "filename": filename,
                        "file_path": file_path,
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "data": df.to_dict('records')
                    }
                except Exception as e:
                    results[file_type] = {
                        "error": f"Could not read {filename}: {str(e)}"
                    }
            
            return Response({
                "message": "Retrieved saved matching results",
                "output_directory": output_dir,
                "available_files": csv_files,
                "results": results
            }, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)

class MatchCustomerBankTransactionsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Check if any PaymentIdentification is grouped
            grouped_mode = PaymentIdentification.objects.filter(grouped=True).exists()

            # === 1. Filter bank transactions with conditions ===
            # Debug: Check what's in the database
            all_bank_transactions = RecoBankTransaction.objects.all()
            all_payment_classes = PaymentClass.objects.all()
            
            # Get unique types and payment classes for debugging
            unique_types = RecoBankTransaction.objects.values_list('type', flat=True).distinct()
            unique_payment_classes = PaymentClass.objects.values_list('code', flat=True).distinct()
            
            # Get all payment_class codes from PaymentClass table
            payment_class_codes = list(PaymentClass.objects.values_list('code', flat=True))

            # Determine last imported ledger entries
            last_ble = BankLedgerEntry.objects.order_by('-uploaded_at', '-id').first()
            last_cle = CustomerLedgerEntry.objects.order_by('-uploaded_at', '-id').first()

            if last_ble is None or last_cle is None:
                return Response({
                    "error": "No ledger entries found to scope matching",
                    "debug_info": {
                        "last_bank_ledger_entry": None if last_ble is None else last_ble.id,
                        "last_customer_ledger_entry": None if last_cle is None else last_cle.id,
                    }
                }, status=404)

            # Filter bank transactions: only those from last BLE, with existing criteria
            filtered_bank_transactions = RecoBankTransaction.objects.filter(
                bank_ledger_entry=last_ble,
                type='origine',
                payment_class__code__in=payment_class_codes
            )
            
            if not filtered_bank_transactions.exists():
                return Response({
                    "error": "No bank transactions found matching the criteria (type='origine')",
                    "debug_info": {
                        "total_bank_transactions": all_bank_transactions.count(),
                        "total_payment_classes": all_payment_classes.count(),
                        "available_types": list(unique_types),
                        "available_payment_classes": list(unique_payment_classes),
                        "transactions_with_type_origine": RecoBankTransaction.objects.filter(type='origine').count(),
                        "bank_transactions": RecoBankTransaction.objects.count()
                    }
                }, status=404)
            
            # Get customer transactions only from last CLE
            customer_transactions = RecoCustomerTransaction.objects.filter(customer_ledger_entry=last_cle)
            
            if not customer_transactions.exists():
                return Response({
                    "error": "No customer transactions found"
                }, status=404)
            
            # === 2. Convert to DataFrames for easier processing ===
            # Convert bank transactions to DataFrame
            bank_data = []
            for bt in filtered_bank_transactions:
                bank_data.append({
                    'id': bt.id,
                    'operation_date': bt.operation_date.strftime('%Y-%m-%d') if bt.operation_date else None,
                    'amount': safe_float(bt.amount, 0),
                    'ref': bt.ref if bt.ref else '',
                    'payment_class_id': bt.payment_class.code if bt.payment_class else '',
                    'label': bt.label if bt.label else ''
                })
            
            df_bank = pd.DataFrame(bank_data)
            
            # Convert customer transactions to DataFrame
            customer_data = []
            for ct in customer_transactions:
                customer_data.append({
                    'id': ct.id,
                    'accounting_date': ct.accounting_date.strftime('%Y-%m-%d') if ct.accounting_date else None,
                    'amount': safe_float(ct.amount, 0),
                    'total_amount': safe_float(ct.total_amount, 0),
                    'document_number': ct.document_number if ct.document_number else '',
                    'external_doc_number': ct.external_doc_number if ct.external_doc_number else '',
                    'payment_type': ct.payment_type if ct.payment_type else '',
                    'description': ct.description if ct.description else ''
                })
            
            df_customer = pd.DataFrame(customer_data)
            
            # === 3. Matching loop ===
            matches = []
            
            for i, bank_row in df_bank.iterrows():
                best_score = -1
                best_match_idx = None
                best_match_data = None
                
                for j, customer_row in df_customer.iterrows():
                    bank_payment_class = bank_row['payment_class_id']
                    customer_payment_type = customer_row['payment_type']
                    if bank_payment_class != customer_payment_type:
                        continue

                    score_date = score_date_tolerance(
                        customer_row['accounting_date'],
                        bank_row['operation_date']
                    )

                    # Use total_amount if grouped_mode, else use amount
                    if grouped_mode:
                        score_amount = score_amount_exact(
                            customer_row['total_amount'],
                            bank_row['amount']
                        )
                        score_reference = 0  # Remove reference score
                        total_score = 0.25 * score_date + 0.6 * score_amount
                    else:
                        score_amount = score_amount_exact(
                            customer_row['amount'],
                            bank_row['amount']
                        )
                        score_reference = score_reference_fuzzy(
                            customer_row['external_doc_number'],
                            bank_row['ref']
                        )
                        total_score = (
                            0.25 * score_date +
                            0.6 * score_amount +
                            0.15 * score_reference
                        )

                    if total_score > best_score:
                        best_score = total_score
                        best_match_idx = j
                        best_match_data = customer_row.to_dict()
                
                matches.append({
                    'bank_transaction_id': bank_row['id'],
                    'bank_operation_date': bank_row['operation_date'],
                    'bank_amount': bank_row['amount'],
                    'bank_ref': bank_row['ref'],
                    'bank_payment_class': bank_row['payment_class_id'],
                    'bank_label': bank_row['label'],
                    'customer_transaction_id': best_match_data['id'] if best_match_data else None,
                    'customer_accounting_date': best_match_data['accounting_date'] if best_match_data else None,
                    'customer_amount': best_match_data['amount'] if best_match_data else None,
                    'customer_total_amount': best_match_data['total_amount'] if best_match_data else None,
                    'customer_document_number': best_match_data['document_number'] if best_match_data else None,
                    'customer_external_doc_number': best_match_data['external_doc_number'] if best_match_data else None,
                    'customer_payment_type': best_match_data['payment_type'] if best_match_data else None,
                    'customer_description': best_match_data['description'] if best_match_data else None,
                    'score': best_score
                })
            
            # === 4. Create results DataFrame ===
            df_matches = pd.DataFrame(matches)
            
            # === 5. Separate high and low confidence matches ===
            df_matches_high = df_matches[df_matches['score'] >= 68].copy()
            df_matches_low = df_matches[df_matches['score'] < 68].copy()
            
            # === 5.5. Update matched_bank_transaction for high-confidence matches ===
            for _, match_row in df_matches_high.iterrows():
                if match_row['customer_transaction_id'] and match_row['bank_transaction_id']:
                    try:
                        customer_transaction = RecoCustomerTransaction.objects.get(id=match_row['customer_transaction_id'])
                        bank_transaction = RecoBankTransaction.objects.get(id=match_row['bank_transaction_id'])
                        customer_transaction.matched_bank_transaction = bank_transaction
                        customer_transaction.save()
                    except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                        continue
            
            # === 6. Save dataframes to files for later access ===
            import os
            from datetime import datetime
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(settings.BASE_DIR, 'matching_results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save high matches dataframe
            high_matches_file = os.path.join(output_dir, f'high_matches_{timestamp}.csv')
            if not df_matches_high.empty:
                df_matches_high.to_csv(high_matches_file, index=False)
                high_matches_saved = True
                high_matches_path = high_matches_file
            else:
                high_matches_saved = False
                high_matches_path = None
            
            # Save low matches dataframe
            low_matches_file = os.path.join(output_dir, f'low_matches_{timestamp}.csv')
            if not df_matches_low.empty:
                df_matches_low.to_csv(low_matches_file, index=False)
                low_matches_saved = True
                low_matches_path = low_matches_file
            else:
                low_matches_saved = False
                low_matches_path = None
            
            # Save all matches dataframe
            all_matches_file = os.path.join(output_dir, f'all_matches_{timestamp}.csv')
            df_matches.to_csv(all_matches_file, index=False)

            # === 6.5. Promote low matches to high if grouped_mode is True ===
            if grouped_mode:
                rows_to_drop = []
                for idx_low, row_low in df_matches_low.iterrows():
                    # Find matching high match by document number
                    match = df_matches_high[
                        (df_matches_high['customer_document_number'] == row_low['customer_document_number'])
                    ]
                    if not match.empty:
                        matched_bank_id = match.iloc[0]['bank_transaction_id']
                        customer_transaction_id = row_low['customer_transaction_id']
                        
                        # Update ALL CustomerTransactions with the same document_number, description, and accounting_date
                        try:
                            ct = RecoCustomerTransaction.objects.get(id=customer_transaction_id)
                            bt = RecoBankTransaction.objects.get(id=matched_bank_id)
                            matching_customer_transactions = RecoCustomerTransaction.objects.filter(
                                document_number=ct.document_number,
                                description=ct.description,
                                accounting_date=ct.accounting_date
                            )
                            for matching_ct in matching_customer_transactions:
                                matching_ct.matched_bank_transaction = bt
                                matching_ct.save()
                        except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                            continue
                        rows_to_drop.append(idx_low)
                # Remove matched rows from df_matches_low
                df_matches_low_updated = df_matches_low.drop(rows_to_drop)
                # Save the updated low matches file
                low_matches_file_updated = os.path.join(output_dir, f'low_matches_{timestamp}_updated.csv')
                df_matches_low_updated.to_csv(low_matches_file_updated, index=False)
            
            # === 7. Calculate statistics ===
            total_rows = len(df_bank)
            high_match_count = len(df_matches_high)
            low_match_count = len(df_matches_low)
            
            high_match_percentage = (100 * high_match_count / total_rows) if total_rows > 0 else 0
            low_match_percentage = (100 * low_match_count / total_rows) if total_rows > 0 else 0
            
            # === 8. Prepare response data ===
            # Sanitize DataFrames to replace inf/nan values before JSON serialization
            df_matches_high_clean = sanitize_dataframe_for_json(df_matches_high) if not df_matches_high.empty else df_matches_high
            df_matches_low_clean = sanitize_dataframe_for_json(df_matches_low) if not df_matches_low.empty else df_matches_low
            
            response_data = {
                "summary": {
                    "total_bank_transactions": total_rows,
                    "high_matches_count": high_match_count,
                    "high_matches_percentage": round(high_match_percentage, 2),
                    "low_matches_count": low_match_count,
                    "low_matches_percentage": round(low_match_percentage, 2)
                },
                "saved_files": {
                    "all_matches_csv": all_matches_file,
                    "high_matches_csv": high_matches_path if high_matches_saved else None,
                    "low_matches_csv": low_matches_path if low_matches_saved else None,
                    "output_directory": output_dir
                },
                "dataframe_info": {
                    "high_matches_shape": df_matches_high.shape if not df_matches_high.empty else (0, 0),
                    "low_matches_shape": df_matches_low.shape if not df_matches_low.empty else (0, 0),
                    "all_matches_shape": df_matches.shape
                },
                "high_matches": df_matches_high_clean.to_dict('records') if not df_matches_high.empty else [],
                "low_matches": df_matches_low_clean.to_dict('records') if not df_matches_low.empty else []
            }
            
            return Response(response_data, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)


class ManualMatchRecoCustomerBankTransactionView(APIView):
    """
    Manually link a RecoCustomerTransaction to a RecoBankTransaction.

    POST body:
    {
        "reco_bank_transaction_id": <int>,
        "reco_customer_transaction_id": <int>
    }

    Effect:
    - Sets RecoCustomerTransaction.matched_bank_transaction to the given RecoBankTransaction.
    - Optionally, propagates the same matched bank transaction (and payment metadata) to all
      RecoCustomerTransaction rows with the same document_number as the selected one.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            bank_tx_id = request.data.get("reco_bank_transaction_id")
            cust_tx_id = request.data.get("reco_customer_transaction_id")

            # Basic validation
            if bank_tx_id is None or cust_tx_id is None:
                return Response(
                    {
                        "error": "reco_bank_transaction_id and reco_customer_transaction_id are required"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                bank_tx_id = int(bank_tx_id)
                cust_tx_id = int(cust_tx_id)
            except (TypeError, ValueError):
                return Response(
                    {
                        "error": "reco_bank_transaction_id and reco_customer_transaction_id must be integers"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Fetch objects
            try:
                bank_tx = RecoBankTransaction.objects.get(id=bank_tx_id)
            except RecoBankTransaction.DoesNotExist:
                return Response(
                    {"error": f"RecoBankTransaction with id={bank_tx_id} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            try:
                cust_tx = RecoCustomerTransaction.objects.get(id=cust_tx_id)
            except RecoCustomerTransaction.DoesNotExist:
                return Response(
                    {"error": f"RecoCustomerTransaction with id={cust_tx_id} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Link the selected customer transaction to the bank transaction
            cust_tx.matched_bank_transaction = bank_tx
            cust_tx.save()

            # Propagate to all customer transactions with the same document_number (if any)
            propagated_count = 0
            if cust_tx.document_number:
                related_qs = RecoCustomerTransaction.objects.filter(
                    document_number=cust_tx.document_number
                ).exclude(id=cust_tx.id)

                for related in related_qs:
                    updated = False
                    if related.matched_bank_transaction_id != bank_tx.id:
                        related.matched_bank_transaction = bank_tx
                        updated = True

                    # Optionally propagate payment_type and payment_status from bank transaction
                    if bank_tx.payment_class and related.payment_type != bank_tx.payment_class.code:
                        related.payment_type = bank_tx.payment_class.code
                        updated = True
                    if bank_tx.payment_status and related.payment_status_id != bank_tx.payment_status_id:
                        related.payment_status = bank_tx.payment_status
                        updated = True

                    if updated:
                        related.save()
                        propagated_count += 1

            response_data = {
                "message": f"Successfully matched customer transaction {cust_tx_id} to bank transaction {bank_tx_id}",
                "reco_bank_transaction_id": bank_tx_id,
                "reco_customer_transaction_id": cust_tx_id,
                "propagated_to_same_document": propagated_count,
                "customer_transaction": {
                    "id": cust_tx.id,
                    "document_number": cust_tx.document_number,
                    "matched_bank_transaction_id": cust_tx.matched_bank_transaction_id,
                },
                "bank_transaction": {
                    "id": bank_tx.id,
                    "bank_id": bank_tx.bank_id,
                    "bank_ledger_entry_id": bank_tx.bank_ledger_entry_id,
                    "internal_number": bank_tx.internal_number,
                    "amount": float(bank_tx.amount),
                },
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response(
                {
                    "error": str(e),
                    "details": error_details,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MatchCustomerBankTransactionsView1(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Check if any PaymentIdentification is grouped
            grouped_mode = PaymentIdentification.objects.filter(grouped=True).exists()

            # === 1. Filter bank transactions with conditions ===
            # Debug: Check what's in the database
            all_bank_transactions = RecoBankTransaction.objects.all()
            all_payment_classes = PaymentClass.objects.all()
            
            # Get unique types and payment classes for debugging
            unique_types = RecoBankTransaction.objects.values_list('type', flat=True).distinct()
            unique_payment_classes = PaymentClass.objects.values_list('code', flat=True).distinct()
            
            # Get all payment_class codes from PaymentClass table
            payment_class_codes = list(PaymentClass.objects.values_list('code', flat=True))

            # Determine last imported ledger entries by last id
            last_ble = BankLedgerEntry.objects.order_by('-id').first()
            last_cle = CustomerLedgerEntry.objects.order_by('-id').first()

            if last_ble is None or last_cle is None:
                return Response({
                    "error": "No ledger entries found to scope matching",
                    "debug_info": {
                        "last_bank_ledger_entry": None if last_ble is None else last_ble.id,
                        "last_customer_ledger_entry": None if last_cle is None else last_cle.id,
                    }
                }, status=404)

            # Filter bank transactions: only those from last BLE, with existing criteria
            filtered_bank_transactions = RecoBankTransaction.objects.filter(
                bank_ledger_entry_id=last_ble.id,
                type='origine',
                payment_class__code__in=payment_class_codes
            )
            
            if not filtered_bank_transactions.exists():
                return Response({
                    "error": "No bank transactions found matching the criteria (type='origine')",
                    "debug_info": {
                        "total_bank_transactions": all_bank_transactions.count(),
                        "total_payment_classes": all_payment_classes.count(),
                        "available_types": list(unique_types),
                        "available_payment_classes": list(unique_payment_classes),
                        "transactions_with_type_origine": RecoBankTransaction.objects.filter(type='origine').count(),
                        "bank_transactions": RecoBankTransaction.objects.count()
                    }
                }, status=404)
            
            # Get customer transactions only from last CLE
            customer_transactions = RecoCustomerTransaction.objects.filter(customer_ledger_entry_id=last_cle.id)

            # If debug flag is set, return diagnostics and skip heavy matching
            if request.query_params.get('debug'):
                from django.db.models import Min, Max
                bank_total_in_ble = RecoBankTransaction.objects.filter(bank_ledger_entry_id=last_ble.id).count()
                bank_filtered_count = filtered_bank_transactions.count()
                customer_total_in_cle = customer_transactions.count()
                bank_codes = list(
                    RecoBankTransaction.objects.filter(bank_ledger_entry_id=last_ble.id)
                    .values_list('payment_class__code', flat=True).distinct()
                )
                customer_types = list(
                    RecoCustomerTransaction.objects.filter(customer_ledger_entry_id=last_cle.id)
                    .values_list('payment_type', flat=True).distinct()
                )
                intersection = sorted(set([c for c in bank_codes if c]) & set([t for t in customer_types if t]))
                bank_dates = RecoBankTransaction.objects.filter(bank_ledger_entry_id=last_ble.id).aggregate(
                    min=Min('operation_date'), max=Max('operation_date')
                )
                customer_dates = RecoCustomerTransaction.objects.filter(customer_ledger_entry_id=last_cle.id).aggregate(
                    min=Min('accounting_date'), max=Max('accounting_date')
                )
                return Response({
                    "debug": True,
                    "grouped_mode": grouped_mode,
                    "last_ble_id": last_ble.id,
                    "last_cle_id": last_cle.id,
                    "counts": {
                        "bank_total_in_ble": bank_total_in_ble,
                        "bank_filtered_count": bank_filtered_count,
                        "customer_total_in_cle": customer_total_in_cle,
                    },
                    "payment_categories": {
                        "bank_codes": bank_codes,
                        "customer_types": customer_types,
                        "intersection": intersection,
                    },
                    "date_ranges": {
                        "bank": bank_dates,
                        "customer": customer_dates,
                    },
                }, status=200)
            
            if not customer_transactions.exists():
                return Response({
                    "error": "No customer transactions found"
                }, status=404)
            
            # === 2. Convert to DataFrames for easier processing ===
            # Convert bank transactions to DataFrame
            bank_data = []
            for bt in filtered_bank_transactions:
                bank_data.append({
                    'id': bt.id,
                    'operation_date': bt.operation_date.strftime('%Y-%m-%d') if bt.operation_date else None,
                    'amount': safe_float(bt.amount, 0),
                    'ref': bt.ref if bt.ref else '',
                    'payment_class_id': bt.payment_class.code if bt.payment_class else '',
                    'label': bt.label if bt.label else ''
                })
            
            df_bank = pd.DataFrame(bank_data)
            
            # Convert customer transactions to DataFrame
            customer_data = []
            for ct in customer_transactions:
                customer_data.append({
                    'id': ct.id,
                    'accounting_date': ct.accounting_date.strftime('%Y-%m-%d') if ct.accounting_date else None,
                    'amount': safe_float(ct.amount, 0),
                    'total_amount': safe_float(ct.total_amount, 0),
                    'document_number': ct.document_number if ct.document_number else '',
                    'external_doc_number': ct.external_doc_number if ct.external_doc_number else '',
                    'payment_type': ct.payment_type if ct.payment_type else '',
                    'description': ct.description if ct.description else ''
                })
            
            df_customer = pd.DataFrame(customer_data)
            
            # === 3. Matching loop ===
            matches = []
            
            for i, bank_row in df_bank.iterrows():
                best_score = -1
                best_match_idx = None
                best_match_data = None
                
                for j, customer_row in df_customer.iterrows():
                    bank_payment_class = bank_row['payment_class_id']
                    customer_payment_type = customer_row['payment_type']
                    if bank_payment_class != customer_payment_type:
                        continue

                    score_date = score_date_tolerance(
                        customer_row['accounting_date'],
                        bank_row['operation_date']
                    )

                    # Use total_amount if grouped_mode, else use amount
                    if grouped_mode:
                        score_amount = score_amount_exact(
                            customer_row['total_amount'],
                            bank_row['amount']
                        )
                        score_reference = 0  # Remove reference score
                        total_score = 0.25 * score_date + 0.6 * score_amount
                    else:
                        score_amount = score_amount_exact(
                            customer_row['amount'],
                            bank_row['amount']
                        )
                        score_reference = score_reference_fuzzy(
                            customer_row['external_doc_number'],
                            bank_row['ref']
                        )
                        total_score = (
                            0.25 * score_date +
                            0.6 * score_amount +
                            0.15 * score_reference
                        )

                    if total_score > best_score:
                        best_score = total_score
                        best_match_idx = j
                        best_match_data = customer_row.to_dict()
                
                matches.append({
                    'bank_transaction_id': bank_row['id'],
                    'bank_operation_date': bank_row['operation_date'],
                    'bank_amount': bank_row['amount'],
                    'bank_ref': bank_row['ref'],
                    'bank_payment_class': bank_row['payment_class_id'],
                    'bank_label': bank_row['label'],
                    'customer_transaction_id': best_match_data['id'] if best_match_data else None,
                    'customer_accounting_date': best_match_data['accounting_date'] if best_match_data else None,
                    'customer_amount': best_match_data['amount'] if best_match_data else None,
                    'customer_total_amount': best_match_data['total_amount'] if best_match_data else None,
                    'customer_document_number': best_match_data['document_number'] if best_match_data else None,
                    'customer_external_doc_number': best_match_data['external_doc_number'] if best_match_data else None,
                    'customer_payment_type': best_match_data['payment_type'] if best_match_data else None,
                    'customer_description': best_match_data['description'] if best_match_data else None,
                    'score': best_score
                })
            
            # === 4. Create results DataFrame ===
            df_matches = pd.DataFrame(matches)
            
            # === 5. Separate high and low confidence matches ===
            df_matches_high = df_matches[df_matches['score'] >= 68].copy()
            df_matches_low = df_matches[df_matches['score'] < 68].copy()
            
            # === 5.5. Update matched_bank_transaction for high-confidence matches ===
            for _, match_row in df_matches_high.iterrows():
                if match_row['customer_transaction_id'] and match_row['bank_transaction_id']:
                    try:
                        customer_transaction = RecoCustomerTransaction.objects.get(id=match_row['customer_transaction_id'])
                        bank_transaction = RecoBankTransaction.objects.get(id=match_row['bank_transaction_id'])
                        customer_transaction.matched_bank_transaction = bank_transaction
                        customer_transaction.save()
                    except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                        continue
            
            # === 6. Save dataframes to files for later access ===
            import os
            from datetime import datetime
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(settings.BASE_DIR, 'matching_results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save high matches dataframe
            high_matches_file = os.path.join(output_dir, f'high_matches_{timestamp}.csv')
            if not df_matches_high.empty:
                df_matches_high.to_csv(high_matches_file, index=False)
                high_matches_saved = True
                high_matches_path = high_matches_file
            else:
                high_matches_saved = False
                high_matches_path = None
            
            # Save low matches dataframe
            low_matches_file = os.path.join(output_dir, f'low_matches_{timestamp}.csv')
            if not df_matches_low.empty:
                df_matches_low.to_csv(low_matches_file, index=False)
                low_matches_saved = True
                low_matches_path = low_matches_file
            else:
                low_matches_saved = False
                low_matches_path = None
            
            # Save all matches dataframe
            all_matches_file = os.path.join(output_dir, f'all_matches_{timestamp}.csv')
            df_matches.to_csv(all_matches_file, index=False)

            # === 6.5. Promote low matches to high if grouped_mode is True ===
            if grouped_mode:
                rows_to_drop = []
                for idx_low, row_low in df_matches_low.iterrows():
                    # Find matching high match by document number
                    match = df_matches_high[
                        (df_matches_high['customer_document_number'] == row_low['customer_document_number'])
                    ]
                    if not match.empty:
                        matched_bank_id = match.iloc[0]['bank_transaction_id']
                        customer_transaction_id = row_low['customer_transaction_id']
                        
                        # Update ALL CustomerTransactions with the same document_number, description, and accounting_date
                        try:
                            ct = RecoCustomerTransaction.objects.get(id=customer_transaction_id)
                            bt = RecoBankTransaction.objects.get(id=matched_bank_id)
                            matching_customer_transactions = RecoCustomerTransaction.objects.filter(
                                document_number=ct.document_number,
                                description=ct.description,
                                accounting_date=ct.accounting_date
                            )
                            for matching_ct in matching_customer_transactions:
                                matching_ct.matched_bank_transaction = bt
                                matching_ct.save()
                        except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                            continue
                        rows_to_drop.append(idx_low)
                # Remove matched rows from df_matches_low
                df_matches_low_updated = df_matches_low.drop(rows_to_drop)
                # Save the updated low matches file
                low_matches_file_updated = os.path.join(output_dir, f'low_matches_{timestamp}_updated.csv')
                df_matches_low_updated.to_csv(low_matches_file_updated, index=False)
            
            # === 7. Calculate statistics ===
            total_rows = len(df_bank)
            high_match_count = len(df_matches_high)
            low_match_count = len(df_matches_low)
            
            high_match_percentage = (100 * high_match_count / total_rows) if total_rows > 0 else 0
            low_match_percentage = (100 * low_match_count / total_rows) if total_rows > 0 else 0
            
            # === 8. Prepare response data ===
            # Sanitize DataFrames to replace inf/nan values before JSON serialization
            df_matches_high_clean = sanitize_dataframe_for_json(df_matches_high) if not df_matches_high.empty else df_matches_high
            df_matches_low_clean = sanitize_dataframe_for_json(df_matches_low) if not df_matches_low.empty else df_matches_low
            
            response_data = {
                "summary": {
                    "total_bank_transactions": total_rows,
                    "high_matches_count": high_match_count,
                    "high_matches_percentage": round(high_match_percentage, 2),
                    "low_matches_count": low_match_count,
                    "low_matches_percentage": round(low_match_percentage, 2)
                },
                "saved_files": {
                    "all_matches_csv": all_matches_file,
                    "high_matches_csv": high_matches_path if high_matches_saved else None,
                    "low_matches_csv": low_matches_path if low_matches_saved else None,
                    "output_directory": output_dir
                },
                "dataframe_info": {
                    "high_matches_shape": df_matches_high.shape if not df_matches_high.empty else (0, 0),
                    "low_matches_shape": df_matches_low.shape if not df_matches_low.empty else (0, 0),
                    "all_matches_shape": df_matches.shape
                },
                "high_matches": df_matches_high_clean.to_dict('records') if not df_matches_high.empty else [],
                "low_matches": df_matches_low_clean.to_dict('records') if not df_matches_low.empty else []
            }
            
            return Response(response_data, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)


class MatchTransactionView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Fetch Data: Load all records from PaymentIdentification table
            payment_identifications = PaymentIdentification.objects.all()
            
            # Load all unmatched RecoBankTransaction entries where payment_class and payment_status are NULL
            bank_transactions = RecoBankTransaction.objects.filter(
                payment_class__isnull=True,
                payment_status__isnull=True
            )
            
            if not bank_transactions.exists():
                return Response({"message": "No unmatched bank transactions found."}, status=200)
            
            updated_count = 0
            
            # Process each transaction and check ALL PaymentIdentification conditions at once
            for transaction in bank_transactions:
                amount = float(transaction.amount)
                label = transaction.label
                
                # Find the best matching PaymentIdentification for this transaction
                best_match = None
                
                for payment_id in payment_identifications:
                    description = payment_id.description
                    debit = payment_id.debit
                    credit = payment_id.credit
                    
                    # Check if description matches
                    if description.lower() not in label.lower():
                        continue
                    
                    # Check amount/debit/credit logic
                    amount_matches = False
                    if amount < 0 and debit:
                        amount_matches = True
                    elif amount > 0 and credit:
                        amount_matches = True
                    
                    if amount_matches:
                        # This is a valid match
                        best_match = payment_id
                        break  # Use the first valid match found
                
                # Update the transaction if a match was found
                if best_match:
                    transaction.payment_class = best_match.payment_status.payment_class
                    transaction.payment_status = best_match.payment_status
                    transaction.accounting_account = best_match.payment_status.accounting_account
                    transaction.type = "origine"  # Set type to "origine" for matched transactions
                    transaction.save()
                    updated_count += 1
            
            return Response({
                "message": f"Updated {updated_count} transactions.",
                "total_transactions_processed": bank_transactions.count(),
                "payment_identifications_checked": len(payment_identifications),
                "matching_rules_applied": {
                    "description_match": "description_in_label",
                    "amount_logic": "negative_amount_with_debit_true_or_positive_amount_with_credit_true"
                }
            }, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)


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
                # Backward compatibility
            }
        
        return Response(user_data)


class TaxManagementView(APIView):
    """
    View to manage Tax entries in the database.
    POST: Add or update tax entries with name and description list
    GET: Get all tax entries
    """
    
    def post(self, request):
        try:
            name = request.data.get('name')
            description = request.data.get('description')
            
            if not name:
                return Response(
                    {"error": "Name is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not description:
                return Response(
                    {"error": "Description is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Ensure description is a list
            if not isinstance(description, list):
                description = [description]
            
            # Try to get existing tax with this name
            try:
                tax = Tax.objects.get(name__iexact=name)
                
                # Add new description if not already present
                if description not in tax.description:
                    tax.description.append(description)
                
                tax.save()
                
                return Response({
                    "message": f"Updated existing tax '{name}' with new description",
                    "tax_id": tax.id,
                    "name": tax.name,
                    "descriptions": tax.description
                }, status=status.HTTP_200_OK)
                
            except Tax.DoesNotExist:
                # Create new tax entry
                tax = Tax.objects.create(
                    name=name,
                    description=description
                )
                
                return Response({
                    "message": f"Created new tax '{name}'",
                    "tax_id": tax.id,
                    "name": tax.name,
                    "descriptions": tax.description
                }, status=status.HTTP_201_CREATED)
                
        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """Get all tax entries"""
        try:
            taxes = Tax.objects.all()
            data = []
            for tax in taxes:
                data.append({
                    "id": tax.id,
                    "name": tax.name,
                    "descriptions": tax.description
                })
            
            return Response(data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def put(self, request):
        """Update tax entry with complete replacement of descriptions"""
        try:
            name = request.data.get('name')
            description = request.data.get('description')
            
            if not name:
                return Response(
                    {"error": "Name is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not description:
                return Response(
                    {"error": "Description is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Ensure description is a list
            if not isinstance(description, list):
                description = [description]
            
            # Try to get existing tax with this name
            try:
                tax = Tax.objects.get(name__iexact=name)
                
                # Replace the entire description list
                tax.description = description
                tax.save()
                
                return Response({
                    "message": f"Updated tax '{name}' with new descriptions",
                    "tax_id": tax.id,
                    "name": tax.name,
                    "descriptions": tax.description
                }, status=status.HTTP_200_OK)
                
            except Tax.DoesNotExist:
                return Response(
                    {"error": f"Tax '{name}' not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
                
        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def patch(self, request):
        """Update specific values in the description list"""
        try:
            name = request.data.get('name')
            old_value = request.data.get('old_value')
            new_value = request.data.get('new_value')
            if not name:
                return Response({"error": "Name is required"}, status=status.HTTP_400_BAD_REQUEST)
            if not old_value:
                return Response({"error": "old_value is required"}, status=status.HTTP_400_BAD_REQUEST)
            if not new_value:
                return Response({"error": "new_value is required"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                tax = Tax.objects.get(name__iexact=name)
                if old_value in tax.description:
                    updated_descriptions = [new_value if desc == old_value else desc for desc in tax.description]
                    tax.description = updated_descriptions
                    tax.save()
                    return Response({
                        "message": f"Updated '{old_value}' to '{new_value}' in tax '{name}'",
                        "tax_id": tax.id,
                        "name": tax.name,
                        "descriptions": tax.description,
                        "updated": {"old_value": old_value, "new_value": new_value}
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({"error": f"Value '{old_value}' not found in tax '{name}'"}, status=status.HTTP_404_NOT_FOUND)
            except Tax.DoesNotExist:
                return Response({"error": f"Tax '{name}' not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class MatchTaxView(APIView):
    """
    View to match bank transactions with tax descriptions and update them with dynamic tax amounts.
    """


    
    def post(self, request):
        try:
            # Get all taxes with their descriptions
            taxes = Tax.objects.all()
            
            # Get all bank transactions from reco table
            bank_transactions = RecoBankTransaction.objects.all()
            
            if not bank_transactions.exists():
                return Response({
                    "message": "No bank transactions found",
                    "matched_count": 0
                }, status=status.HTTP_200_OK)
            
            matched_count = 0
            
            for transaction in bank_transactions:
                label = transaction.label.lower() if transaction.label else ""
                
                matched_tax = None
                for tax in taxes:
                    if tax.description:
                        for desc in tax.description:
                            # Match even if tax keyword is embedded in other text or attached to digits/symbols
                            # Allow matching like: "311224TVA", "TVA LCN", "TVA123", "TVA-456"
                            pattern = r'\b' + re.escape(desc.lower()) + r'\b|(?<=\d)' + re.escape(desc.lower()) + r'(?=\d)|(?<=\d)' + re.escape(desc.lower()) + r'\b|\b' + re.escape(desc.lower()) + r'(?=\d)'
                            if re.search(pattern, label):
                                matched_tax = tax
                                break
                        if matched_tax:
                            break
                
                if matched_tax:
                    transaction.type = matched_tax.name.lower()  # Set type to the actual tax name
                    
                    transaction.save()
                    matched_count += 1
            
            return Response({
                "message": f"Successfully matched {matched_count} transactions with taxes",
                "matched_count": matched_count,
                "total_transactions": bank_transactions.count()
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """Get statistics about tax matching"""
        try:
            total_transactions = RecoBankTransaction.objects.count()
            tax_transactions = RecoBankTransaction.objects.filter(type="tax").count()
            
            tax_keywords = []
            for tax in Tax.objects.all():
                if tax.description and isinstance(tax.description, list):
                    tax_keywords.extend(tax.description)
            
            return Response({
                "total_transactions": total_transactions,
                "tax_transactions": tax_transactions,
                "tax_keywords": tax_keywords
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




    def get(self, request):
        """Get matching statistics"""
        try:
            # Count transactions by type
            all_transactions = BankTransaction.objects.all()
            
            tva_count = 0
            commission_count = 0
            plo_count = 0
            agios_count = 0
            main_count = 0
            
            import re
            commission_regex = r'(?:^|[^a-zA-Z])com(miss(ion)?|m(iss(ion)?)?)?\b'
            
            for trans in all_transactions:
                label = trans.label.lower() if trans.label else ''
                
                if 'tva' in label:
                    tva_count += 1
                elif re.search(commission_regex, label, re.IGNORECASE):
                    commission_count += 1
                elif 'plo' in label:
                    plo_count += 1
                elif 'agios' in label:
                    agios_count += 1
                else:
                    main_count += 1
            
            return Response({
                'success': True,
                'statistics': {
                    'total_transactions': all_transactions.count(),
                    'main_transactions': main_count,
                    'tva_transactions': tva_count,
                    'commission_transactions': commission_count,
                    'plo_transactions': plo_count,
                    'agios_transactions': agios_count
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class MatchBankTransactionTaxesView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            from django.db.models import Q
            import uuid
            
            # Get all bank transactions (RECO)
            all_transactions = RecoBankTransaction.objects.all()
            
            # Separate origine and non-origine transactions
            origine_transactions = RecoBankTransaction.objects.filter(type='origine')
            non_origine_transactions = RecoBankTransaction.objects.filter(~Q(type='origine') & ~Q(type='origine'))
            
            matched_count = 0
            unmatched_count = 0
            
            # Process each non-origine transaction
            for non_origine in non_origine_transactions:
                matched = False
                
                # First, try to match by ref
                if non_origine.ref:
                    matching_origine = origine_transactions.filter(ref=non_origine.ref).first()
                    if matching_origine:
                        # If origine has internal_number, use it; otherwise generate new one
                        if matching_origine.internal_number:
                            non_origine.internal_number = matching_origine.internal_number
                        else:
                            new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                            matching_origine.internal_number = new_internal_number
                            matching_origine.save()
                            non_origine.internal_number = new_internal_number
                        
                        non_origine.save()
                        matched = True
                        matched_count += 1
                
                # If not matched by ref, try to match by date_ref
                if not matched and non_origine.date_ref:
                    matching_origine = origine_transactions.filter(date_ref=non_origine.date_ref).first()
                    if matching_origine:
                        # If origine has internal_number, use it; otherwise generate new one
                        if matching_origine.internal_number:
                            non_origine.internal_number = matching_origine.internal_number
                        else:
                            new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                            matching_origine.internal_number = new_internal_number
                            matching_origine.save()
                            non_origine.internal_number = new_internal_number
                        
                        non_origine.save()
                        matched = True
                        matched_count += 1
                
                # If still not matched, generate a new internal_number
                if not matched:
                    non_origine.internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                    non_origine.save()
                    unmatched_count += 1
            
            # Also ensure all origine transactions have internal_number
            for origine in origine_transactions:
                if not origine.internal_number:
                    origine.internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                    origine.save()

            # === Custom logic for agios transactions (Probabilistic Group-Based Allocation) ===
            import logging
            from .models import RecoCustomerTransaction, CustomerTaxRow, PaymentClass
            from django.db.models import Sum, Q
            from decimal import Decimal
            from collections import defaultdict
            
            logger = logging.getLogger(__name__)
            
            # STEP 1: Group agios by date_ref (bank side)
            agios_transactions = RecoBankTransaction.objects.filter(type='agios')
            agios_by_date_ref = defaultdict(list)
            
            for agios in agios_transactions:
                if agios.date_ref:
                    agios_by_date_ref[agios.date_ref].append(agios)
                else:
                    logger.warning(f"Agios transaction {agios.id} skipped: no date_ref (amount: {agios.amount})")
            
            logger.info(f"Starting agios matching process. Found {agios_transactions.count()} agios transactions grouped into {len(agios_by_date_ref)} date_ref groups.")
            
            agios_matched_count = 0
            agios_unmatched_count = 0
            
            # Process each date_ref group
            for date_ref, agios_group in agios_by_date_ref.items():
                logger.info(f"Processing date_ref={date_ref} with {len(agios_group)} agios transactions")
                
                # STEP 2: Identify candidate customer transactions
                # Filter by accounting_date = date_ref
                customer_candidates = RecoCustomerTransaction.objects.filter(
                    accounting_date=date_ref
                )
                
                if customer_candidates.count() == 0:
                    logger.warning(f"Date_ref {date_ref}: No customer transactions found")
                    agios_unmatched_count += len(agios_group)
                    continue
                
                logger.info(f"Date_ref {date_ref}: Found {customer_candidates.count()} candidate customer transactions")
                
                # Group customer transactions by document_number (document group)
                customer_by_doc = defaultdict(list)
                for cust_tx in customer_candidates:
                    if cust_tx.document_number:
                        customer_by_doc[cust_tx.document_number].append(cust_tx)
                
                logger.info(f"Date_ref {date_ref}: Grouped into {len(customer_by_doc)} document_number groups")
                
                # Process each agios in this date_ref group
                for agios in agios_group:
                    matched = False
                    agios_amount = abs(float(agios.amount))
                    
                    logger.info(f"Processing agios {agios.id}: amount={agios.amount}, date_ref={date_ref}")
                    
                    # STEP 3 & 4: Compute simulated agios and match using tolerance
                    best_match = None
                    best_difference = float('inf')
                    checked_doc_numbers = []
                    no_tax_rows_docs = []
                    mismatch_docs = []
                    
                    for doc_number, cust_tx_group in customer_by_doc.items():
                        checked_doc_numbers.append(doc_number)
                        # Note: Agios transactions don't have payment_class, so we don't filter by it
                        # Compute simulated agios from customer side
                        agios_tax_rows = CustomerTaxRow.objects.filter(
                            transaction__in=cust_tx_group,
                            tax_type__iexact='agios'
                        )
                        
                        logger.debug(f"Agios {agios.id}, doc_number={doc_number}: Found {agios_tax_rows.count()} agios tax rows for {len(cust_tx_group)} customer transactions")
                        
                        # Sum tax_amount for this document group
                        # Note: We use tax_amount (individual) not total_tax_amount (which is the same for all rows with same document_number)
                        simulated_agios = agios_tax_rows.aggregate(
                            total=Sum('tax_amount')
                        )['total'] or 0
                        
                        if simulated_agios == 0:
                            no_tax_rows_docs.append(doc_number)
                            logger.warning(f"Agios {agios.id}, doc_number={doc_number}: No agios tax rows found or total_tax_amount is 0 (checked {agios_tax_rows.count()} rows, {len(cust_tx_group)} customer transactions)")
                            continue
                        
                        simulated_agios_abs = abs(float(simulated_agios))
                        
                        # STEP 4: Match using tolerance (5%)
                        # ABS(simulated_agios - bank_agios) / bank_agios <= 5%
                        difference = abs(agios_amount - simulated_agios_abs)
                        tolerance_ratio = difference / agios_amount if agios_amount > 0 else float('inf')
                        
                        logger.info(f"Agios {agios.id}, doc_number={doc_number}: simulated_agios={simulated_agios_abs}, bank_agios={agios_amount}, difference={difference:.2f}, tolerance_ratio={tolerance_ratio:.4f} (5% threshold: 0.05)")
                        
                        if tolerance_ratio <= 0.05:  # 5% tolerance
                            logger.info(f"Agios {agios.id}, doc_number={doc_number}: ✅ Match found! tolerance_ratio={tolerance_ratio:.4f} <= 0.05")
                            if difference < best_difference:
                                best_difference = difference
                                best_match = {
                                    'doc_number': doc_number,
                                    'customer_transactions': cust_tx_group,
                                    'simulated_agios': simulated_agios_abs,
                                    'difference': difference
                                }
                        else:
                            mismatch_docs.append({
                                'doc_number': doc_number,
                                'simulated': simulated_agios_abs,
                                'bank': agios_amount,
                                'difference': difference,
                                'tolerance_ratio': tolerance_ratio
                            })
                            logger.warning(f"Agios {agios.id}, doc_number={doc_number}: ❌ No match - tolerance_ratio={tolerance_ratio:.4f} > 0.05 (simulated={simulated_agios_abs}, bank={agios_amount}, diff={difference:.2f})")
                    
                    # Log summary for this agios
                    if not best_match:
                        logger.warning(f"Agios {agios.id} summary: Checked {len(checked_doc_numbers)} document_numbers")
                        if no_tax_rows_docs:
                            logger.warning(f"  - {len(no_tax_rows_docs)} docs with no agios tax rows: {no_tax_rows_docs[:5]}")
                        if mismatch_docs:
                            logger.warning(f"  - {len(mismatch_docs)} docs with amount mismatches:")
                            for m in mismatch_docs[:3]:  # Show first 3
                                logger.warning(f"    * {m['doc_number']}: simulated={m['simulated']:.2f}, bank={m['bank']:.2f}, ratio={m['tolerance_ratio']:.4f}")
                    
                    # If no individual match found, try matching sum of all agios in this date_ref group
                    if not best_match and len(agios_group) > 1:
                        logger.info(f"Agios {agios.id}: No individual match found, trying to match sum of all {len(agios_group)} agios for date_ref={date_ref}")
                        total_agios_sum = sum(abs(float(a.amount)) for a in agios_group)
                        logger.info(f"Total agios sum for date_ref {date_ref}: {total_agios_sum}")
                        
                        # Try matching total against each document_number group
                        for doc_number, cust_tx_group in customer_by_doc.items():
                            agios_tax_rows = CustomerTaxRow.objects.filter(
                                transaction__in=cust_tx_group,
                                tax_type__iexact='agios'
                            )
                            simulated_agios = agios_tax_rows.aggregate(total=Sum('tax_amount'))['total'] or 0
                            
                            if simulated_agios == 0:
                                continue
                            
                            simulated_agios_abs = abs(float(simulated_agios))
                            difference = abs(total_agios_sum - simulated_agios_abs)
                            tolerance_ratio = difference / total_agios_sum if total_agios_sum > 0 else float('inf')
                            
                            logger.info(f"Agios {agios.id}, doc_number={doc_number}: Trying total match - total_agios={total_agios_sum}, simulated={simulated_agios_abs}, ratio={tolerance_ratio:.4f}")
                            
                            if tolerance_ratio <= 0.05:
                                logger.info(f"Agios {agios.id}: ✅ Total match found! All {len(agios_group)} agios match doc_number={doc_number}")
                                best_match = {
                                    'doc_number': doc_number,
                                    'customer_transactions': cust_tx_group,
                                    'simulated_agios': simulated_agios_abs,
                                    'difference': difference,
                                    'is_total_match': True
                                }
                                break
                    
                    # STEP 5: Affect agios to GROUP and allocate proportionally
                    if best_match:
                        logger.info(f"Agios {agios.id}: Best match found with doc_number={best_match['doc_number']} (difference={best_match['difference']:.2f})")
                        
                        cust_tx_group = best_match['customer_transactions']
                        total_customer_amount = sum(abs(float(tx.total_amount or tx.amount)) for tx in cust_tx_group)
                        
                        if total_customer_amount == 0:
                            logger.warning(f"Agios {agios.id}: Total customer amount is 0, cannot allocate")
                            agios_unmatched_count += 1
                            continue
                        
                        # Find origine transactions for this customer group and allocate
                        origine_found = False
                        for cust_tx in cust_tx_group:
                            # Find matching origine transaction
                            matching_origine = RecoBankTransaction.objects.filter(
                                type='origine',
                                operation_date=cust_tx.accounting_date,
                                amount=cust_tx.total_amount
                            ).first()
                            
                            if matching_origine:
                                # Calculate allocation ratio
                                customer_amount = abs(float(cust_tx.total_amount or cust_tx.amount))
                                allocation_ratio = customer_amount / total_customer_amount
                                
                                logger.info(f"Agios {agios.id}: Allocating to origine {matching_origine.id} via customer transaction {cust_tx.id} (ratio={allocation_ratio:.4f}, customer_amount={customer_amount}, total={total_customer_amount})")
                                
                                # Link agios to this origine (proportional allocation)
                                agios.internal_number = matching_origine.internal_number
                                agios.payment_class = matching_origine.payment_class
                                agios.payment_status = matching_origine.payment_status
                                agios.accounting_account = matching_origine.accounting_account
                                agios.save()
                                
                                matched = True
                                origine_found = True
                                break  # Use first matching origine found
                        
                        if matched:
                            agios_matched_count += 1
                        elif not origine_found:
                            logger.warning(f"Agios {agios.id}: Amount matched but no origine transaction found for customer group (doc_number={best_match['doc_number']})")
                            agios_unmatched_count += 1
                    else:
                        logger.warning(f"Agios {agios.id}: Could not find matching customer group within 5% tolerance (amount={agios.amount}, date_ref={date_ref})")
                        agios_unmatched_count += 1
            
            logger.info(f"Agios matching completed: {agios_matched_count} matched, {agios_unmatched_count} unmatched out of {agios_transactions.count()} total")

            return Response({
                "message": f"Successfully processed reco bank transaction taxes matching",
                "total_transactions": all_transactions.count(),
                "origine_transactions": origine_transactions.count(),
                "non_origine_transactions": non_origine_transactions.count(),
                "matched_count": matched_count,
                "unmatched_count": unmatched_count,
                "agios_matched_count": agios_matched_count,
                "agios_unmatched_count": agios_unmatched_count,
                "total_with_internal_number": RecoBankTransaction.objects.filter(internal_number__isnull=False).count()
            }, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)


import ast

class FormulaParser(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()
        self.constants = set()
        self.operators = set()
    
    def visit_Name(self, node):
        self.variables.add(node.id)
    
    def visit_Constant(self, node):  # Python 3.8+
        self.constants.add(node.value)
    
    def visit_Num(self, node):  # for older versions
        self.constants.add(node.n)
    
    def visit_BinOp(self, node):
        op_type = type(node.op)
        symbol = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
            ast.Pow: '**',
        }.get(op_type)
        if symbol:
            self.operators.add(symbol)
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.operand)

    def visit_Expr(self, node):
        self.visit(node.value)


class ExtractCustomerTaxesView(APIView):
    """
    View to extract tax rows for customer transactions based on conventions and smart formula engine.
    """
    def post(self, request, *args, **kwargs):
        import ast
        transactions = request.data.get("transactions", None)
        extracted_taxes = []
        if transactions is None:
            from .models import RecoCustomerTransaction, CustomerTaxRow
            transactions = RecoCustomerTransaction.objects.all()

        from datetime import date, datetime
        def resolve_variable(var, tax_rule_lookup, tax_results, params, tx_vars, resolving):
            # Prevent infinite recursion
            if var in resolving:
                return None
            # Check if already calculated (case-insensitive)
            var_lower = var.lower().strip()
            for tax_type, value in tax_results.items():
                if tax_type.lower().strip() == var_lower:
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return value
            # a. taxrule table (by name) - use case-insensitive lookup
            rule = tax_rule_lookup.get(var_lower)
            if rule is not None:
                if rule.calculation_type == "flat":
                    # Use 'rate' or 'value' (support both field names)
                    return getattr(rule, 'rate', getattr(rule, 'value', None))
                elif rule.calculation_type == "formula" and rule.formula:
                    # Recursively resolve variables in the formula
                    class VarFinder(ast.NodeVisitor):
                        def __init__(self):
                            self.vars = set()
                        def visit_Name(self, node):
                            self.vars.add(node.id)
                    try:
                        tree = ast.parse(rule.formula, mode="eval")
                        finder = VarFinder()
                        finder.visit(tree)
                        local_vars = {}
                        for sub_var in finder.vars:
                            local_vars[sub_var] = resolve_variable(sub_var, tax_rule_lookup, tax_results, params, tx_vars, resolving | {var})
                        if all(v is not None for v in local_vars.values()):
                            return eval(compile(tree, '<string>', 'eval'), {}, local_vars)
                        else:
                            return None
                    except Exception:
                        return None
            # b. conventionparameter table
            if var in params:
                return params[var]
            # c. customertransaction table
            if var in tx_vars:
                value = tx_vars[var]
                # Try to parse date strings to date objects
                if isinstance(value, str):
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):  # common formats
                        try:
                            return datetime.strptime(value, fmt).date()
                        except Exception:
                            continue
                return value
            return None

        from .models import CustomerTaxRow
        for tx in transactions:
            account_number = tx.account_number
            tx_vars = {k: v for k, v in tx.__dict__.items() if not k.startswith('_')}
            reference = tx.document_number
            payment_class = tx.payment_type
            payment_status = tx.payment_status
            try:
                agency = Agency.objects.get(code=account_number)
                bank = agency.bank
            except Agency.DoesNotExist:
                continue
            conventions = Convention.objects.filter(bank=bank, is_active=True).order_by("-id")
            for conv in conventions:
                tax_rules_queryset = TaxRule.objects.filter(
                    convention=conv,
                    payment_class__code=payment_class,
                    payment_status=payment_status
                ).order_by('-id')  # Order by id descending to get most recent first
                
                # Deduplicate by tax_type (case-insensitive), keeping the most recent one
                seen_tax_types = {}
                tax_rules = []
                for tr in tax_rules_queryset:
                    if hasattr(tr, 'tax_type') and tr.tax_type:
                        key = tr.tax_type.lower().strip()
                        if key not in seen_tax_types:
                            seen_tax_types[key] = tr
                            tax_rules.append(tr)
                
                params = {p.name: p.value for p in ConventionParameter.objects.all()}
                # Build lookup with case-insensitive keys from the deduplicated list
                tax_rule_lookup = {}
                for tr in tax_rules:
                    if hasattr(tr, 'tax_type') and tr.tax_type:
                        key = tr.tax_type.lower().strip()
                        tax_rule_lookup[key] = tr
                tax_results = {}
                for rule in tax_rules:
                    if rule.calculation_type == "flat":
                        val = getattr(rule, 'rate', getattr(rule, 'value', None))
                        # Round/quantize to 3 decimals and convert to string
                        from decimal import Decimal, ROUND_HALF_UP
                        try:
                            val_decimal = Decimal(str(val)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                            val_str = format(val_decimal, '.3f')
                        except Exception:
                            val_str = str(val)
                        tax_results[rule.tax_type.lower().strip()] = val_str
                        extracted_taxes.append({
                            "tax_name": rule.tax_type,
                            "value": val_str,
                            "type": "flat",
                            "bank": bank.name,
                            "transaction_reference": reference,
                            "convention": conv.name,
                            "tax_rule": rule.id,
                        })
                        # Store in CustomerTaxRow (flat)
                        try:
                            # Create the tax row first
                            tax_row = CustomerTaxRow.objects.create(
                                transaction=tx,
                                tax_type=rule.tax_type,
                                tax_amount=val,
                                applied_formula=None,
                                rate_used=getattr(rule, 'rate', None)
                            )
                            
                            # Calculate total_tax_amount for all rows with same document_number and tax_type
                            from django.db.models import Sum
                            from decimal import Decimal
                            total_tax_amount = CustomerTaxRow.objects.filter(
                                transaction__document_number=tx.document_number,
                                tax_type=rule.tax_type
                            ).aggregate(total=Sum('tax_amount'))['total'] or 0
                            
                            # Update all rows with the same document_number and tax_type with the correct total
                            CustomerTaxRow.objects.filter(
                                transaction__document_number=tx.document_number,
                                tax_type=rule.tax_type
                            ).update(total_tax_amount=total_tax_amount)
                        except Exception as e:
                            print(f"CustomerTaxRow flat error: {e}")
                    elif rule.calculation_type == "formula" and rule.formula:
                        # Parse variables in formula
                        class VarFinder(ast.NodeVisitor):
                            def __init__(self):
                                self.vars = set()
                            def visit_Name(self, node):
                                self.vars.add(node.id)
                        try:
                            tree = ast.parse(rule.formula, mode="eval")
                            finder = VarFinder()
                            finder.visit(tree)
                            formula_vars = finder.vars
                        except Exception:
                            extracted_taxes.append({
                                "error": f"Invalid formula for {rule.tax_type}",
                                "bank": bank.name,
                                "transaction_reference": reference,
                                "convention": conv.name,
                                "tax_rule": rule.id,
                            })
                            continue
                        missing = []
                        local_vars = {}
                        for var in formula_vars:
                            value = resolve_variable(var, tax_rule_lookup, tax_results, params, tx_vars, set())
                            if value is None:
                                missing.append(var)
                            else:
                                # If value is a date, keep as date for now
                                # If value is Decimal, convert to float
                                import decimal
                                if isinstance(value, decimal.Decimal):
                                    value = float(value)
                                local_vars[var] = value
                        # After collecting all variables, convert date operations to day differences
                        # Replace date objects with ints if both operands are dates
                        # We'll use a custom eval context to handle date subtraction
                        def custom_eval(expr, local_vars):
                            import operator
                            from datetime import timedelta
                            def date_sub(a, b):
                                if isinstance(a, (date, datetime)) and isinstance(b, (date, datetime)):
                                    # Always return the absolute number of days between two dates
                                    return abs((a - b).days)
                                raise TypeError("date_sub only supports date - date")
                            # Add more custom ops if needed
                            safe_locals = dict(local_vars)
                            safe_locals['date_sub'] = date_sub
                            # Patch AST to replace date - date with date_sub(a, b)
                            class DateOpTransformer(ast.NodeTransformer):
                                def visit_BinOp(self, node):
                                    self.generic_visit(node)
                                    if isinstance(node.op, ast.Sub):
                                        if (isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)):
                                            return ast.copy_location(
                                                ast.Call(
                                                    func=ast.Name(id='date_sub', ctx=ast.Load()),
                                                    args=[node.left, node.right],
                                                    keywords=[]
                                                ),
                                                node
                                            )
                                    return node
                            try:
                                tree = ast.parse(expr, mode="eval")
                                tree = DateOpTransformer().visit(tree)
                                ast.fix_missing_locations(tree)
                                result = eval(compile(tree, '<string>', 'eval'), {}, safe_locals)
                                # If result is timedelta, convert to days
                                if isinstance(result, timedelta):
                                    return result.days
                                return result
                            except Exception as e:
                                raise e
                        if missing:
                            extracted_taxes.append({
                                "error": f"Missing variables: {', '.join(missing)}",
                                "bank": bank.name,
                                "transaction_reference": reference,
                                "convention": conv.name,
                                "tax_rule": rule.id,
                            })
                            continue
                        try:
                            # Use custom_eval to handle date and timedelta
                            expr = rule.formula
                            result = custom_eval(expr, local_vars)
                            # Round/quantize to 3 decimals and convert to string
                            from decimal import Decimal, ROUND_HALF_UP
                            try:
                                result_decimal = Decimal(str(result)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                                result_str = format(result_decimal, '.3f')
                            except Exception:
                                result_str = str(result)
                            tax_results[rule.tax_type.lower().strip()] = result_str
                            extracted_taxes.append({
                                "tax_name": rule.tax_type,
                                "value": result_str,
                                "type": "formula",
                                "bank": bank.name,
                                "transaction_reference": reference,
                                "convention": conv.name,
                                "tax_rule": rule.id,
                            })
                            # Store in CustomerTaxRow (formula)
                            try:
                                # Create the tax row first
                                tax_row = CustomerTaxRow.objects.create(
                                    transaction=tx,
                                    tax_type=rule.tax_type,
                                    tax_amount=result,
                                    applied_formula=rule.formula,
                                    rate_used=getattr(rule, 'rate', None)
                                )
                                
                                # Calculate total_tax_amount for all rows with same document_number and tax_type
                                from django.db.models import Sum
                                from decimal import Decimal
                                total_tax_amount = CustomerTaxRow.objects.filter(
                                    transaction__document_number=tx.document_number,
                                    tax_type=rule.tax_type
                                ).aggregate(total=Sum('tax_amount'))['total'] or 0
                                
                                # Update all rows with the same document_number and tax_type with the correct total
                                CustomerTaxRow.objects.filter(
                                    transaction__document_number=tx.document_number,
                                    tax_type=rule.tax_type
                                ).update(total_tax_amount=total_tax_amount)
                            except Exception as e:
                                print(f"CustomerTaxRow formula error: {e}")
                        except Exception as e:
                            extracted_taxes.append({
                                "error": f"Evaluation error for {rule.tax_type}: {str(e)}",
                                "bank": bank.name,
                                "transaction_reference": reference,
                                "convention": conv.name,
                                "tax_rule": rule.id,
                            })
        return Response({"extracted_taxes": extracted_taxes}, status=status.HTTP_200_OK)


class UnmatchedTransactionsView(APIView):
    """
    View to generate two lists:
    1. Unmatched bank transactions (all bank transactions minus those in high matches)
    2. Unmatched customer transactions (customer transactions without matched_bank_transaction)
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            from .models import RecoBankTransaction, RecoCustomerTransaction
            
            # === 1. Find the latest high matches CSV file ===
            matching_results_dir = 'matching_results'
            if not os.path.exists(matching_results_dir):
                return Response({
                    "error": "No matching results directory found"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Get all high matches files and find the latest one
            high_matches_files = glob.glob(os.path.join(matching_results_dir, 'high_matches_*.csv'))
            if not high_matches_files:
                return Response({
                    "error": "No high matches files found"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Sort by modification time to get the latest
            latest_high_matches_file = max(high_matches_files, key=os.path.getmtime)
            
            # === 2. Extract bank transaction IDs from high matches ===
            try:
                df_high_matches = pd.read_csv(latest_high_matches_file)
                high_match_bank_ids = set(df_high_matches['bank_transaction_id'].tolist())
            except Exception as e:
                return Response({
                    "error": f"Error reading high matches file: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # === 3. Get unmatched bank transactions ===
            # Get all bank transactions with type='origine' and exclude those in high matches
            all_bank_transactions = RecoBankTransaction.objects.filter(type='origine')
            unmatched_bank_transactions = all_bank_transactions.exclude(id__in=high_match_bank_ids)
            
            # Serialize unmatched bank transactions
            unmatched_bank_data = []
            for tx in unmatched_bank_transactions:
                unmatched_bank_data.append({
                    'id': tx.id,
                    'bank': tx.bank.name if tx.bank else None,
                    'operation_date': tx.operation_date,
                    'value_date': tx.value_date,
                    'label': tx.label,
                    'amount': float(tx.amount),
                    'debit': tx.debit,
                    'credit': tx.credit,
                    'ref': tx.ref,
                    'document_reference': tx.document_reference,
                    'internal_number': tx.internal_number,
                    'type': tx.type,
                    'payment_class': tx.payment_class.name if tx.payment_class else None,
                    'payment_status': tx.payment_status.name if tx.payment_status else None,
                    'date_ref': tx.date_ref
                })
            
            # === 4. Get unmatched customer transactions ===
            # Customer transactions without matched_bank_transaction
            unmatched_customer_transactions = RecoCustomerTransaction.objects.filter(matched_bank_transaction__isnull=True)
            
            # Filter by account_number if provided from frontend
            account_number = request.query_params.get('account_number')
            if account_number:
                unmatched_customer_transactions = unmatched_customer_transactions.filter(account_number=account_number)
            
            # Serialize unmatched customer transactions
            unmatched_customer_data = []
            for tx in unmatched_customer_transactions:
                unmatched_customer_data.append({
                    'id': tx.id,
                    'account_number': tx.account_number,
                    'accounting_date': tx.accounting_date,
                    'document_number': tx.document_number,
                    'description': tx.description,
                    'debit_amount': float(tx.debit_amount) if tx.debit_amount else None,
                    'credit_amount': float(tx.credit_amount) if tx.credit_amount else None,
                    'amount': float(tx.amount),
                    'total_amount': float(tx.total_amount) if tx.total_amount else None,
                    'external_doc_number': tx.external_doc_number,
                    'due_date': tx.due_date,
                    'payment_type': tx.payment_type,
                    'payment_status': tx.payment_status.name if tx.payment_status else None,
                    'customer_ledger_entry': tx.customer_ledger_entry.name if tx.customer_ledger_entry else None,
                    'company': tx.customer_ledger_entry.company.name if tx.customer_ledger_entry and tx.customer_ledger_entry.company else None
                })
            
            # === 5. Prepare response ===
            response_data = {
                "summary": {
                    "total_bank_transactions": all_bank_transactions.count(),
                    "high_matches_count": len(high_match_bank_ids),
                    "unmatched_bank_transactions_count": len(unmatched_bank_data),
                    "unmatched_customer_transactions_count": len(unmatched_customer_data),
                    "latest_high_matches_file": os.path.basename(latest_high_matches_file)
                },
                "unmatched_bank_transactions": unmatched_bank_data,
                "unmatched_customer_transactions": unmatched_customer_data
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


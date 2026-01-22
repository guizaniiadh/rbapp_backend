"""
Attijari BANK Specific Views

This module contains all Attijari-specific views that handle:
- Attijari bank ledger preprocessing
- Attijari-specific matching algorithms (sensitive matching rules)
- Attijari beginning balance extraction
- Attijari customer tax extraction
- Attijari transaction matching
- Other Attijari-specific processing

All views inherit from base classes in base.py and implement Attijari-specific logic.

This module also contains Attijari-specific helper methods/functions that are used
by the views but are specific to Attijari's data format and business rules.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, parsers
from django.db.models import Max, Q, Sum
from django.conf import settings
from django.db import connection
import pandas as pd
import numpy as np
import os
import re
import ast
import uuid
import logging
import glob
import json
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from collections import defaultdict
from rapidfuzz import fuzz

from rbapp.models import (
    BankLedgerEntry, RecoBankTransaction, RecoCustomerTransaction,
    PaymentIdentification, PaymentClass, CustomerLedgerEntry,
    CustomerTaxRow, Agency, Convention, TaxRule, ConventionParameter,
    Comparison, Tax, Bank, Company
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

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


def sanitize_dataframe_for_json(df):
    """
    Replace inf, -inf, and nan values in DataFrame with None (which becomes null in JSON).
    This prevents JSON serialization errors.
    """
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf, np.nan], None)
    return df_clean


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
        series = series.fillna('')
        cleaned = series.astype(str)
        cleaned = cleaned.str.replace(r'\s+', '', regex=True)
        cleaned = cleaned.str.replace(',', '.', regex=False)
        return pd.to_numeric(cleaned, errors='coerce').round(3)
    except Exception as e:
        try:
            return pd.to_numeric(series, errors='coerce').round(3)
        except Exception as e2:
            return pd.Series([0.0] * len(series))


def is_valid_date(d, m):
    try:
        day, month = int(d), int(m)
        return 1 <= day <= 31 and 1 <= month <= 12
    except:
        return False


def extract_info(libelle):
    libelle = str(libelle)
    date_ref = None
    used_date_number = None

    m3 = re.search(r'(\d{2})\s+(\d{2})\s+(20\d{2})', libelle)
    if m3 and is_valid_date(m3.group(1), m3.group(2)):
        try:
            date_ref = datetime.strptime(f"{m3.group(1)} {m3.group(2)} {m3.group(3)}", "%d %m %Y").date()
        except ValueError:
            date_ref = None
        used_date_number = f"{m3.group(1)}{m3.group(2)}{m3.group(3)[-2:]}"
    else:
        m2 = re.search(r'(\d{2})\s+(\d{2})\s+(\d{2})', libelle)
        if m2 and is_valid_date(m2.group(1), m2.group(2)):
            try:
                year = f"20{m2.group(3)}" if int(m2.group(3)) < 50 else f"19{m2.group(3)}"
                date_ref = datetime.strptime(f"{m2.group(1)} {m2.group(2)} {year}", "%d %m %Y").date()
            except ValueError:
                date_ref = None
            used_date_number = f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
        else:
            # Match compact date DDMMYY, but ensure it's not followed by additional digits
            # This prevents matching "010155" when it's part of "010155450017"
            m1 = re.search(r'(\d{2})(\d{2})(\d{2})(?!\d)', libelle)
            if m1 and is_valid_date(m1.group(1), m1.group(2)):
                try:
                    year = f"20{m1.group(3)}" if int(m1.group(3)) < 50 else f"19{m1.group(3)}"
                    date_ref = datetime.strptime(f"{m1.group(1)} {m1.group(2)} {year}", "%d %m %Y").date()
                except ValueError:
                    date_ref = None
                used_date_number = f"{m1.group(1)}{m1.group(2)}{m1.group(3)}"

    ref_match = re.findall(r'\d{5,}', libelle)
    ref = None
    for r in ref_match:
        if used_date_number is None or used_date_number not in r:
            ref = r
            break

    return pd.Series([date_ref if date_ref else None, ref if ref else None])


def get_holiday_dates():
    holidays_param = ConventionParameter.objects.filter(name='holidays').first()
    holidays = set()
    if holidays_param and holidays_param.value:
        for entry in holidays_param.value:
            if isinstance(entry, str):
                entry = entry.strip()
                m = re.match(r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})", entry)
                if m:
                    start = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                    end = datetime.strptime(m.group(2), "%Y-%m-%d").date()
                    for i in range((end - start).days + 1):
                        holidays.add(start + timedelta(days=i))
                else:
                    try:
                        holidays.add(datetime.strptime(entry, "%Y-%m-%d").date())
                    except Exception:
                        pass
    return holidays


def business_days_excluding_holidays(start_date, end_date, holidays):
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    day_count = 0
    curr = start_date
    while curr <= end_date:
        if curr.weekday() < 5 and curr not in holidays:
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


def score_amount_exact(amount1, amount2):
    try:
        return 100 if float(amount1) == float(amount2) else 0
    except Exception:
        return 0


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


def clean_customer_accounting_dataframe(df_comptable):
    """
    Clean customer accounting dataframe by finding the header row and extracting relevant columns.
    """
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

    header_row_index = None
    for i, row in df_comptable.iterrows():
        row_cleaned = [str(cell).strip() for cell in row]
        match_count = sum(1 for cell in row_cleaned if cell in columns_to_keep_clean)
        if match_count >= 3:
            header_row_index = i
            break

    if header_row_index is None:
        raise ValueError("Expected columns not found in the file.")

    df = pd.DataFrame(df_comptable.values[header_row_index + 1:], columns=df_comptable.iloc[header_row_index])
    df.columns = [str(col).strip() for col in df.columns]
    available_columns = [col for col in columns_to_keep_clean if col in df.columns]
    df = df[available_columns]

    return df.reset_index(drop=True)


# ============================================================================
# Attijari-Specific Views
# ============================================================================

class AttijariPreprocessBankLedgerEntryView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.JSONParser, parsers.MultiPartParser, parsers.FormParser]

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
            
            # Normalize columns (keep Référence if it exists)
            target_columns = ['Date Opération', 'Libellé', 'Date Valeur', 'Débit', 'Crédit', 'Référence']
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
            
            # Attijari-specific: Remove negative sign from Débit column (convert all negative values to positive)
            if 'Débit' in df_cleaned.columns:
                # Apply absolute value to ensure all débit values are positive
                # Convert all values to absolute value (negative becomes positive, positive stays positive, 0 stays 0)
                df_cleaned['Débit'] = df_cleaned['Débit'].apply(lambda x: abs(float(x)) if pd.notna(x) else 0)
            # Extract references and dates from Libellé (as fallback)
            if 'Libellé' in df_cleaned.columns:
                extracted = df_cleaned['Libellé'].apply(extract_info)
                extracted.columns = ['date_ref', 'ref_extracted']
                df_cleaned = pd.concat([df_cleaned, extracted], axis=1)
            
            # Use Référence column if present and has value, otherwise use extracted ref from Libellé
            if 'Référence' in df_cleaned.columns:
                # Convert Référence to string and handle NaN/empty values
                df_cleaned['Référence'] = df_cleaned['Référence'].fillna('').astype(str)
                # Use Référence if it has a valid value, otherwise use extracted ref
                df_cleaned['ref'] = df_cleaned.apply(
                    lambda row: row['Référence'] if row['Référence'] and row['Référence'].strip() and row['Référence'] != 'nan' 
                    else (row.get('ref_extracted') if pd.notna(row.get('ref_extracted')) else None),
                    axis=1
                )
            else:
                # No Référence column, use extracted ref from Libellé
                if 'ref_extracted' in df_cleaned.columns:
                    df_cleaned['ref'] = df_cleaned['ref_extracted']
                else:
                    df_cleaned['ref'] = None
            
            # Get the next import batch id (on reco table)
            last_batch = RecoBankTransaction.objects.aggregate(max_id=Max('import_batch_id'))['max_id'] or 0
            import_batch_id = last_batch + 1

            # Get the bank from the agency linked to the ledger entry
            bank = ledger_entry.agency.bank

            # Save transactions to reco database
            created_transactions = []
            for _, row in df_cleaned.iterrows():
                credit = row.get('Crédit', 0) or 0
                debit_raw = row.get('Débit', 0) or 0
                # Attijari-specific: Ensure débit is always positive (remove negative sign)
                # Convert to float and apply absolute value - this ensures negative values become positive
                try:
                    debit = abs(float(debit_raw))
                except (ValueError, TypeError):
                    debit = 0
                amount = credit - debit

                transaction = RecoBankTransaction(
                    bank_ledger_entry=ledger_entry,
                    bank=bank,
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


class AttijariExtractBeginningBalanceView(APIView):
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


class AttijariMatchCustomerBankTransactionsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Check if any PaymentIdentification is grouped
            grouped_mode = PaymentIdentification.objects.filter(grouped=True).exists()

            # === 1. Filter bank transactions with conditions ===
            all_bank_transactions = RecoBankTransaction.objects.all()
            all_payment_classes = PaymentClass.objects.all()
            
            unique_types = RecoBankTransaction.objects.values_list('type', flat=True).distinct()
            unique_payment_classes = PaymentClass.objects.values_list('code', flat=True).distinct()
            
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
                        score_reference = 0
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
            # Also propagate payment_class and payment_status to related customer transactions
            for _, match_row in df_matches_high.iterrows():
                if match_row['customer_transaction_id'] and match_row['bank_transaction_id']:
                    try:
                        customer_transaction = RecoCustomerTransaction.objects.get(id=match_row['customer_transaction_id'])
                        bank_transaction = RecoBankTransaction.objects.get(id=match_row['bank_transaction_id'])
                        customer_transaction.matched_bank_transaction = bank_transaction
                        customer_transaction.save()
                        
                        # Propagate matched_bank_transaction, payment_class and payment_status to all customer transactions with same document_number
                        if customer_transaction.document_number and bank_transaction:
                            # Get payment_class code and payment_status from bank transaction
                            payment_class_code = bank_transaction.payment_class.code if bank_transaction.payment_class else None
                            payment_status = bank_transaction.payment_status
                            
                            # Update all customer transactions with the same document_number
                            related_transactions = RecoCustomerTransaction.objects.filter(
                                document_number=customer_transaction.document_number
                            )
                            
                            updated_count = 0
                            for related_tx in related_transactions:
                                updated = False
                                # Propagate matched_bank_transaction
                                if related_tx.matched_bank_transaction != bank_transaction:
                                    related_tx.matched_bank_transaction = bank_transaction
                                    updated = True
                                # Propagate payment_class
                                if payment_class_code and related_tx.payment_type != payment_class_code:
                                    related_tx.payment_type = payment_class_code
                                    updated = True
                                # Propagate payment_status
                                if payment_status and related_tx.payment_status != payment_status:
                                    related_tx.payment_status = payment_status
                                    updated = True
                                if updated:
                                    related_tx.save()
                                    updated_count += 1
                            
                            if updated_count > 0:
                                logger.info(f"Propagated matched_bank_transaction={bank_transaction.id}, payment_class={payment_class_code} and payment_status={payment_status} to {updated_count} related customer transactions with document_number={customer_transaction.document_number}")
                    except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                        continue
            
            # === 6. Save dataframes to files for later access ===
            output_dir = os.path.join(settings.BASE_DIR, 'matching_results')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            high_matches_file = os.path.join(output_dir, f'high_matches_{timestamp}.csv')
            if not df_matches_high.empty:
                df_matches_high.to_csv(high_matches_file, index=False)
                high_matches_saved = True
                high_matches_path = high_matches_file
            else:
                high_matches_saved = False
                high_matches_path = None
            
            low_matches_file = os.path.join(output_dir, f'low_matches_{timestamp}.csv')
            if not df_matches_low.empty:
                df_matches_low.to_csv(low_matches_file, index=False)
                low_matches_saved = True
                low_matches_path = low_matches_file
            else:
                low_matches_saved = False
                low_matches_path = None
            
            all_matches_file = os.path.join(output_dir, f'all_matches_{timestamp}.csv')
            df_matches.to_csv(all_matches_file, index=False)

            # === 6.5. Promote low matches to high if grouped_mode is True ===
            if grouped_mode:
                rows_to_drop = []
                for idx_low, row_low in df_matches_low.iterrows():
                    match = df_matches_high[
                        (df_matches_high['customer_document_number'] == row_low['customer_document_number'])
                    ]
                    if not match.empty:
                        matched_bank_id = match.iloc[0]['bank_transaction_id']
                        customer_transaction_id = row_low['customer_transaction_id']
                        
                        try:
                            ct = RecoCustomerTransaction.objects.get(id=customer_transaction_id)
                            bt = RecoBankTransaction.objects.get(id=matched_bank_id)
                            matching_customer_transactions = RecoCustomerTransaction.objects.filter(
                                document_number=ct.document_number,
                                description=ct.description,
                                accounting_date=ct.accounting_date
                            )
                            
                            # Get payment_class code and payment_status from bank transaction
                            payment_class_code = bt.payment_class.code if bt.payment_class else None
                            payment_status = bt.payment_status
                            
                            for matching_ct in matching_customer_transactions:
                                matching_ct.matched_bank_transaction = bt
                                # Also propagate payment_class and payment_status
                                if payment_class_code:
                                    matching_ct.payment_type = payment_class_code
                                if payment_status:
                                    matching_ct.payment_status = payment_status
                                matching_ct.save()
                            
                            # Also update all customer transactions with same document_number (broader propagation)
                            if ct.document_number:
                                related_transactions = RecoCustomerTransaction.objects.filter(
                                    document_number=ct.document_number
                                ).exclude(id__in=[m.id for m in matching_customer_transactions])
                                
                                updated_count = 0
                                for related_tx in related_transactions:
                                    updated = False
                                    # Propagate matched_bank_transaction
                                    if related_tx.matched_bank_transaction != bt:
                                        related_tx.matched_bank_transaction = bt
                                        updated = True
                                    # Propagate payment_class
                                    if payment_class_code and related_tx.payment_type != payment_class_code:
                                        related_tx.payment_type = payment_class_code
                                        updated = True
                                    # Propagate payment_status
                                    if payment_status and related_tx.payment_status != payment_status:
                                        related_tx.payment_status = payment_status
                                        updated = True
                                    if updated:
                                        related_tx.save()
                                        updated_count += 1
                                
                                if updated_count > 0:
                                    logger.info(f"Propagated matched_bank_transaction={bt.id}, payment_class={payment_class_code} and payment_status={payment_status} to {updated_count} additional customer transactions with document_number={ct.document_number}")
                        except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                            continue
                        rows_to_drop.append(idx_low)
                df_matches_low_updated = df_matches_low.drop(rows_to_drop)
                low_matches_file_updated = os.path.join(output_dir, f'low_matches_{timestamp}_updated.csv')
                df_matches_low_updated.to_csv(low_matches_file_updated, index=False)
            
            # === 7. Calculate statistics ===
            total_rows = len(df_bank)
            high_match_count = len(df_matches_high)
            low_match_count = len(df_matches_low)
            
            high_match_percentage = (100 * high_match_count / total_rows) if total_rows > 0 else 0
            low_match_percentage = (100 * low_match_count / total_rows) if total_rows > 0 else 0
            
            # === 8. Prepare response data ===
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


class AttijariManualMatchCustomerBankTransactionsView(APIView):
    """
    Manually link a RecoCustomerTransaction to a RecoBankTransaction for Attijari.

    POST body:
    {
        "reco_bank_transaction_id": <int>,
        "reco_customer_transaction_id": <int>
    }

    Effect:
    - Sets RecoCustomerTransaction.matched_bank_transaction to the given RecoBankTransaction.
    - Propagates the same matched bank transaction (and payment metadata) to all
      RecoCustomerTransaction rows with the same document_number.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            bank_tx_id = request.data.get("reco_bank_transaction_id")
            cust_tx_id = request.data.get("reco_customer_transaction_id")

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

            # Link selected customer transaction
            cust_tx.matched_bank_transaction = bank_tx
            cust_tx.save()

            # Propagate to same-document customer transactions
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

                    if bank_tx.payment_class and related.payment_type != bank_tx.payment_class.code:
                        related.payment_type = bank_tx.payment_class.code
                        updated = True
                    if bank_tx.payment_status and related.payment_status_id != bank_tx.payment_status_id:
                        related.payment_status = bank_tx.payment_status
                        updated = True

                    if updated:
                        related.save()
                        propagated_count += 1

            logger.info(
                f"AttijariManualMatchCustomerBankTransactionsView: matched customer_tx={cust_tx_id} to bank_tx={bank_tx_id}, "
                f"propagated_to_same_document={propagated_count}"
            )

            return Response(
                {
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
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(
                f"AttijariManualMatchCustomerBankTransactionsView error: {str(e)}\n{error_details}"
            )
            return Response(
                {
                    "error": str(e),
                    "details": error_details,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AttijariMatchTransactionView(APIView):
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


class AttijariMatchBankTransactionTaxesView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            print("=" * 80)
            print("AttijariMatchBankTransactionTaxesView: Starting processing")
            print("=" * 80)
            logger.info("=" * 80)
            logger.info("AttijariMatchBankTransactionTaxesView: Starting processing")
            logger.info("=" * 80)
            
            # Get all bank transactions (RECO)
            all_transactions = RecoBankTransaction.objects.all()
            total_count = all_transactions.count()
            print(f"Total RecoBankTransaction records: {total_count}")
            logger.info(f"Total RecoBankTransaction records: {total_count}")
            
            # Separate origine and non-origine transactions
            origine_transactions = RecoBankTransaction.objects.filter(type='origine')
            origine_count = origine_transactions.count()
            print(f"Origine transactions (type='origine'): {origine_count}")
            logger.info(f"Origine transactions (type='origine'): {origine_count}")
            
            # Fix the filter - should exclude origine, not double exclude
            non_origine_transactions = RecoBankTransaction.objects.exclude(type='origine')
            non_origine_count = non_origine_transactions.count()
            print(f"Non-origine transactions (type != 'origine'): {non_origine_count}")
            logger.info(f"Non-origine transactions (type != 'origine'): {non_origine_count}")
            
            # Log transactions without type
            no_type_count = RecoBankTransaction.objects.filter(type__isnull=True).count()
            if no_type_count > 0:
                print(f"⚠️ Found {no_type_count} transactions with NULL type")
                logger.warning(f"⚠️ Found {no_type_count} transactions with NULL type")
            
            # Log transactions without internal_number before processing
            no_internal_before = RecoBankTransaction.objects.filter(internal_number__isnull=True).count()
            print(f"Transactions without internal_number (before processing): {no_internal_before}")
            logger.info(f"Transactions without internal_number (before processing): {no_internal_before}")
            
            matched_count = 0
            unmatched_count = 0
            origine_updated_count = 0
            
            # Process each non-origine transaction
            logger.info("-" * 80)
            logger.info("Processing non-origine transactions...")
            for idx, non_origine in enumerate(non_origine_transactions, 1):
                matched = False
                logger.debug(f"[{idx}/{non_origine_count}] Processing non-origine transaction ID={non_origine.id}, type={non_origine.type}, ref={non_origine.ref}, date_ref={non_origine.date_ref}")
                
                # First, try to match by ref
                if non_origine.ref:
                    matching_origine = origine_transactions.filter(ref=non_origine.ref).first()
                    if matching_origine:
                        logger.debug(f"  ✓ Matched by ref='{non_origine.ref}' to origine transaction ID={matching_origine.id}")
                        if matching_origine.internal_number:
                            non_origine.internal_number = matching_origine.internal_number
                            logger.debug(f"  → Assigned existing internal_number: {matching_origine.internal_number}")
                        else:
                            new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                            matching_origine.internal_number = new_internal_number
                            matching_origine.save()
                            non_origine.internal_number = new_internal_number
                            logger.debug(f"  → Generated new internal_number: {new_internal_number} (for both)")
                        
                        # Copy payment class, status, and accounting account from origine
                        non_origine.payment_class = matching_origine.payment_class
                        non_origine.payment_status = matching_origine.payment_status
                        non_origine.accounting_account = matching_origine.accounting_account
                        logger.debug(f"  → Copied payment_class={matching_origine.payment_class}, payment_status={matching_origine.payment_status}, accounting_account={matching_origine.accounting_account}")
                        
                        non_origine.save()
                        matched = True
                        matched_count += 1
                        logger.debug(f"  ✓ Saved non-origine transaction ID={non_origine.id} with internal_number={non_origine.internal_number}")
                    else:
                        logger.debug(f"  ✗ No origine transaction found with ref='{non_origine.ref}'")
                
                # If not matched by ref, try to match by date_ref
                if not matched and non_origine.date_ref:
                    matching_origine = origine_transactions.filter(date_ref=non_origine.date_ref).first()
                    if matching_origine:
                        logger.debug(f"  ✓ Matched by date_ref={non_origine.date_ref} to origine transaction ID={matching_origine.id}")
                        if matching_origine.internal_number:
                            non_origine.internal_number = matching_origine.internal_number
                            logger.debug(f"  → Assigned existing internal_number: {matching_origine.internal_number}")
                        else:
                            new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                            matching_origine.internal_number = new_internal_number
                            matching_origine.save()
                            non_origine.internal_number = new_internal_number
                            logger.debug(f"  → Generated new internal_number: {new_internal_number} (for both)")
                        
                        # Copy payment class, status, and accounting account from origine
                        non_origine.payment_class = matching_origine.payment_class
                        non_origine.payment_status = matching_origine.payment_status
                        non_origine.accounting_account = matching_origine.accounting_account
                        logger.debug(f"  → Copied payment_class={matching_origine.payment_class}, payment_status={matching_origine.payment_status}, accounting_account={matching_origine.accounting_account}")
                        
                        non_origine.save()
                        matched = True
                        matched_count += 1
                        logger.debug(f"  ✓ Saved non-origine transaction ID={non_origine.id} with internal_number={non_origine.internal_number}")
                    else:
                        logger.debug(f"  ✗ No origine transaction found with date_ref={non_origine.date_ref}")
                
                # If still not matched, generate a new internal_number
                if not matched:
                    new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                    non_origine.internal_number = new_internal_number
                    non_origine.save()
                    unmatched_count += 1
                    logger.debug(f"  → No match found, generated new internal_number: {new_internal_number} for transaction ID={non_origine.id}")
            
            print(f"Non-origine processing complete: {matched_count} matched, {unmatched_count} unmatched")
            logger.info(f"Non-origine processing complete: {matched_count} matched, {unmatched_count} unmatched")
            
            # Also ensure all origine transactions have internal_number
            print("-" * 80)
            print("Ensuring all origine transactions have internal_number...")
            logger.info("-" * 80)
            logger.info("Ensuring all origine transactions have internal_number...")
            for idx, origine in enumerate(origine_transactions, 1):
                if not origine.internal_number:
                    new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                    origine.internal_number = new_internal_number
                    origine.save()
                    origine_updated_count += 1
                    logger.debug(f"[{idx}/{origine_count}] Generated internal_number={new_internal_number} for origine transaction ID={origine.id}")
                else:
                    logger.debug(f"[{idx}/{origine_count}] Origine transaction ID={origine.id} already has internal_number={origine.internal_number}")
            
            print(f"Origine processing complete: {origine_updated_count} updated with new internal_number")
            logger.info(f"Origine processing complete: {origine_updated_count} updated with new internal_number")

            # === Custom logic for agios transactions (Probabilistic Group-Based Allocation) ===
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
                        agios_tax_rows = CustomerTaxRow.objects.filter(
                            transaction__in=cust_tx_group,
                            tax_type__iexact='agios'
                        )
                        
                        logger.debug(f"Agios {agios.id}, doc_number={doc_number}: Found {agios_tax_rows.count()} agios tax rows for {len(cust_tx_group)} customer transactions")
                        
                        simulated_agios = agios_tax_rows.aggregate(
                            total=Sum('tax_amount')
                        )['total'] or 0
                        
                        if simulated_agios == 0:
                            no_tax_rows_docs.append(doc_number)
                            logger.warning(f"Agios {agios.id}, doc_number={doc_number}: No agios tax rows found or total_tax_amount is 0 (checked {agios_tax_rows.count()} rows, {len(cust_tx_group)} customer transactions)")
                            continue
                        
                        simulated_agios_abs = abs(float(simulated_agios))
                        
                        # STEP 4: Match using tolerance (5%)
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
                            for m in mismatch_docs[:3]:
                                logger.warning(f"    * {m['doc_number']}: simulated={m['simulated']:.2f}, bank={m['bank']:.2f}, ratio={m['tolerance_ratio']:.4f}")
                    
                    # If no individual match found, try matching sum of all agios in this date_ref group
                    if not best_match and len(agios_group) > 1:
                        logger.info(f"Agios {agios.id}: No individual match found, trying to match sum of all {len(agios_group)} agios for date_ref={date_ref}")
                        total_agios_sum = sum(abs(float(a.amount)) for a in agios_group)
                        logger.info(f"Total agios sum for date_ref {date_ref}: {total_agios_sum}")
                        
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
                            matching_origine = RecoBankTransaction.objects.filter(
                                type='origine',
                                operation_date=cust_tx.accounting_date,
                                amount=cust_tx.total_amount
                            ).first()
                            
                            if matching_origine:
                                customer_amount = abs(float(cust_tx.total_amount or cust_tx.amount))
                                allocation_ratio = customer_amount / total_customer_amount
                                
                                logger.info(f"Agios {agios.id}: Allocating to origine {matching_origine.id} via customer transaction {cust_tx.id} (ratio={allocation_ratio:.4f}, customer_amount={customer_amount}, total={total_customer_amount})")
                                
                                agios.internal_number = matching_origine.internal_number
                                agios.payment_class = matching_origine.payment_class
                                agios.payment_status = matching_origine.payment_status
                                agios.accounting_account = matching_origine.accounting_account
                                agios.save()
                                
                                matched = True
                                origine_found = True
                                break
                        
                        if matched:
                            agios_matched_count += 1
                        elif not origine_found:
                            logger.warning(f"Agios {agios.id}: Amount matched but no origine transaction found for customer group (doc_number={best_match['doc_number']})")
                            agios_unmatched_count += 1
                    else:
                        logger.warning(f"Agios {agios.id}: Could not find matching customer group within 5% tolerance (amount={agios.amount}, date_ref={date_ref})")
                        agios_unmatched_count += 1
            
            logger.info(f"Agios matching completed: {agios_matched_count} matched, {agios_unmatched_count} unmatched out of {agios_transactions.count()} total")

            # Final statistics
            no_internal_after = RecoBankTransaction.objects.filter(internal_number__isnull=True).count()
            with_internal_after = RecoBankTransaction.objects.filter(internal_number__isnull=False).count()
            
            print("=" * 80)
            print("AttijariMatchBankTransactionTaxesView: Final Statistics")
            print("=" * 80)
            print(f"Total transactions: {total_count}")
            print(f"Origine transactions: {origine_count} (updated: {origine_updated_count})")
            print(f"Non-origine transactions: {non_origine_count} (matched: {matched_count}, unmatched: {unmatched_count})")
            print(f"Agios transactions: {agios_transactions.count()} (matched: {agios_matched_count}, unmatched: {agios_unmatched_count})")
            print(f"Transactions WITH internal_number: {with_internal_after}")
            print(f"Transactions WITHOUT internal_number: {no_internal_after}")
            print(f"Internal number assignment: {with_internal_after}/{total_count} ({100*with_internal_after/total_count if total_count > 0 else 0:.1f}%)")
            print("=" * 80)
            logger.info("=" * 80)
            logger.info("AttijariMatchBankTransactionTaxesView: Final Statistics")
            logger.info("=" * 80)
            logger.info(f"Total transactions: {total_count}")
            logger.info(f"Origine transactions: {origine_count} (updated: {origine_updated_count})")
            logger.info(f"Non-origine transactions: {non_origine_count} (matched: {matched_count}, unmatched: {unmatched_count})")
            logger.info(f"Agios transactions: {agios_transactions.count()} (matched: {agios_matched_count}, unmatched: {agios_unmatched_count})")
            logger.info(f"Transactions WITH internal_number: {with_internal_after}")
            logger.info(f"Transactions WITHOUT internal_number: {no_internal_after}")
            logger.info(f"Internal number assignment: {with_internal_after}/{total_count} ({100*with_internal_after/total_count if total_count > 0 else 0:.1f}%)")
            logger.info("=" * 80)

            return Response({
                "message": f"Successfully processed reco bank transaction taxes matching",
                "total_transactions": total_count,
                "origine_transactions": origine_count,
                "origine_updated_count": origine_updated_count,
                "non_origine_transactions": non_origine_count,
                "matched_count": matched_count,
                "unmatched_count": unmatched_count,
                "agios_matched_count": agios_matched_count,
                "agios_unmatched_count": agios_unmatched_count,
                "total_with_internal_number": with_internal_after,
                "total_without_internal_number": no_internal_after,
                "internal_number_coverage_percent": round(100*with_internal_after/total_count if total_count > 0 else 0, 1)
            }, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error("=" * 80)
            logger.error(f"AttijariMatchBankTransactionTaxesView: ERROR occurred")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Traceback:\n{error_details}")
            logger.error("=" * 80)
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)


class AttijariExtractCustomerTaxesView(APIView):
    """
    View to extract tax rows for customer transactions based on conventions and smart formula engine.
    """
    def post(self, request, *args, **kwargs):
        try:
            print("=" * 80)
            print("AttijariExtractCustomerTaxesView: Starting tax extraction")
            print("=" * 80)
            logger.info("=" * 80)
            logger.info("AttijariExtractCustomerTaxesView: Starting tax extraction")
            logger.info("=" * 80)
            
            transactions = request.data.get("transactions", None)
            extracted_taxes = []
            
            if transactions is None:
                print("No transactions provided in request, fetching all RecoCustomerTransaction objects")
                logger.info("No transactions in request body, fetching all transactions")
                transactions = RecoCustomerTransaction.objects.all()
                print(f"Fetched {transactions.count()} transactions from database")
            else:
                print(f"Received {len(transactions)} transactions from request body")
                logger.info(f"Received {len(transactions)} transactions from request body")
                # Convert transaction IDs to objects if needed
                if isinstance(transactions, list) and len(transactions) > 0:
                    if isinstance(transactions[0], (int, str)):
                        print(f"Converting transaction IDs to objects...")
                        transaction_ids = [int(tx_id) for tx_id in transactions]
                        transactions = RecoCustomerTransaction.objects.filter(id__in=transaction_ids)
                        print(f"Found {transactions.count()} transactions from IDs")
                        logger.info(f"Converted {len(transaction_ids)} IDs to {transactions.count()} transaction objects")

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
                        return getattr(rule, 'rate', getattr(rule, 'value', None))
                    elif rule.calculation_type == "formula" and rule.formula:
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
                    if isinstance(value, str):
                        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
                            try:
                                return datetime.strptime(value, fmt).date()
                            except Exception:
                                continue
                    return value
                return None

            total_transactions = transactions.count() if hasattr(transactions, 'count') else len(list(transactions))
            print(f"\nProcessing {total_transactions} transactions...")
            logger.info(f"Processing {total_transactions} transactions")
            
            processed_count = 0
            skipped_no_agency = 0
            skipped_no_conventions = 0
            skipped_no_tax_rules = 0
            flat_tax_count = 0
            formula_tax_count = 0
            error_count = 0
            
            for idx, tx in enumerate(transactions, 1):
                try:
                    account_number = tx.account_number
                    tx_vars = {k: v for k, v in tx.__dict__.items() if not k.startswith('_')}
                    reference = tx.document_number
                    payment_class = tx.payment_type
                    payment_status = tx.payment_status
                    
                    if idx <= 5 or idx % 50 == 0:  # Log first 5 and every 50th
                        print(f"\n[{idx}/{total_transactions}] Processing transaction ID={tx.id}")
                        print(f"  - Account number: {account_number}")
                        print(f"  - Reference: {reference}")
                        print(f"  - Payment class: {payment_class}")
                        print(f"  - Payment status: {payment_status}")
                        logger.debug(f"Transaction {idx}/{total_transactions} - ID={tx.id}, account={account_number}, ref={reference}, payment_class={payment_class}, payment_status={payment_status}")
                    
                    try:
                        agency = Agency.objects.get(code=account_number)
                        bank = agency.bank
                        if idx <= 5:
                            print(f"  ✓ Found agency: {agency.code} → Bank: {bank.name}")
                            logger.debug(f"Found agency {agency.code} for bank {bank.name}")
                    except Agency.DoesNotExist:
                        skipped_no_agency += 1
                        if idx <= 10:
                            print(f"  ✗ Agency not found for account_number: {account_number}")
                            logger.warning(f"Agency not found for account_number: {account_number} (transaction ID={tx.id})")
                        continue
                    
                    conventions = Convention.objects.filter(bank=bank, is_active=True).order_by("-id")
                    conventions_count = conventions.count()
                    
                    if conventions_count == 0:
                        skipped_no_conventions += 1
                        if idx <= 10:
                            print(f"  ✗ No active conventions found for bank: {bank.name}")
                            logger.warning(f"No active conventions for bank {bank.name} (transaction ID={tx.id})")
                        continue
                    
                    if idx <= 5:
                        print(f"  ✓ Found {conventions_count} active convention(s) for bank {bank.name}")
                        logger.debug(f"Found {conventions_count} conventions for bank {bank.name}")
                    for conv_idx, conv in enumerate(conventions, 1):
                        if idx <= 5:
                            print(f"    [{conv_idx}/{conventions_count}] Processing convention: {conv.name} (ID={conv.id})")
                            logger.debug(f"Processing convention {conv.name} (ID={conv.id}) for transaction {tx.id}")
                        
                        tax_rules_queryset = TaxRule.objects.filter(
                            convention=conv,
                            payment_class__code=payment_class,
                            payment_status=payment_status
                        ).order_by('-id')
                        tax_rules_count_before = tax_rules_queryset.count()
                        
                        if idx <= 5:
                            print(f"      - Found {tax_rules_count_before} tax rule(s) matching payment_class={payment_class}, payment_status={payment_status}")
                            logger.debug(f"Found {tax_rules_count_before} tax rules for convention {conv.name}")
                        
                        seen_tax_types = {}
                        tax_rules = []
                        for tr in tax_rules_queryset:
                            if hasattr(tr, 'tax_type') and tr.tax_type:
                                key = tr.tax_type.lower().strip()
                                if key not in seen_tax_types:
                                    seen_tax_types[key] = tr
                                    tax_rules.append(tr)
                        
                        if len(tax_rules) == 0:
                            skipped_no_tax_rules += 1
                            if idx <= 10:
                                print(f"      ✗ No valid tax rules after deduplication")
                                logger.warning(f"No valid tax rules for convention {conv.name} (transaction ID={tx.id})")
                            continue
                        
                        if idx <= 5:
                            print(f"      - After deduplication: {len(tax_rules)} unique tax rule(s)")
                            for rule_idx, rule in enumerate(tax_rules, 1):
                                print(f"        [{rule_idx}] {rule.tax_type} ({rule.calculation_type})")
                        
                        params = {p.name: p.value for p in ConventionParameter.objects.all()}
                        if idx <= 5:
                            print(f"      - Loaded {len(params)} convention parameter(s)")
                            logger.debug(f"Loaded {len(params)} convention parameters")
                        
                        tax_rule_lookup = {}
                        for tr in tax_rules:
                            if hasattr(tr, 'tax_type') and tr.tax_type:
                                key = tr.tax_type.lower().strip()
                                tax_rule_lookup[key] = tr
                        
                        tax_results = {}
                        for rule_idx, rule in enumerate(tax_rules, 1):
                            try:
                                if rule.calculation_type == "flat":
                                    val = getattr(rule, 'rate', getattr(rule, 'value', None))
                                    if idx <= 5:
                                        print(f"        [{rule_idx}] Processing FLAT tax: {rule.tax_type} = {val}")
                                        logger.debug(f"Processing flat tax {rule.tax_type} = {val}")
                                    
                                    try:
                                        val_decimal = Decimal(str(val)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                                        val_str = format(val_decimal, '.3f')
                                    except Exception as e:
                                        val_str = str(val)
                                        if idx <= 5:
                                            print(f"          ⚠️  Decimal conversion warning: {e}")
                                    
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
                                    flat_tax_count += 1
                                    
                                    try:
                                        tax_row = CustomerTaxRow.objects.create(
                                            transaction=tx,
                                            tax_type=rule.tax_type,
                                            tax_amount=val,
                                            applied_formula=None,
                                            rate_used=getattr(rule, 'rate', None)
                                        )
                                        
                                        total_tax_amount = CustomerTaxRow.objects.filter(
                                            transaction__document_number=tx.document_number,
                                            tax_type=rule.tax_type
                                        ).aggregate(total=Sum('tax_amount'))['total'] or 0
                                        
                                        CustomerTaxRow.objects.filter(
                                            transaction__document_number=tx.document_number,
                                            tax_type=rule.tax_type
                                        ).update(total_tax_amount=total_tax_amount)
                                        
                                        if idx <= 5:
                                            print(f"          ✓ Created CustomerTaxRow: ID={tax_row.id}, amount={val}, total={total_tax_amount}")
                                            logger.debug(f"Created CustomerTaxRow ID={tax_row.id} for tax {rule.tax_type}")
                                    except Exception as e:
                                        error_count += 1
                                        print(f"          ❌ CustomerTaxRow flat error: {e}")
                                        logger.error(f"Error creating CustomerTaxRow for flat tax {rule.tax_type} (transaction ID={tx.id}): {e}", exc_info=True)
                                elif rule.calculation_type == "formula" and rule.formula:
                                    if idx <= 5:
                                        print(f"        [{rule_idx}] Processing FORMULA tax: {rule.tax_type}")
                                        print(f"          Formula: {rule.formula}")
                                        logger.debug(f"Processing formula tax {rule.tax_type} with formula: {rule.formula}")
                                    
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
                                        
                                        if idx <= 5:
                                            print(f"          Variables found: {formula_vars}")
                                            logger.debug(f"Formula variables: {formula_vars}")
                                    except Exception as e:
                                        error_count += 1
                                        error_msg = f"Invalid formula for {rule.tax_type}: {str(e)}"
                                        if idx <= 5:
                                            print(f"          ❌ {error_msg}")
                                        logger.error(f"Invalid formula for {rule.tax_type} (transaction ID={tx.id}): {e}", exc_info=True)
                                        extracted_taxes.append({
                                            "error": error_msg,
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
                                            if idx <= 5:
                                                print(f"          ⚠️  Variable '{var}' not resolved")
                                                logger.debug(f"Variable '{var}' not resolved for formula {rule.tax_type}")
                                        else:
                                            import decimal
                                            if isinstance(value, decimal.Decimal):
                                                value = float(value)
                                            local_vars[var] = value
                                            if idx <= 5:
                                                print(f"          ✓ Variable '{var}' = {value}")
                                    
                                    if missing:
                                        error_msg = f"Missing variables: {', '.join(missing)}"
                                        if idx <= 5:
                                            print(f"          ❌ {error_msg}")
                                        logger.warning(f"Missing variables for formula {rule.tax_type} (transaction ID={tx.id}): {missing}")
                                        extracted_taxes.append({
                                            "error": error_msg,
                                            "bank": bank.name,
                                            "transaction_reference": reference,
                                            "convention": conv.name,
                                            "tax_rule": rule.id,
                                        })
                                        continue
                                    
                                    def custom_eval(expr, local_vars):
                                        import operator
                                        def date_sub(a, b):
                                            if isinstance(a, (date, datetime)) and isinstance(b, (date, datetime)):
                                                return abs((a - b).days)
                                            raise TypeError("date_sub only supports date - date")
                                        safe_locals = dict(local_vars)
                                        safe_locals['date_sub'] = date_sub
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
                                            if isinstance(result, timedelta):
                                                return result.days
                                            return result
                                        except Exception as e:
                                            raise e
                                    
                                    try:
                                        expr = rule.formula
                                        if idx <= 5:
                                            print(f"          Evaluating formula with local_vars: {local_vars}")
                                        result = custom_eval(expr, local_vars)
                                        
                                        if idx <= 5:
                                            print(f"          ✓ Formula result: {result}")
                                            logger.debug(f"Formula result for {rule.tax_type}: {result}")
                                        
                                        try:
                                            result_decimal = Decimal(str(result)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                                            result_str = format(result_decimal, '.3f')
                                        except Exception as e:
                                            result_str = str(result)
                                            if idx <= 5:
                                                print(f"          ⚠️  Decimal conversion warning: {e}")
                                        
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
                                        formula_tax_count += 1
                                        
                                        try:
                                            tax_row = CustomerTaxRow.objects.create(
                                                transaction=tx,
                                                tax_type=rule.tax_type,
                                                tax_amount=result,
                                                applied_formula=rule.formula,
                                                rate_used=getattr(rule, 'rate', None)
                                            )
                                            
                                            total_tax_amount = CustomerTaxRow.objects.filter(
                                                transaction__document_number=tx.document_number,
                                                tax_type=rule.tax_type
                                            ).aggregate(total=Sum('tax_amount'))['total'] or 0
                                            
                                            CustomerTaxRow.objects.filter(
                                                transaction__document_number=tx.document_number,
                                                tax_type=rule.tax_type
                                            ).update(total_tax_amount=total_tax_amount)
                                            
                                            if idx <= 5:
                                                print(f"          ✓ Created CustomerTaxRow: ID={tax_row.id}, amount={result}, total={total_tax_amount}")
                                                logger.debug(f"Created CustomerTaxRow ID={tax_row.id} for formula tax {rule.tax_type}")
                                        except Exception as e:
                                            error_count += 1
                                            print(f"          ❌ CustomerTaxRow formula error: {e}")
                                            logger.error(f"Error creating CustomerTaxRow for formula tax {rule.tax_type} (transaction ID={tx.id}): {e}", exc_info=True)
                                    except Exception as e:
                                        error_count += 1
                                        error_msg = f"Evaluation error for {rule.tax_type}: {str(e)}"
                                        if idx <= 5:
                                            print(f"          ❌ {error_msg}")
                                        logger.error(f"Evaluation error for {rule.tax_type} (transaction ID={tx.id}): {e}", exc_info=True)
                                        extracted_taxes.append({
                                            "error": error_msg,
                                            "bank": bank.name,
                                            "transaction_reference": reference,
                                            "convention": conv.name,
                                            "tax_rule": rule.id,
                                        })
                            except Exception as e:
                                error_count += 1
                                logger.error(f"Error processing tax rule {rule.tax_type if hasattr(rule, 'tax_type') else 'unknown'} (transaction ID={tx.id}): {e}", exc_info=True)
                                if idx <= 5:
                                    print(f"        ❌ Error processing tax rule: {e}")
                    
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing transaction {idx if 'idx' in locals() else 'unknown'}: {e}", exc_info=True)
                    if idx <= 10:
                        print(f"  ❌ Error processing transaction: {e}")
                    continue
            
            print(f"\n{'=' * 80}")
            print("AttijariExtractCustomerTaxesView: Final Statistics")
            print(f"{'=' * 80}")
            print(f"Total transactions: {total_transactions}")
            print(f"Processed: {processed_count}")
            print(f"Skipped (no agency): {skipped_no_agency}")
            print(f"Skipped (no conventions): {skipped_no_conventions}")
            print(f"Skipped (no tax rules): {skipped_no_tax_rules}")
            print(f"Flat taxes created: {flat_tax_count}")
            print(f"Formula taxes created: {formula_tax_count}")
            print(f"Errors: {error_count}")
            print(f"Total extracted taxes: {len(extracted_taxes)}")
            print(f"{'=' * 80}")
            
            logger.info("=" * 80)
            logger.info("AttijariExtractCustomerTaxesView: Final Statistics")
            logger.info(f"Total: {total_transactions}, Processed: {processed_count}")
            logger.info(f"Skipped - Agency: {skipped_no_agency}, Conventions: {skipped_no_conventions}, Tax Rules: {skipped_no_tax_rules}")
            logger.info(f"Taxes - Flat: {flat_tax_count}, Formula: {formula_tax_count}, Errors: {error_count}")
            logger.info(f"Total extracted: {len(extracted_taxes)}")
            logger.info("=" * 80)
            
            return Response({
                "extracted_taxes": extracted_taxes,
                "statistics": {
                    "total_transactions": total_transactions,
                    "processed": processed_count,
                    "skipped_no_agency": skipped_no_agency,
                    "skipped_no_conventions": skipped_no_conventions,
                    "skipped_no_tax_rules": skipped_no_tax_rules,
                    "flat_taxes": flat_tax_count,
                    "formula_taxes": formula_tax_count,
                    "errors": error_count,
                    "total_extracted": len(extracted_taxes)
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n{'=' * 80}")
            print("❌ CRITICAL ERROR in AttijariExtractCustomerTaxesView")
            print(f"{'=' * 80}")
            print(f"Error: {str(e)}")
            print(f"\nTraceback:\n{error_details}")
            print(f"{'=' * 80}")
            logger.error("=" * 80)
            logger.error("AttijariExtractCustomerTaxesView: CRITICAL ERROR")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Traceback:\n{error_details}")
            logger.error("=" * 80)
            return Response({
                "error": f"An error occurred: {str(e)}",
                "details": error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AttijariMatchCustomerBankTransactionsView1(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Check if any PaymentIdentification is grouped
            grouped_mode = PaymentIdentification.objects.filter(grouped=True).exists()

            # === 1. Filter bank transactions with conditions ===
            all_bank_transactions = RecoBankTransaction.objects.all()
            all_payment_classes = PaymentClass.objects.all()
            
            unique_types = RecoBankTransaction.objects.values_list('type', flat=True).distinct()
            unique_payment_classes = PaymentClass.objects.values_list('code', flat=True).distinct()
            
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

                    if grouped_mode:
                        score_amount = score_amount_exact(
                            customer_row['total_amount'],
                            bank_row['amount']
                        )
                        score_reference = 0
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
            # Also propagate payment_class and payment_status to related customer transactions
            for _, match_row in df_matches_high.iterrows():
                if match_row['customer_transaction_id'] and match_row['bank_transaction_id']:
                    try:
                        customer_transaction = RecoCustomerTransaction.objects.get(id=match_row['customer_transaction_id'])
                        bank_transaction = RecoBankTransaction.objects.get(id=match_row['bank_transaction_id'])
                        customer_transaction.matched_bank_transaction = bank_transaction
                        customer_transaction.save()
                        
                        # Propagate matched_bank_transaction, payment_class and payment_status to all customer transactions with same document_number
                        if customer_transaction.document_number and bank_transaction:
                            # Get payment_class code and payment_status from bank transaction
                            payment_class_code = bank_transaction.payment_class.code if bank_transaction.payment_class else None
                            payment_status = bank_transaction.payment_status
                            
                            # Update all customer transactions with the same document_number
                            related_transactions = RecoCustomerTransaction.objects.filter(
                                document_number=customer_transaction.document_number
                            )
                            
                            updated_count = 0
                            for related_tx in related_transactions:
                                updated = False
                                # Propagate matched_bank_transaction
                                if related_tx.matched_bank_transaction != bank_transaction:
                                    related_tx.matched_bank_transaction = bank_transaction
                                    updated = True
                                # Propagate payment_class
                                if payment_class_code and related_tx.payment_type != payment_class_code:
                                    related_tx.payment_type = payment_class_code
                                    updated = True
                                # Propagate payment_status
                                if payment_status and related_tx.payment_status != payment_status:
                                    related_tx.payment_status = payment_status
                                    updated = True
                                if updated:
                                    related_tx.save()
                                    updated_count += 1
                            
                            if updated_count > 0:
                                logger.info(f"Propagated matched_bank_transaction={bank_transaction.id}, payment_class={payment_class_code} and payment_status={payment_status} to {updated_count} related customer transactions with document_number={customer_transaction.document_number}")
                    except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                        continue
            
            # === 6. Save dataframes to files for later access ===
            output_dir = os.path.join(settings.BASE_DIR, 'matching_results')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            high_matches_file = os.path.join(output_dir, f'high_matches_{timestamp}.csv')
            if not df_matches_high.empty:
                df_matches_high.to_csv(high_matches_file, index=False)
                high_matches_saved = True
                high_matches_path = high_matches_file
            else:
                high_matches_saved = False
                high_matches_path = None
            
            low_matches_file = os.path.join(output_dir, f'low_matches_{timestamp}.csv')
            if not df_matches_low.empty:
                df_matches_low.to_csv(low_matches_file, index=False)
                low_matches_saved = True
                low_matches_path = low_matches_file
            else:
                low_matches_saved = False
                low_matches_path = None
            
            all_matches_file = os.path.join(output_dir, f'all_matches_{timestamp}.csv')
            df_matches.to_csv(all_matches_file, index=False)

            # === 6.5. Promote low matches to high if grouped_mode is True ===
            if grouped_mode:
                rows_to_drop = []
                for idx_low, row_low in df_matches_low.iterrows():
                    match = df_matches_high[
                        (df_matches_high['customer_document_number'] == row_low['customer_document_number'])
                    ]
                    if not match.empty:
                        matched_bank_id = match.iloc[0]['bank_transaction_id']
                        customer_transaction_id = row_low['customer_transaction_id']
                        
                        try:
                            ct = RecoCustomerTransaction.objects.get(id=customer_transaction_id)
                            bt = RecoBankTransaction.objects.get(id=matched_bank_id)
                            matching_customer_transactions = RecoCustomerTransaction.objects.filter(
                                document_number=ct.document_number,
                                description=ct.description,
                                accounting_date=ct.accounting_date
                            )
                            
                            # Get payment_class code and payment_status from bank transaction
                            payment_class_code = bt.payment_class.code if bt.payment_class else None
                            payment_status = bt.payment_status
                            
                            for matching_ct in matching_customer_transactions:
                                matching_ct.matched_bank_transaction = bt
                                # Also propagate payment_class and payment_status
                                if payment_class_code:
                                    matching_ct.payment_type = payment_class_code
                                if payment_status:
                                    matching_ct.payment_status = payment_status
                                matching_ct.save()
                            
                            # Also update all customer transactions with same document_number (broader propagation)
                            if ct.document_number:
                                related_transactions = RecoCustomerTransaction.objects.filter(
                                    document_number=ct.document_number
                                ).exclude(id__in=[m.id for m in matching_customer_transactions])
                                
                                updated_count = 0
                                for related_tx in related_transactions:
                                    updated = False
                                    # Propagate matched_bank_transaction
                                    if related_tx.matched_bank_transaction != bt:
                                        related_tx.matched_bank_transaction = bt
                                        updated = True
                                    # Propagate payment_class
                                    if payment_class_code and related_tx.payment_type != payment_class_code:
                                        related_tx.payment_type = payment_class_code
                                        updated = True
                                    # Propagate payment_status
                                    if payment_status and related_tx.payment_status != payment_status:
                                        related_tx.payment_status = payment_status
                                        updated = True
                                    if updated:
                                        related_tx.save()
                                        updated_count += 1
                                
                                if updated_count > 0:
                                    logger.info(f"Propagated matched_bank_transaction={bt.id}, payment_class={payment_class_code} and payment_status={payment_status} to {updated_count} additional customer transactions with document_number={ct.document_number}")
                        except (RecoCustomerTransaction.DoesNotExist, RecoBankTransaction.DoesNotExist):
                            continue
                        rows_to_drop.append(idx_low)
                df_matches_low_updated = df_matches_low.drop(rows_to_drop)
                low_matches_file_updated = os.path.join(output_dir, f'low_matches_{timestamp}_updated.csv')
                df_matches_low_updated.to_csv(low_matches_file_updated, index=False)
            
            # === 7. Calculate statistics ===
            total_rows = len(df_bank)
            high_match_count = len(df_matches_high)
            low_match_count = len(df_matches_low)
            
            high_match_percentage = (100 * high_match_count / total_rows) if total_rows > 0 else 0
            low_match_percentage = (100 * low_match_count / total_rows) if total_rows > 0 else 0
            
            # === 8. Prepare response data ===
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


class AttijariPreprocessCustomerLedgerEntryView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [parsers.JSONParser, parsers.MultiPartParser, parsers.FormParser]

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
            
            # Get the agency code from request data (required)
            agency_code = None
            
            if request.content_type == 'application/json':
                agency_code = request.data.get('agency_code') or request.data.get('code')
            elif request.content_type == 'application/x-www-form-urlencoded':
                agency_code = request.POST.get('agency_code') or request.POST.get('code')
            elif request.content_type == 'text/plain':
                try:
                    import json
                    raw_data = request.body.decode('utf-8')
                    parsed_data = json.loads(raw_data)
                    agency_code = parsed_data.get('agency_code') or parsed_data.get('code')
                except:
                    raw_data = request.body.decode('utf-8')
                    if '"code"' in raw_data or '"agency_code"' in raw_data:
                        code_match = re.search(r'"code"\s*:\s*"([^"]+)"', raw_data)
                        agency_match = re.search(r'"agency_code"\s*:\s*"([^"]+)"', raw_data)
                        if code_match:
                            agency_code = code_match.group(1)
                        elif agency_match:
                            agency_code = agency_match.group(1)
            
            # Agency code is required - return error if not provided
            if not agency_code:
                return Response({
                    "error": "Agency code is required. Please provide 'agency_code' or 'code' in the request body.",
                    "message": "The agency code (N° compte bancaire) is required to filter transactions from the customer ledger file."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Filter rows by agency code (N° compte bancaire)
            if 'N° compte bancaire' in df_cleaned.columns:
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
                debit = row.get('Débit', 0) or 0
                credit = row.get('Crédit', 0) or 0
                
                if debit > 0:
                    amount = -debit
                elif credit > 0:
                    amount = credit
                else:
                    amount = 0
                
                accounting_date = row.get('Date comptabilisation')
                if pd.isna(accounting_date):
                    accounting_date = None
                
                due_date = row.get('Date d\'échéance')
                if pd.isna(due_date):
                    due_date = None
                
                # Clean payment type
                payment_type = row.get('Type de règlement', '')
                if payment_type:
                    payment_type = str(payment_type)
                    payment_type = re.sub(r'^\d{2}-\s*', '', payment_type)
                    payment_type = re.sub(r'\b(TUNIS|ZARZIS)\b', '', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bTRAITES\b', 'EFFETS', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bEFFETS CLT\b', 'CLT EFFET', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bVIREMENTS FRS\b', 'FRS VIR', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bVIREMENTS CLT\b', 'CLT VIR', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bEFFETS FRS\b', 'FRS EFFET', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bCHEQUES CLT\b', 'CLT CHQ', payment_type, flags=re.IGNORECASE)
                    payment_type = re.sub(r'\bCHEQUES FRS\b', 'FRS CHQ', payment_type, flags=re.IGNORECASE)
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
                    payment_type=payment_type,
                    amount=amount
                )
                created_transactions.append(transaction)
            
            # Bulk create transactions
            RecoCustomerTransaction.objects.bulk_create(created_transactions)
            
            # Calculate total_amount for all transactions in this batch
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


class AttijariGetMatchingResultsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
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
                        bank_transaction_counts = df['bank_transaction_id'].value_counts()
                        df['is_one_to_many'] = df['bank_transaction_id'].map(bank_transaction_counts) > 1
                        df = df.sort_values(['is_one_to_many', 'bank_transaction_id'], ascending=[False, True])
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


class AttijariTaxComparisonView(APIView):
    def put(self, request):
        """
        For each row in the Comparison table, set customer_total_tax to the sum of tax_amount for all CustomerTaxRow entries with transaction_id == customer_transaction_id.
        """
        updated = 0
        for comp in Comparison.objects.all():
            total_tax = CustomerTaxRow.objects.filter(transaction_id=comp.customer_transaction_id).aggregate(total=Sum('tax_amount'))['total'] or 0
            if comp.customer_total_tax != total_tax:
                comp.customer_total_tax = total_tax
                comp.save(update_fields=['customer_total_tax'])
                updated += 1
        return Response({'message': f'Updated {updated} comparison rows with customer_total_tax.'})
    
    def post(self, request):
        """
        API endpoint to compare customer and bank taxes for matched transactions and populate the Comparison table.
        """
        matched_customers = RecoCustomerTransaction.objects.filter(matched_bank_transaction_id__isnull=False)
        results = []
        count = 0
        
        logger.info(f"BTTaxComparisonView.post: Processing {matched_customers.count()} matched customer transactions")
        
        for cust_tx in matched_customers:
            if not isinstance(cust_tx, RecoCustomerTransaction):
                continue
            bank_tx = cust_tx.matched_bank_transaction
            if not bank_tx:
                logger.warning(f"BTTaxComparisonView: Customer transaction {cust_tx.id} has matched_bank_transaction_id={cust_tx.matched_bank_transaction_id} but bank_tx is None")
                continue
            
            logger.debug(f"BTTaxComparisonView: Processing customer_tx={cust_tx.id}, matched_bank_tx={bank_tx.id}, bank_tx.type={bank_tx.type}, bank_tx.internal_number={bank_tx.internal_number}")
            
            # --- original logic ---
            cust_tax_rows = CustomerTaxRow.objects.filter(transaction=cust_tx)
            for row in cust_tax_rows:
                tax_type = row.tax_type.strip().upper()
                # Use total_tax_amount (aggregated by document_number) for comparison with bank tax
                customer_tax_for_comparison = row.total_tax_amount if row.total_tax_amount is not None else row.tax_amount
                customer_tax_for_comparison = customer_tax_for_comparison.quantize(Decimal('0.001')) if isinstance(customer_tax_for_comparison, Decimal) else round(float(customer_tax_for_comparison), 3)
                
                # Individual tax amount for this specific transaction row
                customer_tax_individual = row.tax_amount.quantize(Decimal('0.001')) if isinstance(row.tax_amount, Decimal) else round(float(row.tax_amount), 3)
                
                # Find matching bank tax
                bank_tax = None
                if bank_tx.internal_number:
                    logger.debug(f"BTTaxComparisonView: Looking for bank tax - customer_tx={cust_tx.id}, tax_type={tax_type}, internal_number={bank_tx.internal_number}")
                    
                    # Check all transactions with this internal_number and type
                    bank_tx_candidates = RecoBankTransaction.objects.filter(
                        internal_number=bank_tx.internal_number,
                        type__iexact=tax_type
                    )
                    logger.debug(f"BTTaxComparisonView: Found {bank_tx_candidates.count()} bank transactions with internal_number={bank_tx.internal_number}, type={tax_type}")
                    
                    if bank_tx_candidates.count() > 0:
                        for candidate in bank_tx_candidates:
                            logger.debug(f"BTTaxComparisonView: Candidate bank_tx: id={candidate.id}, type={candidate.type}, amount={candidate.amount}, internal_number={candidate.internal_number}")
                    
                    bank_tx_row = bank_tx_candidates.order_by('-id').first()
                    if bank_tx_row:
                        val = bank_tx_row.amount
                        bank_tax = val.quantize(Decimal('0.001')) if isinstance(val, Decimal) else round(float(val), 3)
                        logger.info(f"BTTaxComparisonView: ✅ Found bank_tax for customer_tx={cust_tx.id}, tax_type={tax_type}: bank_tx_id={bank_tx_row.id}, amount={bank_tax}")
                    else:
                        # Additional investigation: check if agios transactions exist at all with this internal_number
                        if tax_type == 'AGIOS':
                            all_agios_with_internal = RecoBankTransaction.objects.filter(
                                internal_number=bank_tx.internal_number,
                                type__iexact='agios'
                            )
                            all_agios_any_internal = RecoBankTransaction.objects.filter(type__iexact='agios')
                            logger.warning(f"BTTaxComparisonView: ❌ No bank_tax found for customer_tx={cust_tx.id}, tax_type={tax_type}, internal_number={bank_tx.internal_number}")
                            logger.warning(f"  - Agios transactions with this internal_number: {all_agios_with_internal.count()}")
                            logger.warning(f"  - Total agios transactions in DB: {all_agios_any_internal.count()}")
                            if all_agios_any_internal.count() > 0:
                                sample_agios = all_agios_any_internal[:3]
                                for ag in sample_agios:
                                    logger.warning(f"    Sample agios: id={ag.id}, amount={ag.amount}, internal_number={ag.internal_number}, date_ref={ag.date_ref}")
                        else:
                            logger.warning(f"BTTaxComparisonView: ❌ No bank_tax found for customer_tx={cust_tx.id}, tax_type={tax_type}, internal_number={bank_tx.internal_number}")
                            bank_tax = Decimal('0.000')
                else:
                    logger.warning(f"BTTaxComparisonView: ❌ bank_tx.internal_number is None for customer_tx={cust_tx.id}, bank_tx_id={bank_tx.id}, tax_type={tax_type}")
                    bank_tax = Decimal('0.000')
                
                # Ensure bank_tax is not None
                if bank_tax is None:
                    logger.warning(f"BTTaxComparisonView: bank_tax is None for customer_tx={cust_tx.id}, tax_type={tax_type}, setting to 0.000")
                    bank_tax = Decimal('0.000')
                
                # === INVESTIGATION: Detailed logging for mismatches ===
                # Calculate comparison values first to determine if we should investigate
                abs_cust_val = abs(Decimal(str(customer_tax_for_comparison))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                abs_bank_val = abs(Decimal(str(bank_tax or 0))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                
                # Determine status to see if there's a mismatch
                is_mismatch = not (abs_cust_val == abs_bank_val and abs_cust_val != Decimal('0.000')) and not (abs_cust_val == Decimal('0.000') and abs_bank_val == Decimal('0.000')) and not (abs_cust_val == Decimal('0.000') or abs_bank_val == Decimal('0.000'))
                
                # Log investigation for AGIOS/AGIOS ESCOMPTE or when there's a mismatch
                should_investigate = (tax_type == 'AGIOS ESCOMPTE' or tax_type == 'AGIOS') or is_mismatch
                
                if should_investigate:
                    # Get all customer transactions with same document_number
                    related_cust_txs = RecoCustomerTransaction.objects.filter(
                        document_number=cust_tx.document_number
                    ) if cust_tx.document_number else RecoCustomerTransaction.objects.none()
                    
                    # Get all customer tax rows for this document_number and tax_type
                    all_cust_tax_rows = CustomerTaxRow.objects.filter(
                        transaction__document_number=cust_tx.document_number,
                        tax_type__iexact=tax_type
                    ) if cust_tx.document_number else CustomerTaxRow.objects.none()
                    
                    # Calculate actual sum of individual tax_amounts
                    actual_sum = all_cust_tax_rows.aggregate(total=Sum('tax_amount'))['total'] or 0
                    
                    # Get all bank transactions with same internal_number and type
                    all_bank_txs = RecoBankTransaction.objects.filter(
                        internal_number=bank_tx.internal_number,
                        type__iexact=tax_type
                    ) if bank_tx.internal_number else RecoBankTransaction.objects.none()
                    
                    # Sum all bank transaction amounts
                    bank_tax_sum = all_bank_txs.aggregate(total=Sum('amount'))['total'] or 0
                    bank_tax_sum_abs = abs(Decimal(str(bank_tax_sum)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)) if bank_tax_sum else Decimal('0.000')
                    
                    print(f"\n{'='*80}")
                    print(f"🔍 INVESTIGATION - customer_tx={cust_tx.id}, tax_type={tax_type}")
                    print(f"  - document_number: {cust_tx.document_number}")
                    print(f"  - matched_bank_tx_id: {bank_tx.id}, internal_number: {bank_tx.internal_number}")
                    print(f"  - Related customer transactions: {related_cust_txs.count()} with same document_number")
                    print(f"  - Customer tax rows: {all_cust_tax_rows.count()} rows for document_number={cust_tx.document_number}, tax_type={tax_type}")
                    print(f"  - Individual tax_amount sum: {actual_sum}")
                    print(f"  - total_tax_amount (from row): {row.total_tax_amount}")
                    print(f"  - customer_tax_for_comparison: {customer_tax_for_comparison}")
                    print(f"  - Bank transactions: {all_bank_txs.count()} with internal_number={bank_tx.internal_number}, type={tax_type}")
                    print(f"  - Bank transaction amounts: {[str(tx.amount) for tx in all_bank_txs[:5]]}")
                    print(f"  - Bank tax sum (all transactions): {bank_tax_sum_abs}")
                    print(f"  - Bank tax (single transaction): {abs_bank_val}")
                    print(f"  - Difference (total_tax_amount vs bank_sum): {abs(abs_cust_val - bank_tax_sum_abs):.3f}")
                    print(f"  - Difference (total_tax_amount vs bank_single): {abs(abs_cust_val - abs_bank_val):.3f}")
                    print(f"{'='*80}\n")
                    
                    logger.info(f"🔍 INVESTIGATION - customer_tx={cust_tx.id}, tax_type={tax_type}")
                    logger.info(f"  - document_number: {cust_tx.document_number}")
                    logger.info(f"  - matched_bank_tx_id: {bank_tx.id}, internal_number: {bank_tx.internal_number}")
                    logger.info(f"  - Related customer transactions: {related_cust_txs.count()} with same document_number")
                    logger.info(f"  - Customer tax rows: {all_cust_tax_rows.count()} rows for document_number={cust_tx.document_number}, tax_type={tax_type}")
                    logger.info(f"  - Individual tax_amount sum: {actual_sum}")
                    logger.info(f"  - total_tax_amount (from row): {row.total_tax_amount}")
                    logger.info(f"  - customer_tax_for_comparison: {customer_tax_for_comparison}")
                    logger.info(f"  - Bank transactions: {all_bank_txs.count()} with internal_number={bank_tx.internal_number}, type={tax_type}")
                    logger.info(f"  - Bank transaction amounts: {[str(tx.amount) for tx in all_bank_txs[:5]]}")
                    logger.info(f"  - Bank tax sum (all transactions): {bank_tax_sum_abs}")
                    logger.info(f"  - Bank tax (single transaction): {abs_bank_val}")
                    logger.info(f"  - Difference (total_tax_amount vs bank_sum): {abs(abs_cust_val - bank_tax_sum_abs):.3f}")
                    logger.info(f"  - Difference (total_tax_amount vs bank_single): {abs(abs_cust_val - abs_bank_val):.3f}")
                
                # Use pre-calculated values for comparison
                if abs_cust_val == abs_bank_val and abs_cust_val != Decimal('0.000'):
                    status_str = 'matched'
                elif abs_cust_val == Decimal('0.000') and abs_bank_val == Decimal('0.000'):
                    continue
                elif abs_cust_val == Decimal('0.000') or abs_bank_val == Decimal('0.000'):
                    status_str = 'missing'
                    if tax_type == 'AGIOS' and abs_bank_val == Decimal('0.000'):
                        logger.warning(f"BTTaxComparisonView: ⚠️ MISSING bank_tax for AGIOS - customer_tx={cust_tx.id}, customer_tax={abs_cust_val}, bank_tax={abs_bank_val}, matched_bank_tx_id={bank_tx.id}, internal_number={bank_tx.internal_number}")
                else:
                    status_str = 'mismatch'
                
                # Store individual tax_amount in customer_tax, and total_tax_amount in customer_total_tax
                abs_cust_individual = abs(Decimal(str(customer_tax_individual))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                obj, created = Comparison.objects.update_or_create(
                    customer_transaction=cust_tx,
                    tax_type=tax_type,
                    defaults={
                        'customer_tax': abs_cust_individual,  # Individual tax_amount for this transaction
                        'bank_tax': abs_bank_val,
                        'status': status_str,
                        'customer_total_tax': row.total_tax_amount,  # Aggregated total_tax_amount
                        'matched_bank_transaction_id': getattr(bank_tx, 'id', None),
                    }
                )
                
                if tax_type == 'AGIOS' and abs_bank_val == Decimal('0.000') and abs_cust_val != Decimal('0.000'):
                    logger.warning(f"BTTaxComparisonView: Created/Updated Comparison with missing bank_tax: id={obj.id}, customer_tx={cust_tx.id}, tax_type={tax_type}, customer_tax={abs_cust_val}, bank_tax={abs_bank_val}, matched_bank_tx_id={bank_tx.id}")
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
        """
        count = Comparison.objects.count()
        table_name = Comparison._meta.db_table
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
        except Exception as e:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
        
        return Response({
            'message': f'Emptied Comparison table. Deleted {count} rows.',
            'deleted_count': count,
            'method': 'TRUNCATE' if count > 0 else 'N/A'
        }, status=status.HTTP_200_OK)


class AttijariAssignInternalNumberView(APIView):
    """
    POST endpoint to assign the same internal_number to multiple bank transactions.
    Accepts a list of bank transaction IDs and an internal_number.
    Works with both origine (type='origine') and non-origine transactions.
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            bank_transaction_ids = request.data.get('bank_transaction_ids', [])
            internal_number = request.data.get('internal_number')
            
            # Validate input
            if not bank_transaction_ids:
                return Response(
                    {"error": "bank_transaction_ids is required and must be a non-empty list"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not isinstance(bank_transaction_ids, list):
                return Response(
                    {"error": "bank_transaction_ids must be a list"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not internal_number:
                return Response(
                    {"error": "internal_number is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not isinstance(internal_number, str) or not internal_number.strip():
                return Response(
                    {"error": "internal_number must be a non-empty string"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Clean the internal_number
            internal_number = internal_number.strip()
            
            # Get all requested transactions
            transactions = RecoBankTransaction.objects.filter(id__in=bank_transaction_ids)
            found_ids = list(transactions.values_list('id', flat=True))
            
            # Check if all IDs exist
            missing_ids = set(bank_transaction_ids) - set(found_ids)
            if missing_ids:
                return Response(
                    {
                        "error": f"Some transaction IDs not found: {list(missing_ids)}",
                        "found_ids": found_ids,
                        "missing_ids": list(missing_ids)
                    },
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Count origine and non-origine transactions before update
            origine_count = transactions.filter(type='origine').count()
            non_origine_count = transactions.exclude(type='origine').count()
            
            # Update all transactions with the same internal_number
            updated_count = transactions.update(internal_number=internal_number)
            
            # Log the update
            logger.info(
                f"AttijariAssignInternalNumberView: Assigned internal_number '{internal_number}' to {updated_count} transactions. "
                f"Origine: {origine_count}, Non-origine: {non_origine_count}"
            )
            
            # Refetch updated transactions to get latest data
            updated_transactions_list = list(RecoBankTransaction.objects.filter(id__in=found_ids).values(
                'id', 'type', 'label', 'amount', 'internal_number'
            ))
            
            # Prepare response with details
            updated_transactions = []
            for tx_data in updated_transactions_list:
                updated_transactions.append({
                    'id': tx_data['id'],
                    'type': tx_data['type'],
                    'label': tx_data['label'],
                    'amount': float(tx_data['amount']),
                    'internal_number': tx_data['internal_number'],
                    'is_origine': tx_data['type'] == 'origine'
                })
            
            return Response({
                'message': f"Successfully assigned internal_number '{internal_number}' to {updated_count} transactions",
                'updated_count': updated_count,
                'internal_number': internal_number,
                'updated_transaction_ids': found_ids,
                'origine_count': origine_count,
                'non_origine_count': non_origine_count,
                'transactions': updated_transactions
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"AttijariAssignInternalNumberView error: {str(e)}\n{error_details}")
            return Response({
                'error': str(e),
                'details': error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AttijariTaxManagementView(APIView):
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
            
            if not isinstance(description, list):
                description = [description]
            
            try:
                tax = Tax.objects.get(name__iexact=name)
                
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
            
            if not isinstance(description, list):
                description = [description]
            
            try:
                tax = Tax.objects.get(name__iexact=name)
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


class AttijariMatchTaxView(APIView):
    """
    View to match bank transactions with tax descriptions and update them with dynamic tax amounts.
    Filters by bank and company to ensure correct matching.
    """
    
    def post(self, request):
        try:
            print("=" * 80)
            print("AttijariMatchTaxView: Starting tax matching")
            print("=" * 80)
            logger.info("=" * 80)
            logger.info("AttijariMatchTaxView: Starting tax matching")
            logger.info("=" * 80)
            
            # Get bank_code and company_code from request body
            bank_code = request.data.get('bank_code')
            company_code = request.data.get('company_code')
            
            print(f"Request data received:")
            print(f"  - bank_code: {bank_code}")
            print(f"  - company_code: {company_code}")
            logger.info(f"Request data - bank_code: {bank_code}, company_code: {company_code}")
            
            # Validate required parameters
            if not bank_code:
                error_msg = "bank_code is required in request body"
                print(f"❌ ERROR: {error_msg}")
                logger.error(error_msg)
                return Response({
                    "error": error_msg
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if not company_code:
                error_msg = "company_code is required in request body"
                print(f"❌ ERROR: {error_msg}")
                logger.error(error_msg)
                return Response({
                    "error": error_msg
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get Bank and Company objects
            try:
                bank = Bank.objects.get(code=bank_code)
                print(f"✓ Found Bank: {bank.name} (code: {bank.code})")
                logger.info(f"Found Bank: {bank.name} (code: {bank.code})")
            except Bank.DoesNotExist:
                error_msg = f"Bank with code '{bank_code}' not found"
                print(f"❌ ERROR: {error_msg}")
                logger.error(error_msg)
                return Response({
                    "error": error_msg
                }, status=status.HTTP_404_NOT_FOUND)
            
            try:
                company = Company.objects.get(code=company_code)
                print(f"✓ Found Company: {company.name} (code: {company.code})")
                logger.info(f"Found Company: {company.name} (code: {company.code})")
            except Company.DoesNotExist:
                error_msg = f"Company with code '{company_code}' not found"
                print(f"❌ ERROR: {error_msg}")
                logger.error(error_msg)
                return Response({
                    "error": error_msg
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Filter taxes by bank and company
            taxes = Tax.objects.filter(bank=bank, company=company)
            taxes_count = taxes.count()
            print(f"\nTaxes filtered by bank='{bank.name}' and company='{company.name}': {taxes_count} taxes found")
            logger.info(f"Taxes filtered - bank={bank.name}, company={company.name}, count={taxes_count}")
            
            if taxes_count == 0:
                # Check if taxes exist for this bank or company separately
                taxes_for_bank = Tax.objects.filter(bank=bank).count()
                taxes_for_company = Tax.objects.filter(company=company).count()
                print(f"⚠️  No taxes found for this bank+company combination")
                print(f"   - Taxes for bank '{bank.name}': {taxes_for_bank}")
                print(f"   - Taxes for company '{company.name}': {taxes_for_company}")
                logger.warning(f"No taxes found - bank has {taxes_for_bank} taxes, company has {taxes_for_company} taxes")
                return Response({
                    "message": f"No taxes found for bank '{bank.name}' and company '{company.name}'",
                    "matched_count": 0,
                    "total_transactions": 0,
                    "debug": {
                        "taxes_for_bank": taxes_for_bank,
                        "taxes_for_company": taxes_for_company
                    }
                }, status=status.HTTP_200_OK)
            
            # Sort taxes by longest description length (longest first) to avoid short tax matches inside longer tax descriptions
            # This ensures that "COM / AUT DB PASSAGERE COMMISSION CGESPA EPS" is tested before "COM"
            def get_max_description_length(tax):
                if tax.description and isinstance(tax.description, list):
                    return max([len(desc) for desc in tax.description], default=0)
                return 0
            
            taxes = sorted(taxes, key=get_max_description_length, reverse=True)
            
            # Log tax details
            print(f"\nTaxes to match against (sorted by longest description first):")
            for idx, tax in enumerate(taxes, 1):
                desc_count = len(tax.description) if tax.description else 0
                max_desc_len = get_max_description_length(tax)
                print(f"  [{idx}] {tax.name} - {desc_count} description(s) (max length: {max_desc_len})")
                logger.debug(f"Tax {idx}: {tax.name} with {desc_count} descriptions (max length: {max_desc_len})")
            
            # Filter bank transactions by bank only
            # Company filtering is handled via taxes (taxes are already filtered by company)
            # This simplifies the query and avoids complex relationship paths
            try:
                print(f"\nFiltering bank transactions...")
                print(f"  - Bank: {bank.name} (code: {bank.code})")
                print(f"  - Company: {company.name} (code: {company.code}) - used for tax filtering only")
                logger.info(f"Filtering transactions - bank={bank.name}, company={company.name} (for tax filtering)")
                
                # Filter transactions by bank only
                # Company filtering is already done on taxes, so matching will only use company-specific taxes
                bank_transactions = RecoBankTransaction.objects.filter(bank=bank)
                transactions_count = bank_transactions.count()
                print(f"✓ Bank transactions filtered by bank: {transactions_count} transactions found")
                logger.info(f"Transactions filtered by bank - count={transactions_count}")
                
                if transactions_count == 0:
                    print(f"  ⚠️  No bank transactions found for bank '{bank.name}'")
                    logger.warning(f"No bank transactions found for bank={bank.name}")
            except Exception as e:
                error_msg = f"Error filtering transactions: {str(e)}"
                print(f"❌ ERROR: {error_msg}")
                logger.error(error_msg, exc_info=True)
                import traceback
                traceback.print_exc()
                return Response({
                    "error": error_msg,
                    "details": str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            if transactions_count == 0:
                return Response({
                    "message": f"No bank transactions found for bank '{bank.name}'",
                    "matched_count": 0,
                    "total_transactions": 0
                }, status=status.HTTP_200_OK)
            
            print(f"\n{'=' * 80}")
            print("Starting tax matching process...")
            print(f"{'=' * 80}")
            logger.info("Starting tax matching process")
            
            matched_count = 0
            skipped_count = 0
            no_label_count = 0
            error_count = 0
            
            try:
                for idx, transaction in enumerate(bank_transactions, 1):
                    try:
                        label = transaction.label.lower() if transaction.label else ""
                        # Normalize whitespace: replace multiple spaces with single space for consistent matching
                        label = re.sub(r'\s+', ' ', label).strip()
                        
                        if not transaction.label:
                            no_label_count += 1
                            if idx <= 5:  # Log first 5 for debugging
                                logger.debug(f"Transaction {idx} (ID={transaction.id}): No label, skipping")
                            continue
                        
                        # Debug: Show what we're scanning
                        # Show detailed scan for first 10 transactions or any transaction containing "eff", "comm", or "imp"
                        should_log_detail = idx <= 10 or 'eff' in label.lower() or 'comm' in label.lower() or 'imp' in label.lower()
                        if should_log_detail:
                            print(f"\n  [{idx}] 🔍 SCANNING Transaction ID={transaction.id}")
                            print(f"      Original Label: '{transaction.label}'")
                            print(f"      Label (lowercase, normalized): '{label}'")
                            print(f"      Label length: {len(label)}")
                            print(f"      Testing against {taxes_count} tax(es)...")
                            logger.info(f"Transaction {idx} (ID={transaction.id}): Original='{transaction.label}', Normalized='{label}'")
                        
                        matched_tax = None
                        matched_description = None
                        matched_pattern = None
                        best_match_length = 0  # Track the length of the best match to ensure we get the longest
                        
                        for tax in taxes:
                            try:
                                if tax.description:
                                    # Sort descriptions by length (longest first) to avoid short matches inside longer terms
                                    sorted_descriptions = sorted(tax.description, key=len, reverse=True)
                                    for desc in sorted_descriptions:
                                        try:
                                            # Skip if this description is shorter than a match we already found
                                            # This ensures we always get the longest possible match
                                            if len(desc) < best_match_length:
                                                if idx <= 10:
                                                    print(f"      → Skipping Tax: '{tax.name}' | Description: '{desc}' (length {len(desc)} < best match {best_match_length})")
                                                continue
                                            
                                            # Match even if tax keyword is embedded in other text or attached to digits/symbols
                                            # For descriptions with spaces and special chars like "/", we need to match the full string
                                            desc_original = desc
                                            desc_lower = desc.lower()
                                            # Normalize whitespace: replace multiple spaces with single space for consistent matching
                                            desc_lower = re.sub(r'\s+', ' ', desc_lower).strip()
                                            # Escape the description for regex, but preserve word boundaries for matching
                                            # Use word boundaries, but also allow matching at start/end of string or with non-word chars
                                            escaped_desc = re.escape(desc_lower)
                                            # Pattern: More flexible matching - allow matching anywhere in the string
                                            # Since we've normalized whitespace, we can use a simpler pattern
                                            # Try exact match first, then fallback to word boundaries if needed
                                            pattern = escaped_desc  # Simple substring match (most flexible)
                                            
                                            # Enhanced logging for debugging
                                            # Log for first 10 transactions or any transaction/tax containing "eff", "comm", or "imp"
                                            should_log_tax = idx <= 10 or 'eff' in label.lower() or 'comm' in label.lower() or 'imp' in label.lower() or 'eff' in desc_lower or 'comm' in desc_lower or 'imp' in desc_lower
                                            if should_log_tax:
                                                print(f"      → Testing Tax: '{tax.name}'")
                                                print(f"         Original Description: '{desc_original}'")
                                                print(f"         Normalized Description: '{desc_lower}'")
                                                print(f"         Escaped Description: '{escaped_desc}'")
                                                print(f"         Label (normalized): '{label}'")
                                                print(f"         Pattern: {pattern}")
                                                print(f"         Description length: {len(desc_lower)}, Label length: {len(label)}")
                                                # Check if normalized strings match exactly
                                                exact_match = desc_lower in label
                                                print(f"         Substring check (desc_lower in label): {exact_match}")
                                                if exact_match:
                                                    pos = label.find(desc_lower)
                                                    print(f"         Substring found at position: {pos}")
                                                else:
                                                    # Show why substring doesn't match
                                                    print(f"         Substring NOT found - checking character by character...")
                                                    # Find common prefix
                                                    min_len = min(len(desc_lower), len(label))
                                                    common_prefix = 0
                                                    for i in range(min_len):
                                                        if desc_lower[i] == label[i]:
                                                            common_prefix = i + 1
                                                        else:
                                                            break
                                                    print(f"         Common prefix length: {common_prefix}")
                                                    if common_prefix < min_len:
                                                        print(f"         First difference at position {common_prefix}: desc='{desc_lower[common_prefix:common_prefix+20] if common_prefix < len(desc_lower) else ''}', label='{label[common_prefix:common_prefix+20] if common_prefix < len(label) else ''}'")
                                            
                                            match_result = re.search(pattern, label)
                                            
                                            # Enhanced logging for match result
                                            should_log_result = idx <= 10 or 'eff' in label.lower() or 'comm' in label.lower() or 'imp' in label.lower() or 'eff' in desc_lower or 'comm' in desc_lower or 'imp' in desc_lower
                                            if should_log_result:
                                                if match_result:
                                                    print(f"         ✓ Regex MATCH found! Span: {match_result.span()}, Matched: '{match_result.group()}'")
                                                else:
                                                    print(f"         ✗ Regex NO MATCH")
                                                    # Try simpler pattern for debugging
                                                    simple_pattern = escaped_desc
                                                    simple_match = re.search(simple_pattern, label)
                                                    print(f"         Simple pattern test (no word boundaries): {simple_pattern}")
                                                    print(f"         Simple pattern match: {simple_match is not None}")
                                                    if simple_match:
                                                        print(f"         Simple match found: '{simple_match.group()}' at {simple_match.span()}")
                                                    else:
                                                        # Try even simpler - just the description without escaping
                                                        unescaped_match = desc_lower in label
                                                        print(f"         Direct substring test (desc_lower in label): {unescaped_match}")
                                                        if not unescaped_match:
                                                            # Show character-by-character comparison
                                                            print(f"         Character comparison:")
                                                            print(f"           desc_lower repr: {repr(desc_lower)}")
                                                            print(f"           label repr: {repr(label)}")
                                                            # Check if it's a case issue
                                                            if desc_lower.lower() == label.lower():
                                                                print(f"         ⚠️ Case-insensitive match found but case-sensitive failed")
                                                            # Check if it's a whitespace issue
                                                            desc_no_spaces = re.sub(r'\s+', '', desc_lower)
                                                            label_no_spaces = re.sub(r'\s+', '', label)
                                                            if desc_no_spaces in label_no_spaces:
                                                                print(f"         ⚠️ Match found when ignoring all whitespace!")
                                                                print(f"           desc_no_spaces: {repr(desc_no_spaces)}")
                                                                print(f"           label_no_spaces: {repr(label_no_spaces)}")
                                            
                                            if match_result:
                                                # Always update if this match is longer than or equal to the current best match
                                                # Since we're checking all taxes sorted by length, this ensures we get the longest match
                                                if len(desc) >= best_match_length:
                                                    matched_tax = tax
                                                    matched_description = desc
                                                    matched_pattern = pattern
                                                    matched_span = match_result.span()
                                                    matched_text = label[matched_span[0]:matched_span[1]]
                                                    best_match_length = len(desc)
                                                    
                                                    if idx <= 10 or 'comm.eff.imp' in label.lower() or 'comm.eff.imp' in desc_lower:
                                                        print(f"      ✓ MATCH FOUND!")
                                                        print(f"         Tax: '{tax.name}'")
                                                        print(f"         Original Description: '{desc_original}'")
                                                        print(f"         Normalized Description: '{desc_lower}'")
                                                        print(f"         Matched text in label: '{matched_text}'")
                                                        print(f"         Position: {matched_span[0]}-{matched_span[1]}")
                                                        print(f"         Description length: {len(desc)}, Best match length: {best_match_length}")
                                                        logger.info(f"Match found for transaction {transaction.id}: Tax='{tax.name}', Desc='{desc_lower}', Matched='{matched_text}'")
                                                    
                                                    # Continue checking - there might be an even longer match in other taxes
                                                elif idx <= 10:
                                                    print(f"      → Match found but shorter (length {len(desc)} < best {best_match_length}), skipping")
                                            elif idx <= 10:
                                                print(f"      ✗ No match")
                                        except Exception as e:
                                            logger.error(f"Error matching description '{desc}' for tax '{tax.name}' on transaction {transaction.id}: {str(e)}")
                                            if idx <= 10:
                                                print(f"      ❌ Error: {str(e)}")
                                            continue
                                    # Early exit optimization: if this tax's max description length is less than our best match,
                                    # and we have a match, we can break since remaining taxes have even shorter max lengths
                                    if matched_tax and best_match_length > 0:
                                        max_desc_len = max([len(d) for d in tax.description], default=0) if tax.description else 0
                                        # Since taxes are sorted by max description length descending, if current tax's max < best_match_length,
                                        # all remaining taxes will have max <= current tax's max < best_match_length
                                        # So we already have the longest possible match
                                        if max_desc_len < best_match_length:
                                            if idx <= 10:
                                                print(f"      → Early exit: tax max length {max_desc_len} < best match {best_match_length}")
                                            break
                            except Exception as e:
                                logger.error(f"Error processing tax '{tax.name}' for transaction {transaction.id}: {str(e)}")
                                if idx <= 10:
                                    print(f"      ❌ Error processing tax '{tax.name}': {str(e)}")
                                continue
                        
                        # After checking all taxes, we have the longest match (if any)
                        
                        if matched_tax:
                            try:
                                old_type = transaction.type
                                transaction.type = matched_tax.name.lower()
                                transaction.save()
                                matched_count += 1
                                if idx <= 10 or 'comm.eff.imp' in label.lower():  # Log first 10 matches or comm.eff.imp
                                    print(f"  [{idx}] ✅ FINAL MATCH: Transaction ID={transaction.id}")
                                    print(f"      Original Label: '{transaction.label}'")
                                    print(f"      Normalized Label: '{label}'")
                                    print(f"      Tax: '{matched_tax.name}'")
                                    print(f"      Original Description: '{matched_description}'")
                                    normalized_matched_desc = re.sub(r'\s+', ' ', matched_description.lower()).strip()
                                    print(f"      Normalized Description: '{normalized_matched_desc}'")
                                    print(f"      Pattern Used: {matched_pattern}")
                                    print(f"      Type: {old_type} → {transaction.type}")
                                    print(f"      {'─' * 70}")
                                logger.info(f"Matched transaction {transaction.id}: Original='{transaction.label}', Normalized='{label}' → tax '{matched_tax.name}' (matched on '{matched_description}')")
                            except Exception as e:
                                error_count += 1
                                logger.error(f"Error saving transaction {transaction.id} after match: {str(e)}")
                                print(f"  [{idx}] ❌ Error saving: Transaction ID={transaction.id} | Error: {str(e)}")
                        else:
                            skipped_count += 1
                            if idx <= 10 or 'comm.eff.imp' in label.lower():  # Log first 10 or comm.eff.imp transactions
                                print(f"  [{idx}] ✗ NO MATCH: Transaction ID={transaction.id}")
                                print(f"      Original Label: '{transaction.label}'")
                                print(f"      Normalized Label: '{label}'")
                                print(f"      Result: No matching tax description found after scanning all {taxes_count} tax(es)")
                                print(f"      Best match length found: {best_match_length}")
                                print(f"      {'─' * 70}")
                                logger.warning(f"Skipped transaction {transaction.id}: Original='{transaction.label}', Normalized='{label}' - no matching tax (best_match_length={best_match_length})")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing transaction {idx} (ID={transaction.id if hasattr(transaction, 'id') else 'unknown'}): {str(e)}", exc_info=True)
                        print(f"  [{idx}] ❌ Error processing transaction: {str(e)}")
                        continue
            except Exception as e:
                error_msg = f"Error in matching loop: {str(e)}"
                print(f"❌ CRITICAL ERROR: {error_msg}")
                logger.error(error_msg, exc_info=True)
                import traceback
                traceback.print_exc()
                return Response({
                    "error": error_msg,
                    "details": str(e),
                    "matched_count": matched_count,
                    "error_count": error_count
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            print(f"\n{'=' * 80}")
            print("AttijariMatchTaxView: Final Statistics")
            print(f"{'=' * 80}")
            print(f"Total transactions processed: {transactions_count}")
            print(f"Matched: {matched_count}")
            print(f"Skipped (no match): {skipped_count}")
            print(f"No label: {no_label_count}")
            print(f"Errors: {error_count}")
            print(f"Taxes available: {taxes_count}")
            print(f"Match rate: {100*matched_count/transactions_count if transactions_count > 0 else 0:.1f}%")
            print(f"{'=' * 80}")
            
            logger.info("=" * 80)
            logger.info("AttijariMatchTaxView: Final Statistics")
            logger.info(f"Total transactions: {transactions_count}, Matched: {matched_count}, Skipped: {skipped_count}, No label: {no_label_count}, Errors: {error_count}")
            logger.info(f"Taxes available: {taxes_count}, Match rate: {100*matched_count/transactions_count if transactions_count > 0 else 0:.1f}%")
            logger.info("=" * 80)
            
            return Response({
                "message": f"Successfully matched {matched_count} transactions with taxes",
                "matched_count": matched_count,
                "skipped_count": skipped_count,
                "no_label_count": no_label_count,
                "error_count": error_count,
                "total_transactions": transactions_count,
                "bank": bank.name,
                "company": company.name,
                "taxes_count": taxes_count,
                "match_rate_percent": round(100*matched_count/transactions_count if transactions_count > 0 else 0, 1)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"An error occurred: {str(e)}"
            print(f"\n{'=' * 80}")
            print(f"❌ CRITICAL ERROR in AttijariMatchTaxView")
            print(f"{'=' * 80}")
            print(f"Error: {error_msg}")
            print(f"\nTraceback:\n{error_details}")
            print(f"{'=' * 80}")
            logger.error("=" * 80)
            logger.error("AttijariMatchTaxView: CRITICAL ERROR")
            logger.error(error_msg)
            logger.error(f"Traceback:\n{error_details}")
            logger.error("=" * 80)
            return Response(
                {
                    "error": error_msg,
                    "details": error_details
                },
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


class AttijariSortedRecoBankTransactionsView(APIView):
    """
    View to return RecoBankTransaction records sorted with custom grouping logic:
    - Primary sort: operation_date (ascending)
    - When an "origine" is encountered, its relatives (same internal_number) are grouped directly under it
    - Relatives within each group are sorted by operation_date ascending
    - Transactions without grouping keys maintain their date-sorted position
    """
    
    def get(self, request):
        try:
            bank_code = request.query_params.get('bank_code')
            
            if not bank_code:
                return Response({
                    "error": "bank_code is required as query parameter"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get bank
            try:
                bank = Bank.objects.get(code=bank_code)
            except Bank.DoesNotExist:
                return Response({
                    "error": f"Bank with code '{bank_code}' not found"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Step 1: Fetch all transactions ordered by operation_date
            transactions = RecoBankTransaction.objects.filter(bank=bank).order_by('operation_date', 'id')
            transactions_list = list(transactions)
            
            if not transactions_list:
                return Response({
                    "transactions": [],
                    "total": 0,
                    "grouped_count": 0,
                    "ungrouped_count": 0
                }, status=status.HTTP_200_OK)
            
            # Step 2: Build groups by internal_number
            # Dictionary: {internal_number: {'origine': origine_tx, 'relatives': [relatives...]}}
            groups = {}
            origine_by_internal = {}  # Quick lookup: {internal_number: origine_transaction}
            
            for tx in transactions_list:
                if tx.internal_number:
                    if tx.type == 'origine':
                        if tx.internal_number not in groups:
                            groups[tx.internal_number] = {'origine': tx, 'relatives': []}
                        else:
                            # If groupe exists but no origine yet, set it
                            if 'origine' not in groups[tx.internal_number] or groups[tx.internal_number]['origine'] is None:
                                groups[tx.internal_number]['origine'] = tx
                        origine_by_internal[tx.internal_number] = tx
                    else:
                        # It's a relative
                        if tx.internal_number not in groups:
                            groups[tx.internal_number] = {'origine': None, 'relatives': []}
                        groups[tx.internal_number]['relatives'].append(tx)
            
            # Sort relatives within each group by operation_date
            for internal_num, group_data in groups.items():
                if group_data['relatives']:
                    group_data['relatives'].sort(key=lambda x: (x.operation_date, x.id))
            
            # Step 3: Create sorted result list (FIXED)

            sorted_transactions = []
            processed_ids = set()

            for tx in transactions_list:
                # Already handled
                if tx.id in processed_ids:
                    continue

                # Grouped transaction
                if tx.internal_number and tx.internal_number in groups:
                    group_data = groups[tx.internal_number]
                    origine_tx = group_data['origine']

                    # CASE 1: Origine → add origine + all relatives immediately after
                    if tx.type == 'origine' and origine_tx and origine_tx.id == tx.id:
                        sorted_transactions.append(origine_tx)
                        processed_ids.add(origine_tx.id)

                        for relative in group_data['relatives']:
                            if relative.id not in processed_ids:
                                sorted_transactions.append(relative)
                                processed_ids.add(relative.id)

                    # CASE 2: Relative with an origine → skip but MARK AS PROCESSED
                    elif tx.type != 'origine' and origine_tx:
                        processed_ids.add(tx.id)
                        continue

                    # CASE 3: Orphaned relative → keep its date position
                    elif not origine_tx:
                        sorted_transactions.append(tx)
                        processed_ids.add(tx.id)

                # Ungrouped transaction
                else:
                    sorted_transactions.append(tx)
                    processed_ids.add(tx.id)
            
            # Step 5: Serialize transactions
            serialized_transactions = []
            grouped_count = 0
            ungrouped_count = 0
            processed_groups = set()  # Track which groups we've already counted
            
            for tx in sorted_transactions:
                is_origine = tx.type == 'origine'
                group_id = tx.internal_number if tx.internal_number else None
                group_size = 0
                
                if group_id and group_id in groups:
                    group_data = groups[group_id]
                    if group_data['origine']:
                        # Calculate group size (origine + relatives)
                        group_size = 1 + len(group_data['relatives'])
                        # Only count the group once (when we encounter the origine)
                        if is_origine and group_id not in processed_groups:
                            grouped_count += 1
                            processed_groups.add(group_id)
                    else:
                        # Orphaned relative (no origine) - group_size is just relatives count
                        group_size = len(group_data['relatives'])
                        # Count as ungrouped only once per group
                        if group_id not in processed_groups:
                            ungrouped_count += 1
                            processed_groups.add(group_id)
                else:
                    # No grouping - count as ungrouped
                    ungrouped_count += 1
                
                serialized_transactions.append({
                    'id': tx.id,
                    'bank': tx.bank.name if tx.bank else None,
                    'operation_date': tx.operation_date.isoformat() if tx.operation_date else None,
                    'value_date': tx.value_date.isoformat() if tx.value_date else None,
                    'label': tx.label,
                    'amount': float(tx.amount),
                    'debit': tx.debit,
                    'credit': tx.credit,
                    'ref': tx.ref,
                    'date_ref': tx.date_ref.isoformat() if tx.date_ref else None,
                    'document_reference': tx.document_reference,
                    'internal_number': tx.internal_number,
                    'type': tx.type,
                    'payment_class': tx.payment_class.name if tx.payment_class else None,
                    'payment_status': tx.payment_status.name if tx.payment_status else None,
                    'accounting_account': tx.accounting_account,
                    'group_id': group_id,
                    'is_origine': is_origine,
                    'group_size': group_size
                })
            
            return Response({
                "transactions": serialized_transactions,
                "total": len(serialized_transactions),
                "grouped_count": grouped_count,
                "ungrouped_count": ungrouped_count,
                "bank": bank.name,
                "bank_code": bank.code
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in AttijariSortedRecoBankTransactionsView: {str(e)}\n{error_details}")
            return Response({
                "error": str(e),
                "details": error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AttijariUnmatchedTransactionsView(APIView):
    """
    View to generate two lists:
    1. Unmatched bank transactions (all bank transactions minus those in high matches)
    2. Unmatched customer transactions (customer transactions without matched_bank_transaction)
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
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


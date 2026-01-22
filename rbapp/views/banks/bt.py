"""
BT (Banque de Tunisie) Specific Views

This module contains all BT-specific views that handle:
- BT bank ledger preprocessing
- BT-specific matching algorithms (sensitive matching rules)
- BT beginning balance extraction
- BT customer tax extraction
- BT transaction matching
- Other BT-specific processing

All views inherit from base classes in base.py and implement BT-specific logic.

This module also contains BT-specific helper methods/functions that are used
by the views but are specific to BT's data format and business rules.
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
# BT-Specific Views
# ============================================================================

class BTPreprocessBankLedgerEntryView(APIView):
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


class BTExtractBeginningBalanceView(APIView):
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


class BTMatchCustomerBankTransactionsView(APIView):
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


class BTManualMatchCustomerBankTransactionsView(APIView):
    """
    Manually link a RecoCustomerTransaction to a RecoBankTransaction for BT.

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
                f"BTManualMatchCustomerBankTransactionsView: matched customer_tx={cust_tx_id} to bank_tx={bank_tx_id}, "
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
                f"BTManualMatchCustomerBankTransactionsView error: {str(e)}\n{error_details}"
            )
            return Response(
                {
                    "error": str(e),
                    "details": error_details,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class BTMatchTransactionView(APIView):
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


class BTMatchBankTransactionTaxesView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Get all bank transactions (RECO)
            all_transactions = RecoBankTransaction.objects.all()
            
            # Separate origine and non-origine transactions
            origine_transactions = RecoBankTransaction.objects.filter(type='origine')
            non_origine_transactions = RecoBankTransaction.objects.filter(~Q(type='origine') & ~Q(type='origine'))
            
            matched_count = 0
            unmatched_count = 0
            
            # Track tax transactions for detailed logging
            tax_transactions_matched = []
            tax_transactions_unmatched = []
            
            print("=" * 80)
            print("BTMatchBankTransactionTaxesView: Processing non-origine (tax) transactions")
            print("=" * 80)
            logger.info("=" * 80)
            logger.info("BTMatchBankTransactionTaxesView: Processing non-origine (tax) transactions")
            logger.info("=" * 80)
            
            # Process each non-origine transaction
            for idx, non_origine in enumerate(non_origine_transactions, 1):
                matched = False
                match_method = None
                matching_origine_id = None
                matching_origine_internal_number = None
                match_failure_reason = None
                
                # Log tax transaction details
                tax_info = {
                    'id': non_origine.id,
                    'type': non_origine.type,
                    'amount': float(non_origine.amount) if non_origine.amount else None,
                    'ref': non_origine.ref,
                    'date_ref': non_origine.date_ref.isoformat() if non_origine.date_ref else None,
                    'label': non_origine.label[:50] if non_origine.label else None,
                    'operation_date': non_origine.operation_date.isoformat() if non_origine.operation_date else None
                }
                
                # First, try to match by ref
                if non_origine.ref:
                    matching_origine = origine_transactions.filter(ref=non_origine.ref).first()
                    if matching_origine:
                        if matching_origine.internal_number:
                            non_origine.internal_number = matching_origine.internal_number
                        else:
                            new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                            matching_origine.internal_number = new_internal_number
                            matching_origine.save()
                            non_origine.internal_number = new_internal_number
                        
                        # Inherit payment_class and payment_status from origine transaction
                        if matching_origine.payment_class:
                            non_origine.payment_class = matching_origine.payment_class
                        if matching_origine.payment_status:
                            non_origine.payment_status = matching_origine.payment_status
                        
                        non_origine.save()
                        matched = True
                        matched_count += 1
                        match_method = 'ref'
                        matching_origine_id = matching_origine.id
                        matching_origine_internal_number = matching_origine.internal_number
                        
                        # Log successful match by ref
                        tax_info['matched'] = True
                        tax_info['match_method'] = 'ref'
                        tax_info['matching_origine_id'] = matching_origine.id
                        tax_info['matching_origine_ref'] = matching_origine.ref
                        tax_info['matching_origine_date_ref'] = matching_origine.date_ref.isoformat() if matching_origine.date_ref else None
                        tax_info['matching_origine_internal_number'] = matching_origine.internal_number
                        tax_info['matching_origine_amount'] = float(matching_origine.amount) if matching_origine.amount else None
                        tax_transactions_matched.append(tax_info)
                        
                        if idx <= 10 or non_origine.type and non_origine.type.lower() not in ['origine', 'agios']:
                            print(f"\n[{idx}] ✅ TAX MATCHED (by ref): Transaction ID={non_origine.id}")
                            print(f"     Tax Type: {non_origine.type}")
                            print(f"     Tax Amount: {non_origine.amount}")
                            print(f"     Tax Ref: '{non_origine.ref}'")
                            print(f"     Tax Date Ref: {non_origine.date_ref}")
                            print(f"     → Matched to Origine ID={matching_origine.id}")
                            print(f"        Origine Ref: '{matching_origine.ref}'")
                            print(f"        Origine Date Ref: {matching_origine.date_ref}")
                            print(f"        Internal Number: {matching_origine.internal_number}")
                            print(f"        Origine Amount: {matching_origine.amount}")
                        logger.info(f"Tax transaction {non_origine.id} (type={non_origine.type}) matched to origine {matching_origine.id} by ref='{non_origine.ref}'")
                    else:
                        match_failure_reason = f"No origine found with ref='{non_origine.ref}'"
                
                # If still not matched, generate a new internal_number
                if not matched:
                    non_origine.internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                    non_origine.save()
                    unmatched_count += 1
                    
                    # Log unmatched tax transaction
                    tax_info['matched'] = False
                    tax_info['match_method'] = None
                    tax_info['matching_origine_id'] = None
                    tax_info['matching_origine_internal_number'] = None
                    tax_info['match_failure_reason'] = match_failure_reason or (
                        "No ref" if not non_origine.ref 
                        else "No matching origine found"
                    )
                    tax_info['new_internal_number'] = non_origine.internal_number
                    tax_transactions_unmatched.append(tax_info)
                    
                    if idx <= 10 or non_origine.type and non_origine.type.lower() not in ['origine', 'agios']:
                        print(f"\n[{idx}] ❌ TAX UNMATCHED: Transaction ID={non_origine.id}")
                        print(f"     Tax Type: {non_origine.type}")
                        print(f"     Tax Amount: {non_origine.amount}")
                        print(f"     Tax Ref: '{non_origine.ref}'")
                        print(f"     Tax Date Ref: {non_origine.date_ref}")
                        print(f"     Tax Label: '{non_origine.label[:50] if non_origine.label else None}'")
                        print(f"     Reason: {tax_info['match_failure_reason']}")
                        print(f"     → Generated new internal_number: {non_origine.internal_number}")
                    logger.warning(f"Tax transaction {non_origine.id} (type={non_origine.type}) UNMATCHED - {tax_info['match_failure_reason']}")
            
            # Print summary of tax transactions
            print(f"\n{'=' * 80}")
            print("BTMatchBankTransactionTaxesView: Tax Transactions Summary")
            print(f"{'=' * 80}")
            print(f"Total tax (non-origine) transactions: {len(non_origine_transactions)}")
            print(f"Matched tax transactions: {matched_count}")
            print(f"Unmatched tax transactions: {unmatched_count}")
            print(f"Match rate: {100*matched_count/len(non_origine_transactions) if len(non_origine_transactions) > 0 else 0:.1f}%")
            print(f"{'=' * 80}")
            
            logger.info("=" * 80)
            logger.info("BTMatchBankTransactionTaxesView: Tax Transactions Summary")
            logger.info(f"Total tax transactions: {len(non_origine_transactions)}, Matched: {matched_count}, Unmatched: {unmatched_count}")
            logger.info(f"Match rate: {100*matched_count/len(non_origine_transactions) if len(non_origine_transactions) > 0 else 0:.1f}%")
            
            # Log detailed breakdown by tax type
            if tax_transactions_matched or tax_transactions_unmatched:
                print(f"\n📊 Detailed Tax Transaction Breakdown:")
                print(f"   Matched Tax Transactions: {len(tax_transactions_matched)}")
                for tax in tax_transactions_matched[:5]:  # Show first 5
                    print(f"     - ID={tax['id']}, Type={tax['type']}, Amount={tax['amount']}, Match Method={tax['match_method']}, Origine ID={tax['matching_origine_id']}")
                if len(tax_transactions_matched) > 5:
                    print(f"     ... and {len(tax_transactions_matched) - 5} more matched transactions")
                
                print(f"\n   Unmatched Tax Transactions: {len(tax_transactions_unmatched)}")
                for tax in tax_transactions_unmatched[:5]:  # Show first 5
                    print(f"     - ID={tax['id']}, Type={tax['type']}, Amount={tax['amount']}, Reason={tax['match_failure_reason']}")
                if len(tax_transactions_unmatched) > 5:
                    print(f"     ... and {len(tax_transactions_unmatched) - 5} more unmatched transactions")
                
                logger.info(f"Matched tax transactions details: {tax_transactions_matched[:10]}")
                logger.info(f"Unmatched tax transactions details: {tax_transactions_unmatched[:10]}")
            
            # Also ensure all origine transactions have internal_number
            for origine in origine_transactions:
                if not origine.internal_number:
                    origine.internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
                    origine.save()

            # === Custom logic for agios transactions (Probabilistic Group-Based Allocation) ===
            agios_transactions = RecoBankTransaction.objects.filter(type='agios')
            agios_by_date_ref = defaultdict(list)
            
            # Initialize tracking variables BEFORE processing agios
            agios_matched_count = 0
            agios_unmatched_count = 0
            agios_transactions_matched = []
            agios_transactions_unmatched = []
            
            for agios in agios_transactions:
                if agios.date_ref:
                    agios_by_date_ref[agios.date_ref].append(agios)
                else:
                    logger.warning(f"Agios transaction {agios.id} skipped: no date_ref (amount: {agios.amount})")
                    # Log unmatched agios (no date_ref)
                    agios_info = {
                        'id': agios.id,
                        'type': 'agios',
                        'amount': float(agios.amount) if agios.amount else None,
                        'date_ref': None,
                        'label': agios.label[:50] if agios.label else None,
                        'operation_date': agios.operation_date.isoformat() if agios.operation_date else None,
                        'ref': agios.ref,
                        'matched': False,
                        'match_method': None,
                        'matching_origine_id': None,
                        'matching_origine_internal_number': None,
                        'match_failure_reason': 'No date_ref - cannot match agios without date_ref',
                        'checked_doc_numbers_count': 0,
                        'no_tax_rows_docs_count': 0,
                        'mismatch_docs_count': 0,
                        'checked_doc_numbers': None,
                        'no_tax_rows_doc_numbers': None,
                        'mismatch_doc_details': None
                    }
                    agios_transactions_unmatched.append(agios_info)
                    agios_unmatched_count += 1
            
            print(f"\n{'=' * 80}")
            print("BTMatchBankTransactionTaxesView: Processing AGIOS transactions")
            print(f"{'=' * 80}")
            logger.info(f"Starting agios matching process. Found {agios_transactions.count()} agios transactions grouped into {len(agios_by_date_ref)} date_ref groups.")
            logger.info(f"Already found {agios_unmatched_count} agios without date_ref (unmatched)")
            
            # Process each date_ref group
            for date_ref, agios_group in agios_by_date_ref.items():
                logger.info(f"Processing date_ref={date_ref} with {len(agios_group)} agios transactions")
                
                # STEP 2: Identify candidate customer transactions
                customer_candidates = RecoCustomerTransaction.objects.filter(
                    accounting_date=date_ref
                )
                
                if customer_candidates.count() == 0:
                    logger.warning(f"Date_ref {date_ref}: No customer transactions found")
                    # Log all agios in this group as unmatched (no customer transactions found)
                    for agios in agios_group:
                        agios_info = {
                            'id': agios.id,
                            'type': 'agios',
                            'amount': float(agios.amount) if agios.amount else None,
                            'date_ref': agios.date_ref.isoformat() if agios.date_ref else None,
                            'label': agios.label[:50] if agios.label else None,
                            'operation_date': agios.operation_date.isoformat() if agios.operation_date else None,
                            'ref': agios.ref,
                            'matched': False,
                            'match_method': None,
                            'matching_origine_id': None,
                            'matching_origine_internal_number': None,
                            'match_failure_reason': f'No customer transactions found for date_ref={date_ref}',
                            'checked_doc_numbers_count': 0,
                            'no_tax_rows_docs_count': 0,
                            'mismatch_docs_count': 0,
                            'checked_doc_numbers': None,
                            'no_tax_rows_doc_numbers': None,
                            'mismatch_doc_details': None
                        }
                        agios_transactions_unmatched.append(agios_info)
                        agios_unmatched_count += 1
                    continue
                
                logger.info(f"Date_ref {date_ref}: Found {customer_candidates.count()} candidate customer transactions")
                
                # Group customer transactions by document_number (document group)
                customer_by_doc = defaultdict(list)
                for cust_tx in customer_candidates:
                    if cust_tx.document_number:
                        customer_by_doc[cust_tx.document_number].append(cust_tx)
                
                logger.info(f"Date_ref {date_ref}: Grouped into {len(customer_by_doc)} document_number groups")
                
                # Process each agios in this date_ref group
                for agios_idx, agios in enumerate(agios_group, 1):
                    matched = False
                    agios_amount = abs(float(agios.amount))
                    
                    # Log agios transaction details
                    agios_info = {
                        'id': agios.id,
                        'type': 'agios',
                        'amount': float(agios.amount) if agios.amount else None,
                        'date_ref': agios.date_ref.isoformat() if agios.date_ref else None,
                        'label': agios.label[:50] if agios.label else None,
                        'operation_date': agios.operation_date.isoformat() if agios.operation_date else None,
                        'ref': agios.ref
                    }
                    
                    logger.info(f"Processing agios {agios.id}: amount={agios.amount}, date_ref={date_ref}")
                    
                    if agios_idx <= 5 or len(agios_group) <= 10:
                        print(f"\n  [{agios_idx}/{len(agios_group)}] 🔍 Processing AGIOS Transaction ID={agios.id}")
                        print(f"      Amount: {agios.amount}")
                        print(f"      Date Ref: {agios.date_ref}")
                        print(f"      Label: '{agios.label[:50] if agios.label else None}'")
                        print(f"      Operation Date: {agios.operation_date}")
                    
                    # STEP 3 & 4: Compute simulated agios and match using tolerance
                    best_match = None
                    best_difference = float('inf')
                    checked_doc_numbers = []
                    no_tax_rows_docs = []
                    mismatch_docs = []
                    checked_customer_details = []  # Store detailed info about each checked customer group
                    
                    for doc_number, cust_tx_group in customer_by_doc.items():
                        checked_doc_numbers.append(doc_number)
                        
                        # Calculate total amount for this document group
                        total_amount = sum(abs(float(tx.total_amount or tx.amount)) for tx in cust_tx_group)
                        
                        agios_tax_rows = CustomerTaxRow.objects.filter(
                            transaction__in=cust_tx_group,
                            tax_type__iexact='agios'
                        )
                        
                        logger.debug(f"Agios {agios.id}, doc_number={doc_number}: Found {agios_tax_rows.count()} agios tax rows for {len(cust_tx_group)} customer transactions")
                        
                        simulated_agios = agios_tax_rows.aggregate(
                            total=Sum('tax_amount')
                        )['total'] or 0
                        
                        # Store customer transaction details
                        customer_detail = {
                            'doc_number': doc_number,
                            'total_amount': total_amount,
                            'simulated_agios': abs(float(simulated_agios)) if simulated_agios else 0,
                            'transaction_count': len(cust_tx_group),
                            'customer_transactions': [
                                {
                                    'id': tx.id,
                                    'document_number': tx.document_number,
                                    'external_doc_number': tx.external_doc_number,
                                    'accounting_date': tx.accounting_date.isoformat() if tx.accounting_date else None,
                                    'description': tx.description,
                                    'amount': float(tx.amount) if tx.amount else None,
                                    'total_amount': float(tx.total_amount) if tx.total_amount else None,
                                    'debit_amount': float(tx.debit_amount) if tx.debit_amount else None,
                                    'credit_amount': float(tx.credit_amount) if tx.credit_amount else None,
                                    'payment_type': tx.payment_type,
                                    'account_number': tx.account_number
                                }
                                for tx in cust_tx_group
                            ]
                        }
                        checked_customer_details.append(customer_detail)
                        
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
                            
                            # Log successful agios match
                            agios_info['matched'] = True
                            agios_info['match_method'] = 'customer_tax_matching'
                            agios_info['matching_origine_id'] = matching_origine.id
                            agios_info['matching_origine_ref'] = matching_origine.ref
                            agios_info['matching_origine_date_ref'] = matching_origine.date_ref.isoformat() if matching_origine.date_ref else None
                            agios_info['matching_origine_internal_number'] = matching_origine.internal_number
                            agios_info['matching_origine_amount'] = float(matching_origine.amount) if matching_origine.amount else None
                            agios_info['customer_doc_number'] = best_match['doc_number']
                            agios_info['simulated_agios'] = best_match['simulated_agios']
                            agios_info['difference'] = best_match['difference']
                            agios_info['is_total_match'] = best_match.get('is_total_match', False)
                            agios_transactions_matched.append(agios_info)
                            
                            if agios_idx <= 5 or len(agios_group) <= 10:
                                print(f"      ✅ AGIOS MATCHED: Transaction ID={agios.id}")
                                print(f"         Agios Amount: {agios.amount}")
                                print(f"         Date Ref: {agios.date_ref}")
                                print(f"         → Matched to Origine ID={matching_origine.id}")
                                print(f"            Origine Ref: '{matching_origine.ref}'")
                                print(f"            Origine Date Ref: {matching_origine.date_ref}")
                                print(f"            Internal Number: {matching_origine.internal_number}")
                                print(f"            Origine Amount: {matching_origine.amount}")
                                print(f"            Customer Doc Number: {best_match['doc_number']}")
                                print(f"            Simulated Agios: {best_match['simulated_agios']:.2f}")
                                print(f"            Difference: {best_match['difference']:.2f}")
                                print(f"            Match Type: {'Total Match (sum of agios)' if best_match.get('is_total_match') else 'Individual Match'}")
                            logger.info(f"Agios transaction {agios.id} matched to origine {matching_origine.id} via customer doc_number={best_match['doc_number']}")
                        elif not origine_found:
                            logger.warning(f"Agios {agios.id}: Amount matched but no origine transaction found for customer group (doc_number={best_match['doc_number']})")
                            agios_unmatched_count += 1
                            
                            # Log unmatched agios (amount matched but no origine found)
                            agios_info['matched'] = False
                            agios_info['match_method'] = None
                            agios_info['matching_origine_id'] = None
                            agios_info['matching_origine_internal_number'] = None
                            agios_info['match_failure_reason'] = f"Amount matched to customer doc_number={best_match['doc_number']} but no origine transaction found"
                            agios_info['customer_doc_number'] = best_match['doc_number']
                            agios_info['simulated_agios'] = best_match['simulated_agios']
                            agios_info['difference'] = best_match['difference']
                            agios_info['checked_doc_numbers_count'] = len(checked_doc_numbers)
                            agios_info['no_tax_rows_docs_count'] = len(no_tax_rows_docs)
                            agios_info['mismatch_docs_count'] = len(mismatch_docs)
                        agios_info['checked_doc_numbers'] = ','.join(checked_doc_numbers) if checked_doc_numbers else None
                        agios_info['no_tax_rows_doc_numbers'] = ','.join(no_tax_rows_docs) if no_tax_rows_docs else None
                        mismatch_details_str = '; '.join([
                            f"{m['doc_number']}:sim={m['simulated']:.2f}:bank={m['bank']:.2f}:diff={m['difference']:.2f}:ratio={m['tolerance_ratio']:.4f}"
                            for m in mismatch_docs
                        ]) if mismatch_docs else None
                        agios_info['mismatch_doc_details'] = mismatch_details_str
                        # Store customer transaction details as JSON string
                        import json
                        agios_info['checked_customer_transactions_details'] = json.dumps(checked_customer_details) if 'checked_customer_details' in locals() and checked_customer_details else None
                        agios_transactions_unmatched.append(agios_info)
                        
                        if agios_idx <= 5 or len(agios_group) <= 10:
                                print(f"      ⚠️ AGIOS PARTIALLY MATCHED (no origine): Transaction ID={agios.id}")
                                print(f"         Agios Amount: {agios.amount}")
                                print(f"         Date Ref: {agios.date_ref}")
                                print(f"         Customer Doc Number: {best_match['doc_number']}")
                                print(f"         Reason: Amount matched but no origine transaction found")
                    else:
                        logger.warning(f"Agios {agios.id}: Could not find matching customer group within 5% tolerance (amount={agios.amount}, date_ref={date_ref})")
                        agios_unmatched_count += 1
                        
                        # Log unmatched agios (no customer match found)
                        agios_info['matched'] = False
                        agios_info['match_method'] = None
                        agios_info['matching_origine_id'] = None
                        agios_info['matching_origine_internal_number'] = None
                        agios_info['match_failure_reason'] = f"Could not find matching customer group within 5% tolerance (checked {len(checked_doc_numbers)} doc_numbers)"
                        agios_info['checked_doc_numbers_count'] = len(checked_doc_numbers)
                        agios_info['no_tax_rows_docs_count'] = len(no_tax_rows_docs)
                        agios_info['mismatch_docs_count'] = len(mismatch_docs)
                        # Store actual checked document numbers
                        agios_info['checked_doc_numbers'] = ','.join(checked_doc_numbers) if checked_doc_numbers else None
                        # Store details of documents with no tax rows
                        agios_info['no_tax_rows_doc_numbers'] = ','.join(no_tax_rows_docs) if no_tax_rows_docs else None
                        # Store details of mismatch documents (format: doc_number:simulated:bank:diff:ratio)
                        mismatch_details_str = '; '.join([
                            f"{m['doc_number']}:sim={m['simulated']:.2f}:bank={m['bank']:.2f}:diff={m['difference']:.2f}:ratio={m['tolerance_ratio']:.4f}"
                            for m in mismatch_docs
                        ]) if mismatch_docs else None
                        agios_info['mismatch_doc_details'] = mismatch_details_str
                        # Store customer transaction details as JSON string
                        import json
                        agios_info['checked_customer_transactions_details'] = json.dumps(checked_customer_details) if 'checked_customer_details' in locals() and checked_customer_details else None
                        agios_transactions_unmatched.append(agios_info)
                        
                        if agios_idx <= 5 or len(agios_group) <= 10:
                            print(f"      ❌ AGIOS UNMATCHED: Transaction ID={agios.id}")
                            print(f"         Agios Amount: {agios.amount}")
                            print(f"         Date Ref: {agios.date_ref}")
                            print(f"         Reason: No matching customer group found within 5% tolerance")
                            print(f"         Checked {len(checked_doc_numbers)} document numbers")
                            if no_tax_rows_docs:
                                print(f"         - {len(no_tax_rows_docs)} docs with no agios tax rows")
                            if mismatch_docs:
                                print(f"         - {len(mismatch_docs)} docs with amount mismatches")
            
            logger.info(f"Agios matching completed: {agios_matched_count} matched, {agios_unmatched_count} unmatched out of {agios_transactions.count()} total")
            
            # Print summary of agios transactions
            print(f"\n{'=' * 80}")
            print("BTMatchBankTransactionTaxesView: AGIOS Transactions Summary")
            print(f"{'=' * 80}")
            print(f"Total agios transactions: {agios_transactions.count()}")
            print(f"Matched agios transactions: {agios_matched_count}")
            print(f"Unmatched agios transactions: {agios_unmatched_count}")
            print(f"Match rate: {100*agios_matched_count/agios_transactions.count() if agios_transactions.count() > 0 else 0:.1f}%")
            print(f"{'=' * 80}")
            
            logger.info("=" * 80)
            logger.info("BTMatchBankTransactionTaxesView: AGIOS Transactions Summary")
            logger.info(f"Total agios transactions: {agios_transactions.count()}, Matched: {agios_matched_count}, Unmatched: {agios_unmatched_count}")
            logger.info(f"Match rate: {100*agios_matched_count/agios_transactions.count() if agios_transactions.count() > 0 else 0:.1f}%")
            
            # Log detailed breakdown of agios
            if agios_transactions_matched or agios_transactions_unmatched:
                print(f"\n📊 Detailed AGIOS Transaction Breakdown:")
                print(f"   Matched AGIOS Transactions: {len(agios_transactions_matched)}")
                for idx, agios in enumerate(agios_transactions_matched, 1):
                    print(f"     [{idx}] ID={agios['id']}, Amount={agios['amount']}, Origine ID={agios['matching_origine_id']}, Doc Number={agios.get('customer_doc_number')}, Match Type={'Total Match' if agios.get('is_total_match') else 'Individual Match'}")
                
                print(f"\n   Unmatched AGIOS Transactions: {len(agios_transactions_unmatched)}")
                # Group unmatched by reason for better analysis
                unmatched_by_reason = defaultdict(list)
                for agios in agios_transactions_unmatched:
                    reason = agios.get('match_failure_reason', 'Unknown reason')
                    unmatched_by_reason[reason].append(agios)
                
                for reason, agios_list in unmatched_by_reason.items():
                    print(f"\n     Reason: {reason} ({len(agios_list)} agios)")
                    for idx, agios in enumerate(agios_list, 1):
                        print(f"       [{idx}] ID={agios['id']}, Amount={agios['amount']}, Date Ref={agios.get('date_ref')}, Label='{agios.get('label', '')[:40]}'")
                
                # Also show all unmatched in a flat list
                print(f"\n   All Unmatched AGIOS (flat list):")
                for idx, agios in enumerate(agios_transactions_unmatched, 1):
                    print(f"     [{idx}] ID={agios['id']}, Amount={agios['amount']}, Date Ref={agios.get('date_ref')}, Reason={agios.get('match_failure_reason', 'Unknown')}")
                
                logger.info(f"Matched agios transactions details: {agios_transactions_matched}")
                logger.info(f"Unmatched agios transactions details: {agios_transactions_unmatched}")
                logger.info(f"Unmatched agios grouped by reason: {dict(unmatched_by_reason)}")
            
            # Create comprehensive DataFrame for exploration
            print(f"\n{'=' * 80}")
            print("Creating AGIOS Exploration DataFrame...")
            print(f"{'=' * 80}")
            
            agios_exploration_data = []
            
            # Add all matched agios
            for agios in agios_transactions_matched:
                matching_origine = RecoBankTransaction.objects.filter(id=agios['matching_origine_id']).first() if agios.get('matching_origine_id') else None
                agios_exploration_data.append({
                    'agios_id': agios['id'],
                    'agios_amount': agios['amount'],
                    'agios_date_ref': agios.get('date_ref'),
                    'agios_operation_date': agios.get('operation_date'),
                    'agios_label': agios.get('label'),
                    'agios_ref': agios.get('ref'),
                    'matched': True,
                    'match_method': agios.get('match_method'),
                    'origine_id': agios.get('matching_origine_id'),
                    'origine_ref': agios.get('matching_origine_ref'),
                    'origine_date_ref': agios.get('matching_origine_date_ref'),
                    'origine_amount': agios.get('matching_origine_amount'),
                    'origine_internal_number': agios.get('matching_origine_internal_number'),
                    'customer_doc_number': agios.get('customer_doc_number'),
                    'simulated_agios': agios.get('simulated_agios'),
                    'difference': agios.get('difference'),
                    'is_total_match': agios.get('is_total_match', False),
                    'match_failure_reason': None,
                    'checked_doc_numbers_count': None,
                    'no_tax_rows_docs_count': None,
                    'mismatch_docs_count': None,
                    'checked_doc_numbers': None,
                    'no_tax_rows_doc_numbers': None,
                    'mismatch_doc_details': None
                })
            
            # Add all unmatched agios
            for agios in agios_transactions_unmatched:
                agios_exploration_data.append({
                    'agios_id': agios['id'],
                    'agios_amount': agios['amount'],
                    'agios_date_ref': agios.get('date_ref'),
                    'agios_operation_date': agios.get('operation_date'),
                    'agios_label': agios.get('label'),
                    'agios_ref': agios.get('ref'),
                    'matched': False,
                    'match_method': None,
                    'origine_id': None,
                    'origine_ref': None,
                    'origine_date_ref': None,
                    'origine_amount': None,
                    'origine_internal_number': None,
                    'customer_doc_number': agios.get('customer_doc_number'),
                    'simulated_agios': agios.get('simulated_agios'),
                    'difference': agios.get('difference'),
                    'is_total_match': False,
                    'match_failure_reason': agios.get('match_failure_reason'),
                    'checked_doc_numbers_count': agios.get('checked_doc_numbers_count'),
                    'no_tax_rows_docs_count': agios.get('no_tax_rows_docs_count'),
                    'mismatch_docs_count': agios.get('mismatch_docs_count'),
                    'checked_doc_numbers': agios.get('checked_doc_numbers'),
                    'no_tax_rows_doc_numbers': agios.get('no_tax_rows_doc_numbers'),
                    'mismatch_doc_details': agios.get('mismatch_doc_details'),
                    'checked_customer_transactions_details': agios.get('checked_customer_transactions_details')
                })
            
            # Create DataFrame
            df_agios_exploration = pd.DataFrame(agios_exploration_data)
            
            # Sort by matched status (matched first), then by date_ref, then by amount
            df_agios_exploration['matched_sort'] = df_agios_exploration['matched'].apply(lambda x: 0 if x else 1)
            df_agios_exploration = df_agios_exploration.sort_values(
                ['matched_sort', 'agios_date_ref', 'agios_amount'],
                ascending=[True, True, False]
            ).drop(columns=['matched_sort'])
            
            # Sanitize DataFrame to remove infinity and NaN values before JSON serialization
            df_agios_exploration = sanitize_dataframe_for_json(df_agios_exploration)
            
            # Save to CSV for easy exploration
            output_dir = os.path.join(settings.BASE_DIR, 'matching_results')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            agios_exploration_file = os.path.join(output_dir, f'agios_exploration_{timestamp}.csv')
            df_agios_exploration.to_csv(agios_exploration_file, index=False)
            
            print(f"\n✅ Created AGIOS Exploration DataFrame:")
            print(f"   Total agios: {len(df_agios_exploration)}")
            print(f"   Matched: {df_agios_exploration['matched'].sum()}")
            print(f"   Unmatched: {(~df_agios_exploration['matched']).sum()}")
            print(f"   Saved to: {agios_exploration_file}")
            print(f"\n   DataFrame shape: {df_agios_exploration.shape}")
            print(f"   Columns: {list(df_agios_exploration.columns)}")
            
            # Show sample of the DataFrame
            print(f"\n   Sample (first 5 rows):")
            print(df_agios_exploration.head().to_string())
            
            # Enhanced logging for unmatched agios
            unmatched_df = df_agios_exploration[~df_agios_exploration['matched']]
            if len(unmatched_df) > 0:
                print(f"\n{'=' * 80}")
                print("📋 DETAILED ANALYSIS: Unmatched AGIOS Transactions")
                print(f"{'=' * 80}")
                print(f"Total unmatched: {len(unmatched_df)}")
                
                # Group by failure reason for detailed analysis
                unmatched_by_reason = unmatched_df.groupby('match_failure_reason')
                print(f"\n   Breakdown by failure reason:")
                for reason, group in unmatched_by_reason:
                    print(f"\n   ┌─ {reason} ({len(group)} agios)")
                    print(f"   │")
                    for idx, (_, row) in enumerate(group.iterrows(), 1):
                        print(f"   │  [{idx}] Agios ID={row['agios_id']}")
                        print(f"   │      Amount: {abs(row['agios_amount']):.2f}")
                        print(f"   │      Date Ref: {row['agios_date_ref']}")
                        print(f"   │      Operation Date: {row['agios_operation_date']}")
                        print(f"   │      Label: '{row['agios_label']}'")
                        
                        # Show detailed mismatch information if available
                        if pd.notna(row.get('checked_doc_numbers_count')) and row['checked_doc_numbers_count'] > 0:
                            print(f"   │      📊 Matching Attempts:")
                            print(f"   │         - Checked {int(row['checked_doc_numbers_count'])} document numbers")
                            if pd.notna(row.get('no_tax_rows_docs_count')) and row['no_tax_rows_docs_count'] > 0:
                                print(f"   │         - {int(row['no_tax_rows_docs_count'])} docs had no agios tax rows")
                            if pd.notna(row.get('mismatch_docs_count')) and row['mismatch_docs_count'] > 0:
                                print(f"   │         - {int(row['mismatch_docs_count'])} docs had amount mismatches (>5% tolerance)")
                        
                        # Show customer doc number if partially matched
                        if pd.notna(row.get('customer_doc_number')):
                            print(f"   │      Customer Doc Number: {row['customer_doc_number']}")
                            if pd.notna(row.get('simulated_agios')):
                                print(f"   │      Simulated Agios: {abs(row['simulated_agios']):.2f}")
                                print(f"   │      Bank Agios: {abs(row['agios_amount']):.2f}")
                                diff = abs(row['agios_amount']) - abs(row['simulated_agios'])
                                print(f"   │      Difference: {abs(diff):.2f}")
                                if abs(row['agios_amount']) > 0:
                                    tolerance_pct = (abs(diff) / abs(row['agios_amount'])) * 100
                                    print(f"   │      Tolerance: {tolerance_pct:.2f}% (threshold: 5%)")
                        print(f"   │")
                    print(f"   └─")
                
                # Show summary statistics for unmatched
                print(f"\n   📊 Unmatched AGIOS Statistics:")
                unmatched_amounts = unmatched_df['agios_amount'].abs()
                print(f"      Total amount: {unmatched_amounts.sum():.2f}")
                print(f"      Average amount: {unmatched_amounts.mean():.2f}")
                print(f"      Min amount: {unmatched_amounts.min():.2f}")
                print(f"      Max amount: {unmatched_amounts.max():.2f}")
                
                # Date analysis
                if unmatched_df['agios_date_ref'].notna().any():
                    date_ref_counts = unmatched_df['agios_date_ref'].value_counts()
                    print(f"\n      📅 Date Ref distribution:")
                    for date_ref, count in date_ref_counts.items():
                        total_amount = unmatched_df[unmatched_df['agios_date_ref'] == date_ref]['agios_amount'].abs().sum()
                        print(f"        {date_ref}: {count} agios, Total: {total_amount:.2f}")
                
                # Show detailed breakdown for tolerance mismatches
                tolerance_mismatches = unmatched_df[
                    unmatched_df['match_failure_reason'].str.contains('tolerance', case=False, na=False)
                ]
                if len(tolerance_mismatches) > 0:
                    print(f"\n   🔍 Detailed Analysis: Tolerance Mismatches ({len(tolerance_mismatches)} agios)")
                    print(f"      These agios had customer transactions but amounts didn't match within 5%:")
                    for idx, (_, row) in enumerate(tolerance_mismatches.iterrows(), 1):
                        print(f"\n      [{idx}] Agios ID={row['agios_id']}, Amount={abs(row['agios_amount']):.2f}")
                        print(f"          Date Ref: {row['agios_date_ref']}")
                        if pd.notna(row.get('checked_doc_numbers_count')):
                            print(f"          Checked {int(row['checked_doc_numbers_count'])} document numbers")
                            if pd.notna(row.get('no_tax_rows_docs_count')):
                                print(f"          - {int(row['no_tax_rows_docs_count'])} docs had no agios tax rows")
                            if pd.notna(row.get('mismatch_docs_count')):
                                print(f"          - {int(row['mismatch_docs_count'])} docs had amount mismatches")
                
                # Show detailed breakdown for no customer transactions
                no_customer_txs = unmatched_df[
                    unmatched_df['match_failure_reason'].str.contains('No customer transactions', case=False, na=False)
                ]
                if len(no_customer_txs) > 0:
                    print(f"\n   ⚠️  Detailed Analysis: No Customer Transactions ({len(no_customer_txs)} agios)")
                    print(f"      These agios had date_ref but no customer transactions found:")
                    date_ref_groups = no_customer_txs.groupby('agios_date_ref')
                    for date_ref, group in date_ref_groups:
                        total_amount = group['agios_amount'].abs().sum()
                        print(f"\n      Date Ref: {date_ref} ({len(group)} agios, Total: {total_amount:.2f})")
                        for idx, (_, row) in enumerate(group.iterrows(), 1):
                            print(f"        [{idx}] Agios ID={row['agios_id']}, Amount={abs(row['agios_amount']):.2f}, Label='{row['agios_label']}'")
                
                # Show full unmatched DataFrame
                print(f"\n   📄 Full Unmatched AGIOS DataFrame:")
                print(unmatched_df.to_string(index=False))
                
                logger.info(f"Unmatched agios detailed analysis: {len(unmatched_df)} unmatched")
                logger.info(f"Unmatched agios by reason: {dict(unmatched_by_reason.size())}")
            
            logger.info(f"Created agios exploration DataFrame with {len(df_agios_exploration)} rows")
            logger.info(f"Saved to: {agios_exploration_file}")
            
            # === Final pass: Ensure all transactions with same internal_number inherit payment_class and payment_status from origine ===
            logger.info("Starting final pass to inherit payment_class and payment_status from origine transactions")
            inheritance_count = 0
            for origine in origine_transactions:
                if origine.internal_number and (origine.payment_class or origine.payment_status):
                    # Find all transactions (non-origine and agios) with the same internal_number
                    related_transactions = RecoBankTransaction.objects.filter(
                        internal_number=origine.internal_number
                    ).exclude(type='origine')
                    
                    for related_tx in related_transactions:
                        updated = False
                        if origine.payment_class and related_tx.payment_class != origine.payment_class:
                            related_tx.payment_class = origine.payment_class
                            updated = True
                        if origine.payment_status and related_tx.payment_status != origine.payment_status:
                            related_tx.payment_status = origine.payment_status
                            updated = True
                        
                        if updated:
                            related_tx.save()
                            inheritance_count += 1
                            logger.debug(f"Inherited payment_class/payment_status from origine {origine.id} to transaction {related_tx.id} (internal_number={origine.internal_number})")
            
            logger.info(f"Final pass completed: {inheritance_count} transactions inherited payment_class/payment_status from origine transactions")

            return Response({
                "message": f"Successfully processed reco bank transaction taxes matching",
                "total_transactions": all_transactions.count(),
                "origine_transactions": origine_transactions.count(),
                "non_origine_transactions": non_origine_transactions.count(),
                "matched_count": matched_count,
                "unmatched_count": unmatched_count,
                "agios_matched_count": agios_matched_count,
                "agios_unmatched_count": agios_unmatched_count,
                "tax_transactions_summary": {
                    "total_tax_transactions": len(non_origine_transactions),
                    "matched_tax_count": matched_count,
                    "unmatched_tax_count": unmatched_count,
                    "match_rate_percent": round(100*matched_count/len(non_origine_transactions) if len(non_origine_transactions) > 0 else 0, 1)
                },
                "tax_transactions_matched": tax_transactions_matched,
                "tax_transactions_unmatched": tax_transactions_unmatched,
                "agios_transactions_summary": {
                    "total_agios_transactions": agios_transactions.count(),
                    "matched_agios_count": agios_matched_count,
                    "unmatched_agios_count": agios_unmatched_count,
                    "match_rate_percent": round(100*agios_matched_count/agios_transactions.count() if agios_transactions.count() > 0 else 0, 1)
                },
                "agios_transactions_matched": agios_transactions_matched,
                "agios_transactions_unmatched": agios_transactions_unmatched,
                "agios_exploration_dataframe": {
                    "file_path": agios_exploration_file if 'agios_exploration_file' in locals() else None,
                    "total_rows": len(df_agios_exploration) if 'df_agios_exploration' in locals() else 0,
                    "matched_rows": int(df_agios_exploration['matched'].sum()) if 'df_agios_exploration' in locals() else 0,
                    "unmatched_rows": int((~df_agios_exploration['matched']).sum()) if 'df_agios_exploration' in locals() else 0,
                    "columns": list(df_agios_exploration.columns) if 'df_agios_exploration' in locals() else [],
                    "data": df_agios_exploration.to_dict('records') if 'df_agios_exploration' in locals() else []
                },
                "total_with_internal_number": RecoBankTransaction.objects.filter(internal_number__isnull=False).count()
            }, status=200)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                "error": str(e),
                "details": error_details
            }, status=500)


class BTExtractCustomerTaxesView(APIView):
    """
    View to extract tax rows for customer transactions based on conventions and smart formula engine.
    """
    def post(self, request, *args, **kwargs):
        transactions = request.data.get("transactions", None)
        extracted_taxes = []
        if transactions is None:
            transactions = RecoCustomerTransaction.objects.all()

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
                ).order_by('-id')
                
                seen_tax_types = {}
                tax_rules = []
                for tr in tax_rules_queryset:
                    if hasattr(tr, 'tax_type') and tr.tax_type:
                        key = tr.tax_type.lower().strip()
                        if key not in seen_tax_types:
                            seen_tax_types[key] = tr
                            tax_rules.append(tr)
                
                params = {p.name: p.value for p in ConventionParameter.objects.all()}
                tax_rule_lookup = {}
                for tr in tax_rules:
                    if hasattr(tr, 'tax_type') and tr.tax_type:
                        key = tr.tax_type.lower().strip()
                        tax_rule_lookup[key] = tr
                tax_results = {}
                for rule in tax_rules:
                    if rule.calculation_type == "flat":
                        val = getattr(rule, 'rate', getattr(rule, 'value', None))
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
                        except Exception as e:
                            print(f"CustomerTaxRow flat error: {e}")
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
                                import decimal
                                if isinstance(value, decimal.Decimal):
                                    value = float(value)
                                local_vars[var] = value
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
                            expr = rule.formula
                            result = custom_eval(expr, local_vars)
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


class BTMatchCustomerBankTransactionsView1(APIView):
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


class BTPreprocessCustomerLedgerEntryView(APIView):
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


class BTGetMatchingResultsView(APIView):
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


class BTTaxComparisonView(APIView):
    def get(self, request):
        """
        GET endpoint to retrieve comparison data from the Comparison table.
        Returns all comparison records with customer_tax (individual) and customer_total_tax (aggregated).
        
        Query parameters:
        - customer_transaction_id: Filter by specific customer transaction ID
        - tax_type: Filter by tax type (e.g., 'AGIOS', 'COM', etc.)
        - status: Filter by comparison status ('matched', 'mismatch', 'missing')
        """
        try:
            queryset = Comparison.objects.select_related('customer_transaction').all()
            
            # Apply filters
            customer_transaction_id = request.query_params.get('customer_transaction_id')
            if customer_transaction_id:
                queryset = queryset.filter(customer_transaction_id=customer_transaction_id)
            
            tax_type = request.query_params.get('tax_type')
            if tax_type:
                queryset = queryset.filter(tax_type__iexact=tax_type)
            
            status_filter = request.query_params.get('status')
            if status_filter:
                queryset = queryset.filter(status__iexact=status_filter)
            
            results = []
            for comp in queryset:
                results.append({
                    'id': comp.id,
                    'customer_transaction_id': comp.customer_transaction_id,
                    'matched_bank_transaction_id': comp.matched_bank_transaction_id,
                    'tax_type': comp.tax_type,
                    'customer_tax': format(comp.customer_tax, '.3f') if comp.customer_tax is not None else None,  # Individual tax amount
                    'bank_tax': format(comp.bank_tax, '.3f') if comp.bank_tax is not None else None,
                    'status': comp.status,
                    'difference': format(comp.difference, '.3f') if comp.difference is not None else None,
                    'customer_total_tax': format(comp.customer_total_tax, '.3f') if comp.customer_total_tax is not None else None,  # Aggregated total tax amount
                })
            
            return Response({
                'count': len(results),
                'results': results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"BTTaxComparisonView.get error: {str(e)}\n{error_details}")
            return Response({
                'error': str(e),
                'details': error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
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
                
                # Use total_tax_amount for comparison (aggregated)
                abs_cust_val = abs(Decimal(str(customer_tax_for_comparison))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                abs_bank_val = abs(Decimal(str(bank_tax or 0))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
                
                # Calculate difference (customer - bank) using absolute values to match status calculation
                # This ensures consistent comparison regardless of sign of original values
                difference_val = abs_cust_val - abs_bank_val
                
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
                        'difference': difference_val,
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
                    'customer_tax': format(abs_cust_individual, '.3f'),  # Individual tax amount for this transaction
                    'bank_tax': format(abs_bank_val, '.3f'),
                    'status': status_str,
                    'difference': format(difference_val, '.3f'),
                    'customer_total_tax': format(row.total_tax_amount, '.3f') if row.total_tax_amount is not None else None,  # Aggregated total tax amount
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


class BTSumMatchedBankTransactionsView(APIView):
    """
    GET endpoint to calculate the sum of amounts for all matched RecoBankTransactions.
    
    A RecoBankTransaction is considered "matched" if it has at least one 
    RecoCustomerTransaction that references it via matched_bank_transaction.
    
    Returns:
    - total_sum: Sum of amounts for all matched bank transactions
    - count: Number of matched bank transactions
    - transactions: List of matched transactions with their details (optional, if detail=true)
    
    Query parameters:
    - detail: If true, returns list of matched transactions with details
    - bank_id: Optional filter by bank ID
    - type: Optional filter by transaction type (e.g., 'origine', 'agios', 'COM', etc.)
    """
    
    def get(self, request):
        try:
            # Get matched RecoBankTransactions (those that have at least one matched customer transaction)
            matched_bank_transactions = RecoBankTransaction.objects.filter(
                matched_reco_customer_transactions__isnull=False
            ).distinct()
            
            # Apply optional filters
            bank_id = request.query_params.get('bank_id')
            if bank_id:
                matched_bank_transactions = matched_bank_transactions.filter(bank_id=bank_id)
            
            transaction_type = request.query_params.get('type')
            if transaction_type:
                matched_bank_transactions = matched_bank_transactions.filter(type__iexact=transaction_type)
            
            # Calculate sum of amounts
            total_sum = matched_bank_transactions.aggregate(
                total=Sum('amount')
            )['total'] or Decimal('0.000')
            
            # Count matched transactions
            count = matched_bank_transactions.count()
            
            # Build response
            response_data = {
                'total_sum': format(total_sum, '.3f'),
                'total_sum_decimal': float(total_sum),
                'count': count,
                'message': f'Sum of {count} matched bank transactions: {format(total_sum, ".3f")}'
            }
            
            # Include detailed transaction list if requested
            detail = request.query_params.get('detail', '').lower() == 'true'
            if detail:
                transactions = []
                for tx in matched_bank_transactions.select_related('bank').order_by('-id')[:100]:  # Limit to 100 for performance
                    # Count how many customer transactions are matched to this bank transaction
                    customer_count = tx.matched_reco_customer_transactions.count()
                    
                    transactions.append({
                        'id': tx.id,
                        'bank_id': tx.bank_id,
                        'bank_name': tx.bank.name if tx.bank else None,
                        'operation_date': tx.operation_date.isoformat() if tx.operation_date else None,
                        'label': tx.label,
                        'amount': format(tx.amount, '.3f'),
                        'type': tx.type,
                        'internal_number': tx.internal_number,
                        'matched_customer_count': customer_count,
                    })
                
                response_data['transactions'] = transactions
                response_data['transactions_returned'] = len(transactions)
                if count > 100:
                    response_data['message'] += f' (showing first 100 of {count} transactions)'
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"BTSumMatchedBankTransactionsView.get error: {str(e)}\n{error_details}")
            return Response({
                'error': str(e),
                'details': error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BTAssignInternalNumberView(APIView):
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
                f"BTAssignInternalNumberView: Assigned internal_number '{internal_number}' to {updated_count} transactions. "
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
            logger.error(f"BTAssignInternalNumberView error: {str(e)}\n{error_details}")
            return Response({
                'error': str(e),
                'details': error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BTTaxManagementView(APIView):
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


class BTMatchTaxView(APIView):
    """
    View to match bank transactions with tax descriptions and update them with dynamic tax amounts.
    Filters by bank and company to ensure correct matching.
    """
    
    def post(self, request):
        try:
            print("=" * 80)
            print("BTMatchTaxView: Starting tax matching")
            print("=" * 80)
            logger.info("=" * 80)
            logger.info("BTMatchTaxView: Starting tax matching")
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
                        
                        if not transaction.label:
                            no_label_count += 1
                            if idx <= 5:  # Log first 5 for debugging
                                logger.debug(f"Transaction {idx} (ID={transaction.id}): No label, skipping")
                            continue
                        
                        # Debug: Show what we're scanning
                        if idx <= 10:  # Show detailed scan for first 10 transactions
                            print(f"\n  [{idx}] 🔍 SCANNING Transaction ID={transaction.id}")
                            print(f"      Label: '{transaction.label}'")
                            print(f"      Label (lowercase): '{label}'")
                            print(f"      Testing against {taxes_count} tax(es)...")
                        
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
                                            desc_lower = desc.lower()
                                            # Escape the description for regex, but preserve word boundaries for matching
                                            # Use word boundaries, but also allow matching at start/end of string or with non-word chars
                                            escaped_desc = re.escape(desc_lower)
                                            # Pattern: match as whole word, or between digits, or at boundaries with digits
                                            pattern = r'\b' + escaped_desc + r'\b|(?<=\d)' + escaped_desc + r'(?=\d)|(?<=\d)' + escaped_desc + r'\b|\b' + escaped_desc + r'(?=\d)|^' + escaped_desc + r'\b|\b' + escaped_desc + r'$'
                                            
                                            # Debug: Show pattern being tested
                                            if idx <= 10:
                                                print(f"      → Testing Tax: '{tax.name}' | Description: '{desc}' | Pattern: {pattern}")
                                            
                                            match_result = re.search(pattern, label)
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
                                                
                                                if idx <= 10:
                                                        print(f"      ✓ MATCH FOUND! Matched text: '{matched_text}' (position {matched_span[0]}-{matched_span[1]}) | Description length: {len(desc)}")
                                                
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
                                if idx <= 10:  # Log first 10 matches
                                    print(f"  [{idx}] ✅ FINAL MATCH: Transaction ID={transaction.id}")
                                    print(f"      Label: '{transaction.label[:80]}'")
                                    print(f"      Tax: '{matched_tax.name}'")
                                    print(f"      Matched Description: '{matched_description}'")
                                    print(f"      Pattern Used: {matched_pattern}")
                                    print(f"      Type: {old_type} → {transaction.type}")
                                    print(f"      {'─' * 70}")
                                logger.info(f"Matched transaction {transaction.id}: '{transaction.label[:50]}' → tax '{matched_tax.name}' (matched on '{matched_description}' with pattern '{matched_pattern}')")
                            except Exception as e:
                                error_count += 1
                                logger.error(f"Error saving transaction {transaction.id} after match: {str(e)}")
                                print(f"  [{idx}] ❌ Error saving: Transaction ID={transaction.id} | Error: {str(e)}")
                        else:
                            skipped_count += 1
                            if idx <= 10:  # Only log first 10 skipped for brevity
                                print(f"  [{idx}] ✗ NO MATCH: Transaction ID={transaction.id}")
                                print(f"      Label: '{transaction.label[:80]}'")
                                print(f"      Result: No matching tax description found after scanning all {taxes_count} tax(es)")
                                print(f"      {'─' * 70}")
                                logger.debug(f"Skipped transaction {transaction.id}: '{transaction.label[:50]}' - no matching tax")
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
            print("BTMatchTaxView: Final Statistics")
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
            logger.info("BTMatchTaxView: Final Statistics")
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
            print(f"❌ CRITICAL ERROR in BTMatchTaxView")
            print(f"{'=' * 80}")
            print(f"Error: {error_msg}")
            print(f"\nTraceback:\n{error_details}")
            print(f"{'=' * 80}")
            logger.error("=" * 80)
            logger.error("BTMatchTaxView: CRITICAL ERROR")
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


class BTUnmatchedTransactionsView(APIView):
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


class BTSortedRecoBankTransactionsView(APIView):
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
            
            # Step 3: Create sorted result list
            # Strategy: Maintain date order, but when an origine is encountered,
            # insert ALL its relatives (same internal_number) immediately after it,
            # regardless of where they appear in the date-sorted list

            sorted_transactions = []
            processed_ids = set()
            
            # Track the original position of each transaction
            original_positions = {tx.id: idx + 1 for idx, tx in enumerate(transactions_list)}

            for idx, tx in enumerate(transactions_list):
                # Already handled
                if tx.id in processed_ids:
                    continue

                # Grouped transaction
                if tx.internal_number and tx.internal_number in groups:
                    group_data = groups[tx.internal_number]
                    origine_tx = group_data['origine']

                    # CASE 1: Origine → add origine at its original date-sorted position + ALL its relatives immediately after
                    if tx.type == 'origine' and origine_tx and origine_tx.id == tx.id:
                        # The origine should maintain its original position in the date-sorted list
                        # Count how many transactions (excluding this origine's relatives) should come before it
                        origine_original_pos = original_positions[tx.id]
                        relatives_ids = {r.id for r in group_data['relatives']}
                        
                        # Count transactions before this origine that are not its relatives
                        transactions_before = sum(1 for i in range(origine_original_pos - 1) 
                                                if transactions_list[i].id not in relatives_ids 
                                                and transactions_list[i].id not in processed_ids)
                        
                        # Ensure we've added all transactions that should come before the origine
                        # If we haven't, it means we're processing out of order (shouldn't happen)
                        # But we'll add the origine at the correct position by ensuring sorted_transactions
                        # has the right number of items before it
                        
                        # Since we iterate in order and only skip relatives, the current position
                        # in sorted_transactions should be correct for maintaining date order
                        sorted_transactions.append(tx)
                        processed_ids.add(tx.id)

                        # Add ALL relatives for this origine, regardless of their date position
                        # They will be sorted by operation_date within the group (already sorted in Step 2)
                        for relative in group_data['relatives']:
                            if relative.id not in processed_ids:
                                sorted_transactions.append(relative)
                                processed_ids.add(relative.id)

                    # CASE 2: Relative with an origine → skip it here
                    # (it will be added when we encounter its origine)
                    elif tx.type != 'origine' and origine_tx:
                        # Skip this relative - it will be added when we encounter its origine
                        # Don't mark as processed yet, let it be handled when origine is encountered
                        continue

                    # CASE 3: Orphaned relative → keep its date position
                    elif not origine_tx:
                        sorted_transactions.append(tx)
                        processed_ids.add(tx.id)

                # Ungrouped transaction
                else:
                    sorted_transactions.append(tx)
                    processed_ids.add(tx.id)
            
            # Step 4: Serialize original transactions for comparison
            original_serialized = []
            for idx, tx in enumerate(transactions_list):
                is_origine = tx.type == 'origine'
                group_id = tx.internal_number if tx.internal_number else None
                group_size = 0
                
                # Determine if transaction is matched (colored)
                is_matched = bool(tx.payment_class and tx.payment_status)
                
                # For non-origine transactions with a group: inherit color from their origine
                if not is_origine and group_id and group_id in groups:
                    group_data = groups[group_id]
                    origine_tx = group_data.get('origine')
                    if origine_tx:
                        # Check if the origine is matched to a RecoCustomerTransaction
                        origine_is_matched_to_customer = origine_tx.matched_reco_customer_transactions.exists()
                        should_be_colored = origine_is_matched_to_customer
                    else:
                        # If non-origine has no origine in group, we'll handle that later
                        should_be_colored = False
                else:
                    # Origine transactions or non-origine with no group: not colored
                    should_be_colored = False
                
                if group_id and group_id in groups:
                    group_data = groups[group_id]
                    if group_data['origine']:
                        group_size = 1 + len(group_data['relatives'])
                    else:
                        group_size = len(group_data['relatives'])
                
                original_serialized.append({
                    'id': tx.id,
                    'original_position': idx + 1,
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
                    'group_size': group_size,
                    'is_matched': is_matched,
                    'should_be_colored': should_be_colored
                })
            
            # Step 5: Serialize sorted transactions
            serialized_transactions = []
            grouped_count = 0
            ungrouped_count = 0
            processed_groups = set()  # Track which groups we've already counted
            
            for idx, tx in enumerate(sorted_transactions):
                is_origine = tx.type == 'origine'
                group_id = tx.internal_number if tx.internal_number else None
                group_size = 0
                
                # Determine if transaction is matched (colored)
                # A transaction is matched if it has payment_class and payment_status
                is_matched = bool(tx.payment_class and tx.payment_status)
                
                # For non-origine transactions with a group: inherit color from their origine
                if not is_origine and group_id and group_id in groups:
                    group_data = groups[group_id]
                    origine_tx = group_data.get('origine')
                    if origine_tx:
                        # Check if the origine is matched to a RecoCustomerTransaction
                        origine_is_matched_to_customer = origine_tx.matched_reco_customer_transactions.exists()
                        should_be_colored = origine_is_matched_to_customer
                    else:
                        # If non-origine has no origine in group, we'll handle that later
                        should_be_colored = False
                else:
                    # Origine transactions or non-origine with no group: not colored
                    should_be_colored = False
                
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
                    'sorted_position': idx + 1,
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
                    'group_size': group_size,
                    'is_matched': is_matched,
                    'should_be_colored': should_be_colored
                })
            
            return Response({
                "transactions": serialized_transactions,
                "original_transactions": original_serialized,
                "total": len(serialized_transactions),
                "original_total": len(original_serialized),
                "grouped_count": grouped_count,
                "ungrouped_count": ungrouped_count,
                "bank": bank.name,
                "bank_code": bank.code
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in BTSortedRecoBankTransactionsView: {str(e)}\n{error_details}")
            return Response({
                "error": str(e),
                "details": error_details
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


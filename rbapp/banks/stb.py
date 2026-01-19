# # === Helper functions (from views.py) ===
# import pandas as pd
# import numpy as np
# import re
# from datetime import datetime
# from rapidfuzz import fuzz

# # --- Matching and normalization helpers ---
# def score_date_tolerance(source_date_str, target_date_str):
#     try:
#         source_date = datetime.strptime(source_date_str, "%Y-%m-%d")
#         target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
#         day_diff = abs((source_date - target_date).days)
#         if day_diff == 0:
#             return 100
#         elif day_diff == 1:
#             return 95
#         elif day_diff == 2:
#             return 90
#         elif day_diff == 3:
#             return 85
#         elif day_diff <= 5:
#             return 75
#         elif day_diff <= 8:
#             return 60
#         else:
#             return 0
#     except Exception:
#         return 0

# def score_amount_exact(amount1, amount2):
#     try:
#         return 100 if float(amount1) == float(amount2) else 0
#     except Exception:
#         return 0

# def score_reference_fuzzy(ref1, ref2):
#     try:
#         if pd.isna(ref1) or pd.isna(ref2):
#             return 0
#         ref1_str = str(ref1)
#         ref2_str = str(ref2)
#         ref1_digits = re.sub(r'\D', '', ref1_str)
#         ref2_digits = re.sub(r'\D', '', ref2_str)
#         len_ref1 = len(ref1_digits)
#         min_match_len = int(len_ref1 * 0.7) if len_ref1 else 0
#         score_substring = 0
#         if len_ref1 and len(ref2_digits) >= min_match_len:
#             for i in range(len(ref2_digits) - min_match_len + 1):
#                 sub = ref2_digits[i:i + min_match_len]
#                 if sub in ref1_digits:
#                     score_substring = 100
#                     break
#         score_fuzzy = fuzz.partial_ratio(ref1_str, ref2_str)
#         return max(score_substring, score_fuzzy)
#     except Exception:
#         return 0

# def score_payment_type_fuzzy(payment_type1, payment_type2):
#     try:
#         if pd.isna(payment_type1) or pd.isna(payment_type2):
#             return 0
#         payment_type1_str = str(payment_type1).strip()
#         payment_type2_str = str(payment_type2).strip()
#         if not payment_type1_str or not payment_type2_str:
#             return 0
#         return fuzz.token_sort_ratio(payment_type1_str, payment_type2_str)
#     except Exception:
#         return 0

# def score_document_number_fuzzy(doc_num1, doc_num2):
#     try:
#         if pd.isna(doc_num1) or pd.isna(doc_num2):
#             return 0
#         doc_num1_str = str(doc_num1).strip()
#         doc_num2_str = str(doc_num2).strip()
#         if not doc_num1_str or not doc_num2_str:
#             return 0
#         if doc_num1_str == doc_num2_str:
#             return 100
#         return fuzz.partial_ratio(doc_num1_str, doc_num2_str)
#     except Exception:
#         return 0

# def clean_bank_dataframe(df):
#     column_mapping = {
#         'Date Opération': ['Opération', 'Date opération', 'Date', 'DATE OPERATION'],
#         'Libellé': ['Intitulé', "Libellé de l'opération", 'Libellé Opération', 'LIBELLE'],
#         'Date Valeur': ['Valeur', 'Date de valeur', 'Date valeur', 'DATE VALEUR'],
#         'Débit': ['DEBIT (TND)', 'Débit', 'Débit (TND)'],
#         'Crédit': ['CREDIT (TND)', 'Crédit', 'Crédit (TND)'],
#         'Montant': ['Montant', 'MONTANT', 'Amount'],
#         'Sens Opération': ['Sens', 'Sens Opération', 'SENS OPERATION'],
#         'Référence': ['Référence', 'Reference', 'REFERENCE'],
#         'Référence Dossier': ['REFERENCE DOSSIER']
#     }
#     reverse_mapping = {}
#     for standard_col, aliases in column_mapping.items():
#         for alias in aliases:
#             reverse_mapping[alias.strip().lower()] = standard_col
#     header_row_index = None
#     for i in range(len(df)):
#         row = df.iloc[i]
#         normalized = [str(cell).strip().lower() for cell in row]
#         match_count = sum(1 for cell in normalized if cell in reverse_mapping)
#         if match_count >= 2:
#             header_row_index = i
#             break
#     if header_row_index is None:
#         raise ValueError("Aucune ligne contenant des noms de colonnes valides n'a été trouvée.")
#     df = pd.DataFrame(df.values[header_row_index + 1:], columns=df.iloc[header_row_index])
#     df = df.dropna(axis=1, how='all')
#     df = df.rename(columns={
#         col: reverse_mapping[str(col).strip().lower()]
#         for col in df.columns
#         if str(col).strip().lower() in reverse_mapping
#     })
#     if 'Montant' in df.columns:
#         df['Montant'] = pd.to_numeric(df['Montant'], errors='coerce')
#     if 'Sens Opération' in df.columns and 'Montant' in df.columns:
#         df['Débit'] = np.where(df['Sens Opération'].str.upper() == 'D', df['Montant'], 0)
#         df['Crédit'] = np.where(df['Sens Opération'].str.upper() == 'C', df['Montant'], 0)
#     desired_columns = [
#         'Date Opération', 'Libellé', 'Date Valeur',
#         'Débit', 'Crédit', 'Référence', 'Référence Dossier'
#     ]
#     df = df[[col for col in desired_columns if col in df.columns]]
#     return df.reset_index(drop=True)

# def normalize_amount_column(series: pd.Series) -> pd.Series:
#     try:
#         series = series.fillna('')
#         cleaned = series.astype(str)
#         cleaned = cleaned.str.replace(r'\s+', '', regex=True)
#         cleaned = cleaned.str.replace(',', '.', regex=False)
#         return pd.to_numeric(cleaned, errors='coerce').round(3)
#     except Exception as e:
#         try:
#             return pd.to_numeric(series, errors='coerce').round(3)
#         except Exception as e2:
#             return pd.Series([0.0] * len(series))

# def extract_largest_number(text):
#     if pd.isna(text):
#         return None
#     matches = re.findall(r'\d{5,}', text)
#     if not matches:
#         return None
#     return max(matches, key=len)

# def is_valid_date(d, m):
#     try:
#         day, month = int(d), int(m)
#         return 1 <= day <= 31 and 1 <= month <= 12
#     except:
#         return False

# def extract_info(libelle):
#     libelle = str(libelle)
#     date_ref = None
#     used_date_number = None
#     m3 = re.search(r'(\d{2})\s+(\d{2})\s+(20\d{2})', libelle)
#     if m3:
#         d, m, y = m3.groups()
#         if is_valid_date(d, m):
#             date_ref = f"{d}/{m}/{y}"
#             used_date_number = m3.group(0)
#     ref = None
#     if used_date_number:
#         ref = libelle.replace(used_date_number, '').strip()
#     else:
#         ref = libelle
#     return pd.Series([date_ref, ref])

# def clean_customer_accounting_dataframe(df_comptable):
#     columns_to_keep = [
#         'N° compte bancaire',
#         'Date comptabilisation',
#         'N° document',
#         'Description',
#         'Montant débit',
#         'Montant crédit',
#         'N° doc. externe',
#         "Date d'échéance",
#         'Type de règlement'
#     ]
#     columns_to_keep_clean = [col.strip() for col in columns_to_keep]
#     header_row_index = None
#     for i, row in df_comptable.iterrows():
#         row_cleaned = [str(cell).strip() for cell in row]
#         match_count = sum(1 for cell in row_cleaned if cell in columns_to_keep_clean)
#         if match_count >= 3:
#             header_row_index = i
#             break
#     if header_row_index is None:
#         raise ValueError("Expected columns not found in the file.")
#     df = pd.DataFrame(df_comptable.values[header_row_index + 1:], columns=df_comptable.iloc[header_row_index])
#     df.columns = [str(col).strip() for col in df.columns]
#     available_columns = [col for col in columns_to_keep_clean if col in df.columns]
#     df = df[available_columns]
#     return df.reset_index(drop=True)

# # --- PreprocessBankLedgerEntryView (post) ---
# def preprocess_bank_ledger_entry(ledger_entry, file_path):
#     """
#     Processes a bank ledger Excel file, cleans and normalizes the data, and returns a list of transaction dicts.
#     Args:
#         ledger_entry: The BankLedgerEntry instance (caller must provide)
#         file_path: Path to the uploaded Excel file
#     Returns:
#         dict: {'transactions': [...], 'count': int} or {'error': str}
#     """
#     import os
#     import pandas as pd
#     try:
#         ext = os.path.splitext(file_path)[1].lower()
#         if ext == '.xlsx':
#             df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
#         elif ext == '.xls':
#             df_raw = pd.read_excel(file_path, header=None, engine='xlrd')
#         else:
#             return {'error': 'Unsupported file extension'}
#         df_cleaned = clean_bank_dataframe(df_raw)
#         if 'Libellé' in df_cleaned.columns:
#             df_cleaned['Libellé'] = df_cleaned['Libellé'].astype(str)
#         target_columns = ['Date Opération', 'Libellé', 'Date Valeur', 'Débit', 'Crédit']
#         df_cleaned = df_cleaned[[col for col in target_columns if col in df_cleaned.columns]]
#         if 'Crédit' in df_cleaned.columns:
#             df_cleaned['Crédit'] = normalize_amount_column(df_cleaned['Crédit'])
#         if 'Débit' in df_cleaned.columns:
#             df_cleaned['Débit'] = normalize_amount_column(df_cleaned['Débit'])
#         if 'Date Opération' in df_cleaned.columns:
#             df_cleaned['Date Opération'] = pd.to_datetime(df_cleaned['Date Opération'], dayfirst=True, errors='coerce')
#         if 'Date Valeur' in df_cleaned.columns:
#             df_cleaned['Date Valeur'] = pd.to_datetime(df_cleaned['Date Valeur'], dayfirst=True, errors='coerce')
#         df_cleaned['Crédit'] = df_cleaned['Crédit'].fillna(0)
#         df_cleaned['Débit'] = df_cleaned['Débit'].fillna(0)
#         if 'Libellé' in df_cleaned.columns:
#             extracted = df_cleaned['Libellé'].apply(extract_info)
#             extracted.columns = ['date_ref', 'ref']
#             df_cleaned = pd.concat([df_cleaned, extracted], axis=1)
#         transactions = []
#         for _, row in df_cleaned.iterrows():
#             credit = row.get('Crédit', 0) or 0
#             debit = row.get('Débit', 0) or 0
#             amount = credit - debit
#             transactions.append({
#                 'bank_ledger_entry': ledger_entry,
#                 'operation_date': row.get('Date Opération'),
#                 'label': row.get('Libellé', ''),
#                 'value_date': row.get('Date Valeur'),
#                 'debit': debit if debit != 0 else None,
#                 'credit': credit if credit != 0 else None,
#                 'date_ref': row.get('date_ref'),
#                 'ref': row.get('ref'),
#                 'document_reference': '',
#                 'amount': amount
#             })
#         return {'transactions': transactions, 'count': len(transactions)}
#     except Exception as e:
#         return {'error': str(e)}

# # --- PreprocessCustomerLedgerEntryView (post) ---
# def preprocess_customer_ledger_entry(ledger_entry, file_path, agency_code=None):
#     """
#     Processes a customer ledger Excel file, cleans and normalizes the data, and returns a list of transaction dicts.
#     Args:
#         ledger_entry: The CustomerLedgerEntry instance (caller must provide)
#         file_path: Path to the uploaded Excel file
#         agency_code: Optional agency code to filter rows
#     Returns:
#         dict: {'transactions': [...], 'count': int, ...} or {'error': str}
#     """
#     import os
#     import pandas as pd
#     try:
#         ext = os.path.splitext(file_path)[1].lower()
#         if ext == '.xlsx':
#             df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
#         elif ext == '.xls':
#             df_raw = pd.read_excel(file_path, header=None, engine='xlrd')
#         else:
#             return {'error': 'Unsupported file extension'}
#         df_cleaned = clean_customer_accounting_dataframe(df_raw)
#         if not agency_code:
#             agency_code = '53-5'
#         if 'N° compte bancaire' in df_cleaned.columns:
#             df_filtered = df_cleaned[df_cleaned['N° compte bancaire'] == agency_code].copy()
#             if df_filtered.empty:
#                 return {'error': f"No transactions found for agency code '{agency_code}'", 'agency_code': agency_code, 'total_rows_before_filter': len(df_cleaned)}
#             df_cleaned = df_filtered
#         else:
#             return {'error': "Column 'N° compte bancaire' not found in the file"}
#         df_final = df_cleaned.rename(columns={
#             'Montant débit': 'Crédit',
#             'Montant crédit': 'Débit'
#         })
#         if 'Crédit' in df_final.columns:
#             df_final['Crédit'] = normalize_amount_column(df_final['Crédit'])
#         if 'Débit' in df_final.columns:
#             df_final['Débit'] = normalize_amount_column(df_final['Débit'])
#         if 'Date comptabilisation' in df_final.columns:
#             df_final['Date comptabilisation'] = pd.to_datetime(df_final['Date comptabilisation'], dayfirst=True, errors='coerce').dt.date
#         if "Date d'échéance" in df_final.columns:
#             df_final["Date d'échéance"] = pd.to_datetime(df_final["Date d'échéance"], dayfirst=True, errors='coerce').dt.date
#         df_final['Crédit'] = df_final['Crédit'].fillna(0)
#         df_final['Débit'] = df_final['Débit'].fillna(0)
#         transactions = []
#         for _, row in df_final.iterrows():
#             debit = row.get('Débit', 0) or 0
#             credit = row.get('Crédit', 0) or 0
#             if debit > 0:
#                 amount = -debit
#             elif credit > 0:
#                 amount = credit
#             else:
#                 amount = 0
#             accounting_date = row.get('Date comptabilisation')
#             if pd.isna(accounting_date):
#                 accounting_date = None
#             due_date = row.get("Date d'échéance")
#             if pd.isna(due_date):
#                 due_date = None
#             payment_type = row.get('Type de règlement', '')
#             if payment_type:
#                 payment_type = str(payment_type)
#                 payment_type = re.sub(r'^\d{2}-\s*', '', payment_type)
#                 payment_type = re.sub(r'\b(TUNIS|ZARZIS)\b', '', payment_type, flags=re.IGNORECASE)
#                 payment_type = re.sub(r'\bTRAITES\b', 'EFFETS', payment_type, flags=re.IGNORECASE)
#                 payment_type = re.sub(r'\bEFFETS CLT\b', 'CLT EFFET', payment_type, flags=re.IGNORECASE)
#                 payment_type = re.sub(r'\s+', ' ', payment_type).strip()
#             transactions.append({
#                 'customer_ledger_entry': ledger_entry,
#                 'account_number': str(row.get('N° compte bancaire', '')),
#                 'accounting_date': accounting_date,
#                 'document_number': str(row.get('N° document', '')),
#                 'description': str(row.get('Description', '')),
#                 'debit_amount': debit if debit != 0 else 0,
#                 'credit_amount': credit if credit != 0 else 0,
#                 'external_doc_number': str(row.get('N° doc. externe', '')),
#                 'due_date': due_date,
#                 'payment_type': payment_type,
#                 'amount': amount
#             })
#         return {'transactions': transactions, 'count': len(transactions), 'agency_code': agency_code}
#     except Exception as e:
#         return {'error': str(e)}

# # --- GetMatchingResultsView (get) ---
# def get_matching_results(output_dir):
#     """
#     Retrieves the results of the latest bank-customer transaction matching process from saved CSV files.
#     Args:
#         output_dir: Directory where matching result CSVs are stored
#     Returns:
#         dict: { 'results': {...}, 'available_files': [...], ... } or {'error': str}
#     """
#     import os
#     import pandas as pd
#     try:
#         if not os.path.exists(output_dir):
#             return {'error': 'No matching results found. Run the matching endpoint first.'}
#         csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
#         if not csv_files:
#             return {'error': 'No matching result files found.'}
#         csv_files.sort(reverse=True)
#         latest_files = {}
#         for file in csv_files:
#             if 'all_matches' in file:
#                 latest_files['all_matches'] = file
#             elif 'high_matches' in file:
#                 latest_files['high_matches'] = file
#             elif 'low_matches' in file:
#                 latest_files['low_matches'] = file
#         results = {}
#         for file_type, filename in latest_files.items():
#             file_path = os.path.join(output_dir, filename)
#             try:
#                 df = pd.read_csv(file_path)
#                 results[file_type] = {
#                     'filename': filename,
#                     'file_path': file_path,
#                     'shape': df.shape,
#                     'columns': list(df.columns),
#                     'data': df.to_dict('records')
#                 }
#             except Exception as e:
#                 results[file_type] = {'error': f'Could not read {filename}: {str(e)}'}
#         return {
#             'message': 'Retrieved saved matching results',
#             'output_directory': output_dir,
#             'available_files': csv_files,
#             'results': results
#         }
#     except Exception as e:
#         return {'error': str(e)}

# # --- MatchCustomerBankTransactionsView (post) ---
# def match_customer_bank_transactions(filtered_bank_transactions, customer_transactions, output_dir):
#     """
#     Performs matching between customer and bank transactions based on several criteria, saves results, and returns summary.
#     Args:
#         filtered_bank_transactions: List of bank transaction dicts (with required fields)
#         customer_transactions: List of customer transaction dicts (with required fields)
#         output_dir: Directory to save result CSVs
#     Returns:
#         dict: { 'summary': {...}, 'saved_files': {...}, ... } or {'error': str}
#     """
#     import os
#     import pandas as pd
#     from datetime import datetime
#     try:
#         df_bank = pd.DataFrame(filtered_bank_transactions)
#         df_customer = pd.DataFrame(customer_transactions)
#         matches = []
#         for i, bank_row in df_bank.iterrows():
#             best_score = -1
#             best_match_data = None
#             for j, customer_row in df_customer.iterrows():
#                 bank_payment_class = bank_row['payment_class_id']
#                 customer_payment_type = customer_row['payment_type']
#                 if bank_payment_class != customer_payment_type:
#                     continue
#                 score_date = score_date_tolerance(customer_row['accounting_date'], bank_row['operation_date'])
#                 score_amount_regular = score_amount_exact(customer_row['amount'], bank_row['amount'])
#                 if score_amount_regular == 0:
#                     score_amount = score_amount_exact(customer_row['total_amount'], bank_row['amount'])
#                 else:
#                     score_amount = score_amount_regular
#                 score_reference = score_reference_fuzzy(customer_row['external_doc_number'], bank_row['ref'])
#                 score_payment_type = score_payment_type_fuzzy(customer_row['payment_type'], bank_row['payment_class_id'])
#                 total_score = (
#                     0.25 * score_date +
#                     0.6 * score_amount +
#                     0.15 * score_reference
#                 )
#                 if total_score > best_score:
#                     best_score = total_score
#                     best_match_data = customer_row.to_dict()
#             matches.append({
#                 'bank_transaction_id': bank_row['id'],
#                 'bank_operation_date': bank_row['operation_date'],
#                 'bank_amount': bank_row['amount'],
#                 'bank_ref': bank_row['ref'],
#                 'bank_payment_class': bank_row['payment_class_id'],
#                 'bank_label': bank_row['label'],
#                 'customer_transaction_id': best_match_data['id'] if best_match_data else None,
#                 'customer_accounting_date': best_match_data['accounting_date'] if best_match_data else None,
#                 'customer_amount': best_match_data['amount'] if best_match_data else None,
#                 'customer_total_amount': best_match_data['total_amount'] if best_match_data else None,
#                 'customer_document_number': best_match_data['document_number'] if best_match_data else None,
#                 'customer_external_doc_number': best_match_data['external_doc_number'] if best_match_data else None,
#                 'customer_payment_type': best_match_data['payment_type'] if best_match_data else None,
#                 'customer_description': best_match_data['description'] if best_match_data else None,
#                 'score': best_score
#             })
#         df_matches = pd.DataFrame(matches)
#         df_matches_high = df_matches[df_matches['score'] >= 68].copy()
#         df_matches_low = df_matches[df_matches['score'] < 68].copy()
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         os.makedirs(output_dir, exist_ok=True)
#         high_matches_file = os.path.join(output_dir, f'high_matches_{timestamp}.csv')
#         low_matches_file = os.path.join(output_dir, f'low_matches_{timestamp}.csv')
#         all_matches_file = os.path.join(output_dir, f'all_matches_{timestamp}.csv')
#         if not df_matches_high.empty:
#             df_matches_high.to_csv(high_matches_file, index=False)
#         if not df_matches_low.empty:
#             df_matches_low.to_csv(low_matches_file, index=False)
#         df_matches.to_csv(all_matches_file, index=False)
#         return {
#             'summary': {
#                 'total_bank_transactions': len(df_bank),
#                 'high_matches_count': len(df_matches_high),
#                 'high_matches_percentage': round(100 * len(df_matches_high) / len(df_bank), 2) if len(df_bank) > 0 else 0,
#                 'low_matches_count': len(df_matches_low),
#                 'low_matches_percentage': round(100 * len(df_matches_low) / len(df_bank), 2) if len(df_bank) > 0 else 0
#             },
#             'saved_files': {
#                 'all_matches_csv': all_matches_file,
#                 'high_matches_csv': high_matches_file if not df_matches_high.empty else None,
#                 'low_matches_csv': low_matches_file if not df_matches_low.empty else None,
#                 'output_directory': output_dir
#             },
#             'dataframe_info': {
#                 'high_matches_shape': df_matches_high.shape if not df_matches_high.empty else (0, 0),
#                 'low_matches_shape': df_matches_low.shape if not df_matches_low.empty else (0, 0),
#                 'all_matches_shape': df_matches.shape
#             },
#             'high_matches': df_matches_high.to_dict('records') if not df_matches_high.empty else [],
#             'low_matches': df_matches_low.to_dict('records') if not df_matches_low.empty else []
#         }
#     except Exception as e:
#         return {'error': str(e)}

# # --- MatchTransactionView (post) ---
# def match_transactions(bank_transactions, payment_identifications):
#     """
#     Matches bank transactions to payment identifications and updates their status and class accordingly.
#     Args:
#         bank_transactions: List of bank transaction dicts (with required fields)
#         payment_identifications: List of payment identification dicts (with required fields)
#     Returns:
#         dict: { 'updated_count': int, ... } or {'error': str}
#     """
#     try:
#         updated_count = 0
#         for transaction in bank_transactions:
#             amount = float(transaction['amount'])
#             label = transaction['label']
#             best_match = None
#             for payment_id in payment_identifications:
#                 description = payment_id['description']
#                 debit = payment_id['debit']
#                 credit = payment_id['credit']
#                 if description.lower() not in label.lower():
#                     continue
#                 amount_matches = False
#                 if amount < 0 and debit:
#                     amount_matches = True
#                 elif amount > 0 and credit:
#                     amount_matches = True
#                 if amount_matches:
#                     best_match = payment_id
#                     break
#             if best_match:
#                 transaction['payment_class'] = best_match['payment_status']['payment_class']
#                 transaction['payment_status'] = best_match['payment_status']
#                 transaction['type'] = 'origine'
#                 updated_count += 1
#         return {
#             'updated_count': updated_count,
#             'total_transactions_processed': len(bank_transactions),
#             'payment_identifications_checked': len(payment_identifications),
#             'matching_rules_applied': {
#                 'description_match': 'description_in_label',
#                 'amount_logic': 'negative_amount_with_debit_true_or_positive_amount_with_credit_true'
#             }
#         }
#     except Exception as e:
#         return {'error': str(e)}

# # --- MatchBankTransactionTaxesView (post) ---
# def match_bank_transaction_taxes(all_transactions, origine_transactions, non_origine_transactions):
#     """
#     Matches non-origin bank transactions to origin transactions by reference or date, assigns internal numbers, and ensures all origin transactions have internal numbers.
#     Args:
#         all_transactions: List of all bank transaction dicts
#         origine_transactions: List of origin bank transaction dicts
#         non_origine_transactions: List of non-origin bank transaction dicts
#     Returns:
#         dict: { 'matched_count': int, 'unmatched_count': int, ... } or {'error': str}
#     """
#     import uuid
#     try:
#         matched_count = 0
#         unmatched_count = 0
#         for non_origine in non_origine_transactions:
#             matched = False
#             if non_origine.get('ref'):
#                 matching_origine = next((o for o in origine_transactions if o.get('ref') == non_origine['ref']), None)
#                 if matching_origine:
#                     if matching_origine.get('internal_number'):
#                         non_origine['internal_number'] = matching_origine['internal_number']
#                     else:
#                         new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
#                         matching_origine['internal_number'] = new_internal_number
#                         non_origine['internal_number'] = new_internal_number
#                     matched = True
#                     matched_count += 1
#             if not matched and non_origine.get('date_ref'):
#                 matching_origine = next((o for o in origine_transactions if o.get('date_ref') == non_origine['date_ref']), None)
#                 if matching_origine:
#                     if matching_origine.get('internal_number'):
#                         non_origine['internal_number'] = matching_origine['internal_number']
#                     else:
#                         new_internal_number = f"INT_{uuid.uuid4().hex[:8].upper()}"
#                         matching_origine['internal_number'] = new_internal_number
#                         non_origine['internal_number'] = new_internal_number
#                     matched = True
#                     matched_count += 1
#             if not matched:
#                 non_origine['internal_number'] = f"INT_{uuid.uuid4().hex[:8].upper()}"
#                 unmatched_count += 1
#         for origine in origine_transactions:
#             if not origine.get('internal_number'):
#                 origine['internal_number'] = f"INT_{uuid.uuid4().hex[:8].upper()}"
#         return {
#             'matched_count': matched_count,
#             'unmatched_count': unmatched_count,
#             'total_transactions': len(all_transactions),
#             'origine_transactions': len(origine_transactions),
#             'non_origine_transactions': len(non_origine_transactions),
#             'total_with_internal_number': sum(1 for t in all_transactions if t.get('internal_number'))
#         }
#     except Exception as e:
#         return {'error': str(e)}

# # === Bank-specific logic as standalone functions ===
# # (To be filled in next step: move the logic from the views, as pure functions)

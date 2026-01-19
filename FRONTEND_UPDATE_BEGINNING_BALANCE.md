# Frontend Update: Beginning Balance Extraction

## Update Summary

The beginning balance extraction API now uses a configurable label parameter per bank instead of a hardcoded value. Each bank must have a `beginning_balance_label` field configured in the admin panel (e.g., "SOLDE DEBUT PERIODE") before the extraction endpoint can work. When calling the API, it will automatically use the bank's configured label to search for the beginning balance transaction in the ledger entries. If a bank doesn't have this label configured, the API will return a 400 error with a message indicating that the bank needs to be configured in the parameters page. The API endpoint remains the same (`GET /api/bank-ledger-entries/{id}/extract-beginning-balance/`), but now requires proper bank configuration beforehand. Frontend should handle the 400 error case by displaying a user-friendly message directing users to configure the bank's beginning balance label in the admin panel.








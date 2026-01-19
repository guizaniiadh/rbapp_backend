# Extract Beginning Balance API - Quick Reference

## Endpoint
```
GET /api/bank-ledger-entries/{bank_ledger_entry_id}/extract-beginning-balance/
```

**Alternative (Legacy):**
```
GET /api/extract-beginning-balance/?bank_ledger_entry_id={id}
```

## Authentication
- **Required**: JWT Bearer Token
- **Header**: `Authorization: Bearer {token}`

## Request
- **Method**: GET
- **Path Parameter**: `bank_ledger_entry_id` (integer, optional - uses latest if omitted)

## Success Response (200 OK)
```json
{
  "beginning_balance": 15000.50,
  "transaction_id": 456,
  "transaction_label": "SOLDE DEBUT PERIODE 01/01/2025",
  "operation_date": "2025-01-01",
  "bank_ledger_entry_id": 123,
  "bank_code": "BT",
  "bank_name": "Banque de Tunisie",
  "search_label_used": "SOLDE DEBUT PERIODE",
  "source": "debit"
}
```

## Error Responses

| Status | Error | Description |
|--------|-------|-------------|
| 400 | Beginning balance label not configured | Bank missing `beginning_balance_label` configuration |
| 404 | Beginning balance transaction not found | No transaction matches the search label |
| 404 | Beginning balance value not found | Transaction found but debit/amount are null/zero |
| 404 | Bank ledger entry not found | Invalid `bank_ledger_entry_id` |
| 500 | Server error | Unexpected server error |

## Quick Example (JavaScript)
```javascript
const response = await fetch(
  `/api/bank-ledger-entries/${ledgerEntryId}/extract-beginning-balance/`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);
const data = await response.json();
console.log('Beginning Balance:', data.beginning_balance);
```

## Important Notes
- Bank must have `beginning_balance_label` configured (e.g., "SOLDE DEBUT PERIODE")
- Search is case-insensitive and uses "contains" matching
- Extracts from `debit` column first, falls back to `amount` if needed
- If no `bank_ledger_entry_id` provided, uses the latest entry



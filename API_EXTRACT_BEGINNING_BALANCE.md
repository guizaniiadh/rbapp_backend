# Extract Beginning Balance API Documentation

## Overview
This API endpoint extracts the beginning balance from a bank ledger entry by searching for a transaction with a label that contains the bank's configured beginning balance label text (e.g., "SOLDE DEBUT PERIODE"). The balance is extracted from the debit column of the matching transaction.

---

## Endpoint Details

### **Primary Endpoint (Recommended)**
```
GET /api/bank-ledger-entries/{bank_ledger_entry_id}/extract-beginning-balance/
```

### **Legacy Endpoint (Backward Compatible)**
```
GET /api/extract-beginning-balance/?bank_ledger_entry_id={bank_ledger_entry_id}
```

---

## Authentication
- **Required**: Yes
- **Type**: JWT Token (Bearer Token)
- **Header**: `Authorization: Bearer {token}`

---

## Request Parameters

### Path Parameters (Primary Endpoint)
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_ledger_entry_id` | integer | Yes* | The ID of the bank ledger entry to extract balance from |

*If not provided in the primary endpoint, the API will use the latest bank ledger entry.

### Query Parameters (Legacy Endpoint)
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_ledger_entry_id` | integer | No | The ID of the bank ledger entry. If omitted, uses the latest entry |

---

## Request Examples

### Example 1: Using Primary Endpoint
```http
GET /api/bank-ledger-entries/123/extract-beginning-balance/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

### Example 2: Using Legacy Endpoint with Query Parameter
```http
GET /api/extract-beginning-balance/?bank_ledger_entry_id=123
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

### Example 3: Using Latest Entry (Legacy Endpoint)
```http
GET /api/extract-beginning-balance/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

---

## Success Response

### Status Code: `200 OK`

### Response Body
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

### Response Fields
| Field | Type | Description |
|-------|------|-------------|
| `beginning_balance` | float | The extracted beginning balance amount |
| `transaction_id` | integer | ID of the transaction that contains the beginning balance |
| `transaction_label` | string | The full label text of the matching transaction |
| `operation_date` | string (ISO format) | Date of the operation (YYYY-MM-DD) or null |
| `bank_ledger_entry_id` | integer | ID of the bank ledger entry used |
| `bank_code` | string | Code of the bank |
| `bank_name` | string | Name of the bank |
| `search_label_used` | string | The label text that was searched for (from bank configuration) |
| `source` | string | Source of the balance: "debit" or "amount" |

---

## Error Responses

### 1. Beginning Balance Label Not Configured
**Status Code**: `400 Bad Request`

```json
{
  "error": "Beginning balance label not configured",
  "message": "Please configure 'beginning_balance_label' for bank 'Banque de Tunisie' in the parameters page",
  "bank_code": "BT",
  "bank_name": "Banque de Tunisie"
}
```

**Cause**: The bank associated with the ledger entry doesn't have a `beginning_balance_label` configured in the system.

**Solution**: Configure the `beginning_balance_label` field for the bank in the admin/parameters page.

---

### 2. Beginning Balance Transaction Not Found
**Status Code**: `404 Not Found`

```json
{
  "error": "Beginning balance transaction not found",
  "message": "No transaction with label containing 'SOLDE DEBUT PERIODE' found",
  "bank_ledger_entry_id": 123,
  "search_label": "SOLDE DEBUT PERIODE"
}
```

**Cause**: No transaction in the ledger entry has a label containing the configured search text.

**Solution**: Verify that the bank ledger entry contains a transaction with a label matching the configured `beginning_balance_label`.

---

### 3. Beginning Balance Value Not Found
**Status Code**: `404 Not Found`

```json
{
  "error": "Beginning balance value not found",
  "message": "Transaction found but debit and amount are both null/zero"
}
```

**Cause**: A matching transaction was found, but both the debit and amount fields are null or zero.

**Solution**: Verify the transaction data in the bank ledger entry.

---

### 4. Bank Ledger Entry Not Found
**Status Code**: `404 Not Found`

```json
{
  "error": "Bank ledger entry not found"
}
```

**Cause**: The provided `bank_ledger_entry_id` doesn't exist.

**Solution**: Verify the bank ledger entry ID is correct.

---

### 5. No Bank Ledger Entries Found
**Status Code**: `404 Not Found`

```json
{
  "error": "No bank ledger entries found"
}
```

**Cause**: No bank ledger entries exist in the system (when trying to use the latest entry).

**Solution**: Upload a bank ledger entry first.

---

### 6. Server Error
**Status Code**: `500 Internal Server Error`

```json
{
  "error": "Error message",
  "details": "Full traceback details..."
}
```

**Cause**: An unexpected server error occurred.

---

## Frontend Implementation Examples

### JavaScript/TypeScript (Fetch API)
```typescript
async function extractBeginningBalance(bankLedgerEntryId: number): Promise<any> {
  const token = localStorage.getItem('authToken'); // Your token storage method
  
  try {
    const response = await fetch(
      `/api/bank-ledger-entries/${bankLedgerEntryId}/extract-beginning-balance/`,
      {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || error.error);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error extracting beginning balance:', error);
    throw error;
  }
}

// Usage
extractBeginningBalance(123)
  .then(data => {
    console.log('Beginning Balance:', data.beginning_balance);
  })
  .catch(error => {
    console.error('Failed:', error);
  });
```

### Axios Example
```typescript
import axios from 'axios';

async function extractBeginningBalance(bankLedgerEntryId: number) {
  try {
    const response = await axios.get(
      `/api/bank-ledger-entries/${bankLedgerEntryId}/extract-beginning-balance/`,
      {
        headers: {
          'Authorization': `Bearer ${getAuthToken()}`,
        },
      }
    );
    return response.data;
  } catch (error) {
    if (error.response) {
      // Handle API error response
      throw new Error(error.response.data.message || error.response.data.error);
    }
    throw error;
  }
}
```

### React Hook Example
```typescript
import { useState } from 'react';

function useExtractBeginningBalance() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  const extractBalance = async (bankLedgerEntryId: number) => {
    setLoading(true);
    setError(null);
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(
        `/api/bank-ledger-entries/${bankLedgerEntryId}/extract-beginning-balance/`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || errorData.error);
      }

      const result = await response.json();
      setData(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { extractBalance, loading, error, data };
}
```

---

## Notes

1. **Bank Configuration Required**: Before using this endpoint, ensure the bank has a `beginning_balance_label` configured in the admin panel (e.g., "SOLDE DEBUT PERIODE").

2. **Case-Insensitive Search**: The search for the beginning balance label is case-insensitive. "SOLDE DEBUT PERIODE" will match "solde debut periode" or "Solde Debut Periode".

3. **Balance Extraction Priority**: 
   - First, tries to extract from the `debit` column
   - If debit is null/zero, falls back to the `amount` column
   - The `source` field in the response indicates which field was used

4. **Latest Entry Fallback**: If no `bank_ledger_entry_id` is provided, the API automatically uses the most recently created bank ledger entry.

5. **Transaction Matching**: The API searches for transactions where the label **contains** the configured text (not exact match). For example, if the label is "SOLDE DEBUT PERIODE", it will match "SOLDE DEBUT PERIODE 01/01/2025".

---

## Related Endpoints

- **Preprocess Bank Ledger Entry**: `POST /api/bank-ledger-entries/{id}/preprocess/`
- **Get Bank Ledger Entries**: `GET /api/bank-ledger-entries/`
- **Get Banks**: `GET /api/banks/`

---

## Version
- **API Version**: 1.0
- **Last Updated**: 2025-01-XX








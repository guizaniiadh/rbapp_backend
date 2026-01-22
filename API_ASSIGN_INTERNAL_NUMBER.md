# API: Assign Internal Number to Bank Transactions

## Endpoint
`POST /api/{bank_code}/assign-internal-number/`

**Example**: `/api/1/assign-internal-number/` (for bank code 1, e.g., BT)

## Overview
Assigns the same `internal_number` to multiple selected bank transactions. Works with both origine (`type='origine'`) and non-origine transactions.

## Request Format

### Headers
```
Content-Type: application/json
Authorization: Bearer <token>
```

### Request Body
```json
{
  "bank_transaction_ids": [123, 456, 789, 101],
  "internal_number": "INT-2024-001"
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_transaction_ids` | Array of Integers | **Yes** | List of bank transaction IDs to update |
| `internal_number` | String | **Yes** | The internal number to assign to all selected transactions |

## Response Format

### Success Response (200 OK)
```json
{
  "message": "Successfully assigned internal_number 'INT-2024-001' to 4 transactions",
  "updated_count": 4,
  "internal_number": "INT-2024-001",
  "updated_transaction_ids": [123, 456, 789, 101],
  "origine_count": 2,
  "non_origine_count": 2,
  "transactions": [
    {
      "id": 123,
      "type": "origine",
      "label": "Transaction Label",
      "amount": 1000.500,
      "internal_number": "INT-2024-001",
      "is_origine": true
    },
    {
      "id": 456,
      "type": "COM",
      "label": "Another Transaction",
      "amount": 500.250,
      "internal_number": "INT-2024-001",
      "is_origine": false
    }
  ]
}
```

### Error Responses

#### 400 Bad Request - Missing IDs
```json
{
  "error": "bank_transaction_ids is required and must be a non-empty list"
}
```

#### 400 Bad Request - Missing Internal Number
```json
{
  "error": "internal_number is required"
}
```

#### 404 Not Found - Some IDs Don't Exist
```json
{
  "error": "Some transaction IDs not found: [999, 888]",
  "found_ids": [123, 456],
  "missing_ids": [999, 888]
}
```

## Frontend Usage Example

### TypeScript/JavaScript Example
```typescript
async function assignInternalNumber(
  bankCode: string, 
  transactionIds: number[], 
  internalNumber: string
) {
  const response = await fetch(`/api/${bankCode}/assign-internal-number/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      bank_transaction_ids: transactionIds,
      internal_number: internalNumber
    })
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to assign internal number');
  }
  
  const data = await response.json();
  console.log(`Updated ${data.updated_count} transactions`);
  console.log(`Origine: ${data.origine_count}, Non-origine: ${data.non_origine_count}`);
  
  return data;
}

// Usage example
const selectedTransactionIds = [123, 456, 789];
const newInternalNumber = "INT-2024-001";

assignInternalNumber('1', selectedTransactionIds, newInternalNumber)
  .then(result => {
    console.log('Success:', result.message);
    // Refresh your transaction list or update UI
  })
  .catch(error => {
    console.error('Error:', error.message);
  });
```

### React Component Example
```typescript
import React, { useState } from 'react';

function AssignInternalNumberButton({ selectedTransactions, bankCode, onSuccess }) {
  const [internalNumber, setInternalNumber] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAssign = async () => {
    if (!internalNumber.trim()) {
      setError('Please enter an internal number');
      return;
    }

    if (selectedTransactions.length === 0) {
      setError('Please select at least one transaction');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const transactionIds = selectedTransactions.map(tx => tx.id);
      const response = await fetch(`/api/${bankCode}/assign-internal-number/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          bank_transaction_ids: transactionIds,
          internal_number: internalNumber.trim()
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to assign internal number');
      }

      alert(`Successfully assigned internal number to ${data.updated_count} transactions`);
      setInternalNumber('');
      onSuccess && onSuccess(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={internalNumber}
        onChange={(e) => setInternalNumber(e.target.value)}
        placeholder="Enter internal number"
        disabled={loading}
      />
      <button onClick={handleAssign} disabled={loading || selectedTransactions.length === 0}>
        {loading ? 'Assigning...' : `Assign to ${selectedTransactions.length} transactions`}
      </button>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
```

## Important Notes

1. **Bulk Update**: All selected transactions will receive the same `internal_number`
2. **Mixed Types**: You can select both origine (`type='origine'`) and non-origine transactions in the same request
3. **Validation**: The API validates that all transaction IDs exist before updating
4. **Atomic Operation**: All updates happen in a single database operation
5. **Authentication**: Requires authentication token
6. **Response Details**: The response includes counts of origine vs non-origine transactions updated

## Workflow

1. User selects multiple bank transactions (can be mix of origine and non-origine)
2. User enters an internal number
3. User clicks "Assign Internal Number" button
4. Frontend sends POST request with selected transaction IDs and internal number
5. Backend validates and updates all transactions
6. Frontend receives confirmation and updates UI



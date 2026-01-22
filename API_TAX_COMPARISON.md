# API: Tax Comparison - Getting Customer Tax

## Overview
The Tax Comparison API allows you to retrieve customer tax information. There are two tax values available:
- **`customer_tax`**: Individual tax amount for a specific transaction and tax type
- **`customer_total_tax`**: Aggregated total tax amount (sum of all tax amounts for transactions with the same document number)

## Endpoints

### GET `/api/{bank_code}/tax-comparison/`
Retrieve existing comparison data without triggering a new comparison.

**Example URL**: `/api/1/tax-comparison/` (for bank code 1, e.g., BT)

#### Query Parameters (all optional):
- `customer_transaction_id`: Filter by specific customer transaction ID
- `tax_type`: Filter by tax type (e.g., 'AGIOS', 'COM', etc.)
- `status`: Filter by comparison status ('matched', 'mismatch', 'missing')

#### Response Format:
```json
{
  "count": 10,
  "results": [
    {
      "id": 1,
      "customer_transaction_id": 123,
      "matched_bank_transaction_id": 456,
      "tax_type": "AGIOS",
      "customer_tax": "5.500",           // ⭐ Individual tax amount for this transaction
      "bank_tax": "5.500",
      "status": "matched",
      "customer_total_tax": "10.000"     // Aggregated total tax amount
    },
    {
      "id": 2,
      "customer_transaction_id": 123,
      "matched_bank_transaction_id": 456,
      "tax_type": "COM",
      "customer_tax": "2.300",           // ⭐ Individual tax amount for this transaction
      "bank_tax": "2.300",
      "status": "matched",
      "customer_total_tax": "4.600"      // Aggregated total tax amount
    }
  ]
}
```

### POST `/api/{bank_code}/tax-comparison/`
Trigger a new tax comparison process and get results immediately.

**Example URL**: `/api/1/tax-comparison/` (for bank code 1, e.g., BT)

#### Request Body:
```json
{}
```
(No body required - processes all matched customer transactions)

#### Response Format:
```json
{
  "message": "Populated 10 comparison rows.",
  "results": [
    {
      "customer_transaction_id": 123,
      "matched_bank_transaction_id": 456,
      "internal_number": "INT-001",
      "tax_type": "AGIOS",
      "customer_tax": "5.500",           // ⭐ Individual tax amount for this transaction
      "bank_tax": "5.500",
      "status": "matched",
      "customer_total_tax": "10.000"     // Aggregated total tax amount
    }
  ]
}
```

## Key Differences

| Field | Description | Use Case |
|-------|-------------|----------|
| `customer_tax` | **Individual tax amount** for this specific transaction row | Use this when you need the tax amount for a single transaction |
| `customer_total_tax` | **Aggregated total tax amount** (sum of all tax amounts for transactions with the same document number) | Use this when you need the total tax across multiple transactions with the same document number |

## Frontend Usage Examples

### Example 1: Get all comparison data
```typescript
async function getAllComparisons(bankCode: string) {
  const response = await fetch(`/api/${bankCode}/tax-comparison/`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  const data = await response.json();
  
  // Access individual customer tax
  data.results.forEach(comparison => {
    console.log(`Transaction ${comparison.customer_transaction_id}:`);
    console.log(`  Individual tax: ${comparison.customer_tax}`);      // Individual amount
    console.log(`  Total tax: ${comparison.customer_total_tax}`);      // Aggregated amount
  });
  
  return data;
}
```

### Example 2: Get customer tax for a specific transaction
```typescript
async function getCustomerTaxForTransaction(bankCode: string, transactionId: number) {
  const response = await fetch(
    `/api/${bankCode}/tax-comparison/?customer_transaction_id=${transactionId}`,
    {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );
  
  const data = await response.json();
  
  // Get individual customer tax amounts
  const customerTaxes = data.results.map(comp => ({
    taxType: comp.tax_type,
    individualTax: comp.customer_tax,        // ⭐ Individual tax amount
    totalTax: comp.customer_total_tax         // Aggregated total
  }));
  
  return customerTaxes;
}
```

### Example 3: Filter by tax type
```typescript
async function getAGIOSTaxes(bankCode: string) {
  const response = await fetch(
    `/api/${bankCode}/tax-comparison/?tax_type=AGIOS`,
    {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );
  
  const data = await response.json();
  
  // Sum all individual AGIOS taxes
  const totalIndividualAGIOS = data.results.reduce((sum, comp) => {
    return sum + parseFloat(comp.customer_tax || 0);
  }, 0);
  
  return totalIndividualAGIOS;
}
```

### Example 4: Trigger new comparison and get results
```typescript
async function runTaxComparison(bankCode: string) {
  const response = await fetch(`/api/${bankCode}/tax-comparison/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({})
  });
  
  const data = await response.json();
  
  // Access customer_tax (individual) from results
  data.results.forEach(result => {
    console.log(`Individual tax: ${result.customer_tax}`);
    console.log(`Total tax: ${result.customer_total_tax}`);
  });
  
  return data;
}
```

## Important Notes

1. **`customer_tax`** is the field you should use when you need the **individual tax amount** for a specific transaction
2. **`customer_total_tax`** is the **aggregated total** across multiple transactions with the same document number
3. Both fields are returned in all responses (GET and POST)
4. Values are formatted as strings with 3 decimal places (e.g., "5.500")
5. Use GET endpoint to retrieve existing data without triggering a new comparison
6. Use POST endpoint to trigger a new comparison process

## Summary

**To get customer tax (individual amount):**
- Use the `customer_tax` field from either GET or POST endpoint responses
- This gives you the tax amount for a specific transaction row

**To get customer total tax (aggregated amount):**
- Use the `customer_total_tax` field from either GET or POST endpoint responses
- This gives you the total tax across all transactions with the same document number



# API: Extract Customer Taxes

## Endpoint
`POST /api/extract-customer-taxes/`

## Overview
Extracts tax rows for customer transactions based on conventions and tax rules. **Now requires `company_id` and `bank_id` to identify the specific convention.**

## Changes from Previous Version

### ⚠️ Breaking Changes
1. **REQUIRED**: `company_id` and `bank_id` must now be included in the request
2. The API no longer automatically determines the bank from transaction account numbers
3. Convention is now identified by the unique company+bank combination

## Request Format

### Headers
```
Content-Type: application/json
Authorization: Bearer <token>  # If authentication is required
```

### Request Body

```json
{
  "company_id": 1,              // REQUIRED: Integer - Company ID
  "bank_id": 2,                 // REQUIRED: Integer - Bank ID
  "transactions": [             // OPTIONAL: Array of transaction objects
    {
      "id": 123,
      "payment_type": "FRS EFFET",
      "payment_status": 5,
      "document_number": "DOC123",
      // ... other transaction fields
    }
  ]
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `company_id` | Integer | **Yes** | The ID of the company |
| `bank_id` | Integer | **Yes** | The ID of the bank |
| `transactions` | Array | No | Array of transaction objects. If not provided, all `RecoCustomerTransaction` records will be processed |

### Transaction Object Requirements

Each transaction in the `transactions` array should have:
- `payment_type` (String): Payment class code (e.g., "FRS EFFET", "CLT EFFET")
- `payment_status` (Integer/Object): Payment status ID or object
- `document_number` (String): Document reference number
- Other transaction fields as needed

## Response Format

### Success Response (200 OK)

```json
{
  "extracted_taxes": [
    {
      "tax_name": "com",
      "value": "0.300",
      "type": "flat",
      "bank": "Bank Name",
      "transaction_reference": "DOC123",
      "convention": "Convention Name",
      "tax_rule": 5
    },
    {
      "tax_name": "TVA",
      "value": "0.057",
      "type": "formula",
      "bank": "Bank Name",
      "transaction_reference": "DOC123",
      "convention": "Convention Name",
      "tax_rule": 6
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `extracted_taxes` | Array | List of extracted tax calculations |
| `extracted_taxes[].tax_name` | String | Name of the tax type (e.g., "com", "TVA", "agios") |
| `extracted_taxes[].value` | String | Calculated tax value (formatted to 3 decimal places) |
| `extracted_taxes[].type` | String | Calculation type: "flat" or "formula" |
| `extracted_taxes[].bank` | String | Bank name |
| `extracted_taxes[].transaction_reference` | String | Document number of the transaction |
| `extracted_taxes[].convention` | String | Convention name used |
| `extracted_taxes[].tax_rule` | Integer | ID of the tax rule applied |

### Error Responses

#### 400 Bad Request - Missing Required Parameters
```json
{
  "error": "company_id and bank_id are required in the request data"
}
```

#### 404 Not Found - No Active Convention
```json
{
  "error": "No active convention found for company_id=1 and bank_id=2"
}
```

#### Error in Tax Calculation
```json
{
  "extracted_taxes": [
    {
      "error": "Missing variables: var1, var2",
      "bank": "Bank Name",
      "transaction_reference": "DOC123",
      "convention": "Convention Name",
      "tax_rule": 5
    }
  ]
}
```

## How It Works

1. **Convention Lookup**: The API finds the active convention for the specified `company_id` and `bank_id` combination
2. **Transaction Filtering**: For each transaction, the API filters tax rules that match:
   - The transaction's `payment_type` (payment class code)
   - The transaction's `payment_status`
   - The specific convention
3. **Tax Calculation**: 
   - **Flat taxes**: Uses the rate directly from the tax rule
   - **Formula taxes**: Evaluates the formula using variables from other tax rules, convention parameters, or transaction fields
4. **Deduplication**: If multiple tax rules with the same `tax_type` exist for the same payment_class and payment_status, only the most recent one is used
5. **Storage**: Creates `CustomerTaxRow` records in the database for each calculated tax

## Important Notes

1. **Convention Uniqueness**: Each convention is unique per company+bank combination. The API will use the most recent active convention if multiple exist.

2. **Payment Class & Status Matching**: Tax rules are only applied when they match BOTH:
   - The transaction's `payment_type` (payment class code)
   - The transaction's `payment_status`

3. **Transaction Requirements**: Transactions without `payment_type` or `payment_status` will be skipped.

4. **Tax Rule Deduplication**: If multiple tax rules with the same `tax_type` exist for the same payment_class and payment_status combination, only the most recent one (by ID) is used.

## Example Usage

### JavaScript/TypeScript Example

```typescript
async function extractCustomerTaxes(companyId: number, bankId: number, transactions?: any[]) {
  const response = await fetch('/api/extract-customer-taxes/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      company_id: companyId,
      bank_id: bankId,
      transactions: transactions // Optional
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to extract taxes');
  }

  const data = await response.json();
  return data.extracted_taxes;
}

// Usage
const taxes = await extractCustomerTaxes(1, 2, [
  {
    id: 123,
    payment_type: "FRS EFFET",
    payment_status: 5,
    document_number: "DOC123"
  }
]);
```

### cURL Example

```bash
curl -X POST http://localhost:8000/api/extract-customer-taxes/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "company_id": 1,
    "bank_id": 2,
    "transactions": [
      {
        "id": 123,
        "payment_type": "FRS EFFET",
        "payment_status": 5,
        "document_number": "DOC123"
      }
    ]
  }'
```

## Migration Guide

If you're updating from the previous version:

1. **Add `company_id` and `bank_id`** to all API calls
2. **Remove any logic** that was determining the bank from account numbers
3. **Ensure** the company and bank IDs are available in your frontend context
4. **Handle the new error responses** for missing parameters or conventions

## Testing

Before deploying, test with:
- Valid company_id and bank_id with active convention
- Invalid company_id or bank_id
- Missing company_id or bank_id
- Transactions with different payment_class and payment_status combinations
- Transactions without payment_class or payment_status


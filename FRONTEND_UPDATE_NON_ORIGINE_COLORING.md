# Frontend Update: Coloring Non-Origin Transactions in Groups

## Overview
A new field `is_non_origine_in_group` has been added to identify non-origin transactions within each transaction group (same `internal_number`).

## Field Description
- **Field Name**: `is_non_origine_in_group`
- **Type**: Boolean (`true`/`false`)
- **Purpose**: Identifies non-origin transactions that belong to a group with an origin transaction

### When `is_non_origine_in_group = true`:
- Transaction is NOT an origin (`is_origine = false`)
- Transaction has an `internal_number` (part of a group)
- The group has an origin transaction

### When `is_non_origine_in_group = false`:
- Transaction IS an origin (`is_origine = true`)
- Transaction has no `internal_number` (not grouped)
- Non-origin transaction in a group without an origin (orphaned)

## API Response
The field is included in:
- `BTSortedRecoBankTransactionsView` - `/api/banks/bt/sorted-transactions/?bank_code=1`
- `STBSortedRecoBankTransactionsView` - `/api/banks/stb/sorted-transactions/?bank_code=2`

Both endpoints return transactions with the new field in the response.

## Frontend Implementation Examples

### React/TypeScript Example
```typescript
interface Transaction {
  id: number;
  internal_number: string;
  is_origine: boolean;
  is_non_origine_in_group: boolean;
  group_size: number;
  // ... other fields
}

// In your component
const TransactionRow = ({ transaction }: { transaction: Transaction }) => {
  const getRowClassName = () => {
    if (transaction.is_origine) {
      return 'transaction-origin';
    }
    if (transaction.is_non_origine_in_group) {
      return 'transaction-non-origin-group'; // Apply your styling here
    }
    return 'transaction-default';
  };

  return (
    <tr className={getRowClassName()}>
      {/* Your transaction row content */}
    </tr>
  );
};
```

### CSS Styling Example
```css
/* Origin transaction */
.transaction-origin {
  background-color: #e3f2fd; /* Light blue */
  font-weight: bold;
}

/* Non-origin transactions within a group */
.transaction-non-origin-group {
  background-color: #fff3e0; /* Light orange */
  padding-left: 20px; /* Indent to show hierarchy */
}

/* Default transaction */
.transaction-default {
  background-color: #ffffff;
}
```

### JavaScript/Vanilla JS Example
```javascript
// When rendering transactions from API
transactions.forEach(transaction => {
  const row = document.createElement('tr');
  
  // Apply styling based on transaction type
  if (transaction.is_non_origine_in_group) {
    row.classList.add('non-origin-in-group');
    row.style.backgroundColor = '#fff3e0';
    row.style.paddingLeft = '20px'; // Indent to show it's part of a group
  } else if (transaction.is_origine) {
    row.classList.add('origin-transaction');
    row.style.backgroundColor = '#e3f2fd';
    row.style.fontWeight = 'bold';
  }
  
  // Add row to table
  tableBody.appendChild(row);
});
```

### Vue.js Example
```vue
<template>
  <tr 
    :class="{
      'origin-transaction': transaction.is_origine,
      'non-origin-in-group': transaction.is_non_origine_in_group
    }"
  >
    <!-- Transaction content -->
  </tr>
</template>

<script>
export default {
  props: {
    transaction: {
      type: Object,
      required: true
    }
  }
}
</script>

<style scoped>
.origin-transaction {
  background-color: #e3f2fd;
  font-weight: bold;
}

.non-origin-in-group {
  background-color: #fff3e0;
  padding-left: 20px;
}
</style>
```

## CSV Usage
When working with CSV exports, filter non-origin transactions in groups:

```python
import pandas as pd

df = pd.read_csv('sorted_transactions_combined_1_*.csv')

# Filter non-origin transactions within groups
non_origin_in_groups = df[df['is_non_origine_in_group'] == True]

# Example: Color rows in Excel/Google Sheets
# Use conditional formatting: =$T2=TRUE (assuming column T is is_non_origine_in_group)
```

## Visual Hierarchy Example
```
┌─────────────────────────────────────┐
│ Origin Transaction (is_origine=true)  │ ← Bold, blue background
├─────────────────────────────────────┤
│   Non-origin 1 (is_non_origine=true)│ ← Indented, orange background
│   Non-origin 2 (is_non_origine=true)│ ← Indented, orange background
└─────────────────────────────────────┘
```

## Notes
- The field will be available in new CSV exports after regenerating the data
- Use this field in combination with `internal_number` to group related transactions visually
- Non-origin transactions inherit coloring from their origin transaction's match status via `should_be_colored`




